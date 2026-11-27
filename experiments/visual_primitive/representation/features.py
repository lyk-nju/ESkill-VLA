"""Extract frozen features without loading any test labels into the feature readout."""
from contextlib import ExitStack
from pathlib import Path
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from inputs import _annotated_tokens, load_rgb
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config
from openpi.transforms import TokenizePrompt, resize_image

out=Path(sys.argv[1])
cfg=json.loads((out/"config.json").read_text())
base=Path(cfg["base"])
recipe=json.loads((base/"config.json").read_text())
model_cfg=Pi0Config(action_horizon=10,gradient_checkpointing=False,discrete_state_input=False)
torch.set_num_threads(2)
torch.manual_seed(cfg["seed"])
device=torch.device("cuda:0")
print("loading",recipe["base_weights"],torch.cuda.get_device_name(device),flush=True)
model=Pi0(model_cfg)
model.load_base_weights(recipe["base_weights"])
model.to(device,dtype=torch.bfloat16).eval().requires_grad_(False)
assert not any(p.requires_grad for p in model.parameters())
tokenizer=TokenizePrompt(recipe["tokenizer"],model_cfg.max_token_len)
pali=model.paligemma_with_expert.paligemma
rng=torch.Generator().manual_seed(cfg["seed"])
def matrix(a,b):
    return (torch.randn(a,b,generator=rng)/b**.5).to(device)
visual_proj={width:matrix(width,144) for width in (model_cfg.vision_width,model_cfg.vlm_width)}
text_proj=matrix(model_cfg.vlm_width,64)
def reduce(x):
    x=F.normalize(x.float(),dim=-1)
    return (x @ visual_proj[x.shape[-1]]).half().cpu().numpy()
@torch.inference_mode()
def capture(rows):
    seen={}
    def residual_input(key):
        def hook(mod,args):
            seen[key]=reduce(args[0][:,:256])
        return hook
    def tower(mod,args,value):
        seen["vision_tower"]=reduce(value.last_hidden_state)
    texts=[_annotated_tokens(r["prompt"],tokenizer,{}) for r in rows]
    tokens=torch.tensor(np.stack([t["tokens"] for t in texts]),device=device,dtype=torch.long)
    mask=torch.tensor(np.stack([t["token_mask"] for t in texts]),device=device,dtype=torch.bool)
    body=torch.zeros_like(mask)
    for i,t in enumerate(texts):
        body[i,t["body"]]=True
    inputs=dict(images=torch.tensor(np.stack([resize_image(load_rgb(r),224).transpose(2,0,1) for r in rows]),device=device).float()[:,None]/127.5-1,
                image_mask=torch.ones(len(rows),1,dtype=torch.bool,device=device),tokens=tokens,token_mask=mask)
    with ExitStack() as hooks:
        hooks.callback(pali.vision_tower.register_forward_hook(tower).remove)
        for layer in (5,11):
            hooks.callback(pali.language_model.layers[layer+1].input_layernorm.register_forward_pre_hook(residual_input("L"+str(layer))).remove)
        prefix=model.encode_observation(inputs)
    assert prefix.visual_tokens.shape[1]==256 and prefix.visual_mask[:,:256].all() and not prefix.visual_mask[:,256:].any()
    seen["projector"]=reduce(prefix.visual_tokens)
    seen["L17_final"]=reduce(prefix.context[:,:256])
    embeddings=model.paligemma_with_expert.embed_text(tokens).float()
    sentence=(embeddings*body[...,None]).sum(1)/body.sum(1)[:,None]
    seen["text"]=(F.normalize(sentence,dim=-1)@text_proj).half().cpu().numpy()
    return seen
for split in ("fit","check","test"):
    folder=out/"features"/split
    folder.mkdir(exist_ok=True,parents=True)
    data=json.loads((base/"inputs"/("test.json" if split=="test" else "dev.json")).read_text())
    data=[r for r in data if r["split"]==split and r["stage"]=="Pick"]
    files={}
    for k in range(0,len(data),cfg["batch_capture"]):
        rows=data[k:k+cfg["batch_capture"]]
        batch=capture(rows)
        for key,x in batch.items():
            assert np.isfinite(x).all()
            if key not in files:
                files[key]=np.lib.format.open_memmap(folder/(key+".npy"),mode="w+",dtype=x.dtype,shape=(len(data),*x.shape[1:]))
            files[key][k:k+len(rows)]=x
        if k%160==0: print(split,k,len(data),flush=True)
    for f in files.values(): f.flush()
    (folder/"ids.json").write_text(json.dumps([r["id"] for r in data]))
    print("finished",split,len(data),flush=True)
