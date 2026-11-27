"""Fit only on E0 Fit, lock a Check winner, and evaluate with that immutable lock."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from evaluate import patch_overlap,decode_maps,write_csv,write
from inputs import role_spans

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def noun(prompt):
    span=role_spans(prompt)["source"]
    if span is None: return None
    word=prompt[span[0]:span[1]].lower()
    return {"bowl":"black bowl","barbecue sauce":"bbq sauce"}.get(word,word)

class Split:
    def __init__(self,out,split):
        protocol=out/"protocol"/split
        features=out/"features"/split
        self.rows=read(protocol/"rows.json")
        assert [r["id"] for r in self.rows]==read(features/"ids.json")
        self.text=torch.from_numpy(np.load(features/"text.npy").astype(np.float32))
        self.features=features;self.protocol=protocol
        self.nouns=[noun(r["prompt"]) for r in self.rows]
        scenes=read(protocol/"boxes.json")
        self.names=[sorted(scenes[r["rgb_sha256"]]) for r in self.rows]
        cap=max(len(n) for n in self.names)
        regions=np.zeros((len(self.rows),cap,256),np.float32)
        labels=np.full(len(self.rows),-1)
        valid=np.zeros(regions.shape[:2],bool)
        for i,(r,names) in enumerate(zip(self.rows,self.names)):
            if not names:continue
            boxes=scenes[r["rgb_sha256"]]
            weights=patch_overlap(np.array([boxes[n] for n in names]))
            regions[i,:len(names)]=weights/weights.sum(1,keepdims=True)
            valid[i,:len(names)]=True
            if len(names)>1 and r["target"] in names: labels[i]=names.index(r["target"])
        self.rois=torch.from_numpy(regions);self.valid=torch.from_numpy(valid);self.labels=torch.from_numpy(labels)
        target=patch_overlap(np.array([r["current_bbox"] for r in self.rows]))
        self.target=torch.from_numpy(target/target.sum(1,keepdims=True))
    def features_for(self,stream):
        folder=self.protocol if stream=="attention" else self.features
        return torch.from_numpy(np.load(folder/(stream+".npy")).astype(np.float32))
    def assessment(self,scores,probs):
        scores=scores.detach().cpu().numpy();probs=probs.detach().cpu().numpy()
        pred=decode_maps(probs,"local3")
        ranks=np.einsum("bkp,bp->bk",self.rois.numpy(),scores)
        assert np.isfinite(pred).all() and np.isfinite(scores).all()
        result=[]
        for i,r in enumerate(self.rows):
            label=int(self.labels[i]); names=self.names[i]; values=ranks[i,:len(names)]
            rank=None
            if label>=0: rank=1+int(np.sum(np.delete(values,label)>=values[label]))
            target=np.array(r["current_bbox"])
            err=float(np.linalg.norm(pred[i]-(target[:2]+target[2:])/2))
            result.append(dict(id=r["id"],group_id=r["group_id"],suite=r["suite"],unit=r["unit"],
                               condition=r["condition"],target=r["target"],prompt=r["prompt"],x=float(pred[i,0]),y=float(pred[i,1]),
                               error=err,hit=int(err<=10),rank=rank,r1=None if rank is None else int(rank==1),
                               r2=None if rank is None else int(rank<=2),candidates=len(names),
                               winner=names[int(np.argmax(values))] if names else None,
                               roi_scores={n:float(v) for n,v in zip(names,values)}))
        return result

class SmallReadout(nn.Module):
    def __init__(self,local):
        super().__init__()
        self.coeff=nn.Linear(64,144,bias=True)
        nn.init.zeros_(self.coeff.weight);nn.init.zeros_(self.coeff.bias)
        self.kernel=nn.Parameter(torch.zeros(1,1,3,3)) if local else None
    def forward(self,x,text):
        scores=(x*self.coeff(text)[:,None,:]).sum(-1)/(144**.5)
        if self.kernel is not None:
            image=F.pad(scores.reshape(-1,1,16,16),(1,1,1,1),mode="replicate")
            scores=scores+F.conv2d(image,self.kernel).flatten(1)
        return scores

def standard(x,mean,sd):
    return ((x-mean)/sd.clamp_min(1e-5)).clamp(-10,10)
def prototypes(x,data):
    # One positive per distinct image/entity. Fit labels are used only here and in losses.
    unique=set(); pooled={};priors={}
    for i,(r,n) in enumerate(zip(data.rows,data.nouns)):
        k=(r["rgb_sha256"],r["target"])
        if k in unique:continue
        unique.add(k)
        assert n is not None
        weights=data.target[i]
        pooled.setdefault(n,[]).append(weights@F.normalize(x[i],dim=-1))
        priors.setdefault(n,[]).append(weights)
    return ({n:F.normalize(torch.stack(v).mean(0),dim=0) for n,v in pooled.items()},
            {n:torch.stack(v).mean(0) for n,v in priors.items()})

def learned(model,x,text):
    with torch.no_grad(): logits=torch.cat([model(v,t) for v,t in zip(x.split(48),text.split(48))])
    return logits,logits.softmax(-1)
def stats(records):
    valid=[r for r in records if r["rank"] is not None]
    return dict(n=len(records),valid=len(valid),r1=sum(r["r1"] for r in valid),r2=sum(r["r2"] for r in valid),
                hits=sum(r["hit"] for r in records))
def save_output(folder,name,res,prob):
    folder.mkdir(exist_ok=True,parents=True)
    with (folder/(name+".jsonl")).open("w") as f:
        for r in res: f.write(json.dumps(r,allow_nan=False)+"\n")
    np.save(folder/(name+".npy"),prob.detach().numpy().astype(np.float16))
def inferred(x,t,data,state,variant):
    if variant=="fit_prototype":
        vec=torch.stack([state["prototypes"][n] for n in data.nouns])
        scores=(F.normalize(x,dim=-1)*vec[:,None]).sum(-1)*10
        return scores,scores.softmax(-1)
    if variant=="noun_position_prior":
        prob=torch.stack([state["priors"][n] for n in data.nouns])
        return prob,prob
    model=SmallReadout(variant=="local3_bilinear")
    model.load_state_dict(state[variant])
    return learned(model,x,t)

def fit(out,cfg):
    if (out/"selection.json").exists(): raise FileExistsError("Use a new run for any change to fitted probes")
    data=Split(out,"fit"); check=Split(out,"check")
    mean_t=data.text.mean(0);std_t=data.text.std(0,unbiased=False)
    t=standard(data.text,mean_t,std_t);tc=standard(check.text,mean_t,std_t)
    metrics=[];losses=[]
    models=out/"models";models.mkdir(exist_ok=True)
    for stream in cfg["streams"]:
        x=data.features_for(stream)
        mu=x.mean((0,1));sd=x.std((0,1),unbiased=False)
        x=standard(x,mu,sd)
        xc=standard(check.features_for(stream),mu,sd)
        proto,priors=prototypes(x,data)
        state=dict(mean=mu,sd=sd,mean_t=mean_t,std_t=std_t,prototypes=proto,priors=priors)
        for variant in ("linear_bilinear","local3_bilinear"):
            torch.manual_seed(cfg["seed"])
            gen=torch.Generator().manual_seed(cfg["seed"])
            model=SmallReadout(variant=="local3_bilinear")
            optim=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
            for epoch in range(cfg["epochs"]):
                total=0
                for idx in torch.randperm(len(x),generator=gen).split(cfg["batch_fit"]):
                    logits=model(x[idx],t[idx])
                    dense=-(data.target[idx]*logits.log_softmax(-1)).sum(-1).mean()
                    valid=data.labels[idx]>=0
                    roi=(data.rois[idx]@logits.unsqueeze(-1)).squeeze(-1).masked_fill(~data.valid[idx],-1e5)
                    loss=dense+(F.cross_entropy(roi[valid],data.labels[idx][valid]) if valid.any() else 0)
                    optim.zero_grad();loss.backward();optim.step()
                    assert torch.isfinite(loss)
                    total+=float(loss.detach())*len(idx)
                losses.append(dict(stream=stream,method=variant,epoch=epoch+1,loss=total/len(x)))
            state[variant]=model.state_dict()
        weights=models/(stream+".pt")
        torch.save(state,weights)
        methods=["fit_prototype","linear_bilinear","local3_bilinear"]
        if stream==cfg["streams"][0]: methods.append("noun_position_prior")
        for v in methods:
            sc,pr=inferred(xc,tc,check,state,v)
            pred=check.assessment(sc,pr)
            name=stream+"_"+v
            metrics.append(dict(name=name,stream=stream,method=v,**stats(pred)))
            save_output(out/"check_predictions",name,pred,pr)
        print("fit/check complete",stream,metrics[-1],flush=True)
    p=torch.from_numpy(np.load(check.protocol/"native.npy"))
    native=check.assessment(p,p)
    metrics.append(dict(name="native_L12H3",stream="native",method="fixed",**stats(native)))
    save_output(out/"check_predictions","native_L12H3",native,p)
    eligible=[r for r in metrics if r["method"] in ("linear_bilinear","local3_bilinear")]
    winner=max(eligible,key=lambda z:(z["r1"]/z["valid"],z["hits"]/z["n"]))
    write_csv(out/"check_table.csv",metrics);write_csv(out/"fit_curves.csv",losses)
    write(out/"selection.json",dict(winner=winner,all=metrics,config_sha256=sha(out/"config.json"),
          fits={s:sha(models/(s+".pt")) for s in cfg["streams"]},fit_ids_sha256=sha(out/"features/fit/ids.json"),
          check_ids_sha256=sha(out/"features/check/ids.json"),selection_uses_test=False))
    print("locked",winner,flush=True)

def test(out,cfg):
    lock=read(out/"selection.json")
    assert sha(out/"config.json")==lock["config_sha256"]
    data=Split(out,"test")
    for stream in cfg["streams"]:
        path=out/"models"/(stream+".pt")
        assert sha(path)==lock["fits"][stream]
        state=torch.load(path,map_location="cpu")
        x=standard(data.features_for(stream),state["mean"],state["sd"])
        t=standard(data.text,state["mean_t"],state["std_t"])
        for row in lock["all"]:
            if row["stream"]!=stream: continue
            scores,p=inferred(x,t,data,state,row["method"])
            save_output(out/"test_predictions",row["name"],data.assessment(scores,p),p)
    p=torch.from_numpy(np.load(data.protocol/"native.npy"))
    save_output(out/"test_predictions","native_L12H3",data.assessment(p,p),p)
    print("evaluated",len(data.rows),flush=True)

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("out",type=Path);cli.add_argument("phase",choices=("fit","test"))
    args=cli.parse_args()
    torch.set_num_threads(3);torch.set_num_interop_threads(1)
    (fit if args.phase=="fit" else test)(args.out,read(args.out/"config.json"))
