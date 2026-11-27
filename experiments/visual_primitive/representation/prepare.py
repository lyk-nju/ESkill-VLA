"""Prepare eligible candidate annotations and the historical sentence attention input."""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from evaluate import batches, normalize_maps
from openpi.policies.eskill_dataset import EpisodeReader

out=Path(sys.argv[1])
cfg=json.loads((out/"config.json").read_text())
base=Path(cfg["base"])
def read(path):
    return json.loads(path.read_text())

def dev_scene(row):
    reader=EpisodeReader(row["episode_ref"])
    boxes={}
    for name,item in reader.meta["scene_entities"].items():
        if item["kind"]!="object" or item["part"]!="whole": continue
        seq=item["regions"]["image"]
        i=reader.row(seq,row["frame"])
        region=reader.region(seq,row["frame"],include_mask=True)
        if region is None or region["bbox"] is None or region["mask"] is None: continue
        box=region["bbox"]
        ref=seq["visible_masks"]
        if not reader.array(seq["visible_masks_valid"])[i]: continue
        visible=np.unpackbits(reader._raw(ref["uri"],ref["key"])[i],bitorder="big").sum()
        if min(box[:2])>=2 and max(box[2:])<=254 and visible>=64 and visible>=.5*region["mask"].sum():
            boxes[name]=box.tolist()
    return boxes

dev=read(base/"inputs/dev.json")
test=read(base/"inputs/test.json")
rgb=lambda rs:{r["rgb_sha256"] for r in rs}
picks={s:[r for r in dev+test if r["stage"]=="Pick" and r["split"]==s] for s in ("fit","check","test")}
assert [len(picks[s]) for s in picks]==[408,204,1307]
assert not ({r["parent_id"] for r in picks["fit"]}&{r["parent_id"] for r in picks["check"]})
for s,t in (("fit","check"),("fit","test"),("check","test")): assert not (rgb(picks[s])&rgb(picks[t]))
regions=read(Path(cfg["historical"])/"regions.json")
for split,rows in picks.items():
    folder=out/"protocol"/split
    folder.mkdir(parents=True,exist_ok=True)
    scenes={}
    for r in rows:
        key=r["rgb_sha256"]
        if key not in scenes:
            if split=="test":
                g=regions[key]
                scenes[key]={e:g["boxes"][e] for e in g["present"] if g["eligible"][e]} if g["geometry_ok"] else {}
            else: scenes[key]=dev_scene(r)
        if r["target"] in scenes[key]:
            np.testing.assert_allclose(r["current_bbox"],scenes[key][r["target"]],atol=1e-5)
    (folder/"boxes.json").write_text(json.dumps(scenes))
    (folder/"rows.json").write_text(json.dumps(rows))
    print(split,"rows",len(rows),"candidate-eligible",sum(r["target"] in scenes[r["rgb_sha256"]] and len(scenes[r["rgb_sha256"]])>1 for r in rows),flush=True)
for bucket,source,splits in [("dev",dev,("fit","check")),("test",test,("test",))]:
    files={}; look={}
    for split in splits:
        n=len(picks[split])
        folder=out/"protocol"/split
        files[split]=(np.lib.format.open_memmap(folder/"attention.npy",mode="w+",dtype="float16",shape=(n,256,144)),
                      np.lib.format.open_memmap(folder/"native.npy",mode="w+",dtype="float32",shape=(n,256)))
        look.update({r["id"]:(split,i) for i,r in enumerate(picks[split])})
    seen=set()
    for rs,methods,maps,mass,valid in batches(base,bucket,source):
        j=methods.index("sentence")
        assert valid[:,j].all()
        vision=mass[:,j]; prob=normalize_maps(maps[:,j],vision)
        feat=np.log1p(256*prob*vision[...,None]).reshape(len(rs),144,256).transpose(0,2,1)
        for k,r in enumerate(rs):
            if r["id"] in look:
                split,i=look[r["id"]]
                files[split][0][i]=feat[k];files[split][1][i]=prob[k,12,3];seen.add(r["id"])
    assert seen==set(look)
    for f in files.values():
        for item in f: item.flush()
print("prepared")
