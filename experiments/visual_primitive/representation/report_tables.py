"""Evaluate every prespecified representation and compute paired uncertainty for the locked winner."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import sys
from evaluate import write_csv,write,patch_overlap

out=Path(sys.argv[1])
lock=json.loads((out/"selection.json").read_text())
cfg=json.loads((out/"config.json").read_text())
winner=lock["winner"]["name"]
rows=json.loads((out/"protocol/test/rows.json").read_text())
regions=json.loads((out/"protocol/test/boxes.json").read_text())
idx={r["id"]:i for i,r in enumerate(rows)}
base={r["id"]:r for r in map(json.loads,(out/"test_predictions/native_L12H3.jsonl").read_text().splitlines())}

def bins(r):
    root=r["unit"]; cond=r["condition"]; suite=r["suite"]
    return ["all",root,root+"/"+cond,root+"/"+cond+"/"+suite]
def cluster(values,groups):
    if not len(values): return None
    sums=defaultdict(list)
    for v,g in zip(values,groups): sums[g].append(float(v))
    ans={"mean":float(np.mean(values)),"n":len(values),"groups":len(sums)}
    if len(sums)>=3:
        a=np.array([(sum(v),len(v)) for v in sums.values()])
        i=np.random.default_rng(17).integers(len(a),size=(2000,len(a)))
        b=a[i].sum(1);ans["ci95"]=np.percentile(b[:,0]/b[:,1],[2.5,97.5]).tolist()
    return ans
def center(r): return (np.array(r["current_bbox"][:2])+r["current_bbox"][2:])/2
def pair_data(pred):
    pairs=[]
    for r in rows:
        if r["condition"] not in {"other_query","added_query","target_moved"}:continue
        prior=rows[idx[r["paired_id"]]]
        a,b=pred[prior["id"]],pred[r["id"]]
        d=dict(id=r["id"],anchor_id=prior["id"],unit=r["unit"],suite=r["suite"],group_id=r["group_id"],
               both_centers=int(a["hit"] and b["hit"]),both_top1=int(a["r1"]==1 and b["r1"]==1))
        if r["condition"]=="target_moved":
            shift=center(r)-center(prior)
            response=np.array([b["x"]-a["x"],b["y"]-a["y"]])
            d.update(gt_shift=float(np.linalg.norm(shift)),pred_shift=float(np.linalg.norm(response)),
                     shift_error=float(np.linalg.norm(response-shift)),
                     gain_over_static=float(np.linalg.norm(shift)-np.linalg.norm(response-shift)),
                     parallel_ratio=float(response@shift/(shift@shift)))
        else:
            assert prior["rgb_sha256"]==r["rgb_sha256"]
            first,other=prior["target"],r["target"]
            da=a["roi_scores"][first]-a["roi_scores"][other]
            db=b["roi_scores"][other]-b["roi_scores"][first]
            d.update(reversal=int(da>0 and db>0),a_margin=da,b_margin=db)
        pairs.append(d)
    return pairs
base_pairs={r["id"]:r for r in pair_data(base)}
summaries=[];pair_summ=[];by_sample=[];locked_pairs=[];contrasts={};coverage={}
for spec in lock["all"]:
    name=spec["name"]
    folder=out/"test_predictions"
    data=[json.loads(x) for x in (folder/(name+".jsonl")).read_text().splitlines()]
    assert [r["id"] for r in data]==list(idx)
    pr=np.load(folder/(name+".npy"))
    pred={r["id"]:r for r in data}
    conditions=defaultdict(list)
    for i,r in enumerate(data):
        original=rows[i];sc=regions[original["rgb_sha256"]]
        if r["rank"] is not None:
            names=sorted(sc);weights=patch_overlap(np.array([sc[n] for n in names]))
            # An additional common rule for all methods: area-normalized spatial probability.
            prob_density=(weights@pr[i].astype(float))/weights.sum(1)
            j=names.index(r["target"])
            r["density_top1"]=int(prob_density[j]>np.delete(prob_density,j).max())
        else: r["density_top1"]=None
        for k in bins(r):conditions[k].append(r)
        if name==winner:
            by_sample.append(dict(id=r["id"],suite=r["suite"],unit=r["unit"],condition=r["condition"],
                                  target=r["target"],prompt=r["prompt"],probe_winner=r["winner"],native_winner=base[r["id"]]["winner"],
                                  r1=r["r1"],native_r1=base[r["id"]]["r1"],hit=r["hit"],native_hit=base[r["id"]]["hit"],
                                  probe_error=r["error"],native_error=base[r["id"]]["error"],probe_x=r["x"],probe_y=r["y"],
                                  native_x=base[r["id"]]["x"],native_y=base[r["id"]]["y"]))
    for k,group in conditions.items():
        valid=[r for r in group if r["rank"] is not None]
        summaries.append(dict(model=name,subset=k,total=len(group),multicandidate_n=len(valid),r1=sum(r["r1"] for r in valid),
                              r2=sum(r["r2"] for r in valid),density_r1=sum(r["density_top1"] for r in valid),
                              center_hits=sum(r["hit"] for r in group),median_error=float(np.median([r["error"] for r in group]))))
        if name==winner:
            contrasts[k]={}
            for metric,sub in [("hit",group),("r1",valid)]:
                delta=[r[metric]-base[r["id"]][metric] for r in sub]
                contrasts[k][metric]=cluster(delta,[r["group_id"] for r in sub])
                if delta:contrasts[k][metric].update(gained=delta.count(1),lost=delta.count(-1))
    pairs=pair_data(pred)
    scopes=defaultdict(list)
    for p in pairs:
        for k in (p["unit"],p["unit"]+"/"+p["suite"]):scopes[k].append(p)
    for scope,subset in scopes.items():
        ps=dict(model=name,subset=scope,pairs=len(subset),both_top1=sum(p["both_top1"] for p in subset),
                both_centers=sum(p["both_centers"] for p in subset))
        if "reversal" in subset[0]:ps["correct_reversals"]=sum(p["reversal"] for p in subset)
        else:
            for k in ("gt_shift","pred_shift","shift_error","parallel_ratio","gain_over_static"):
                ps[k+"_median"]=float(np.median([p[k] for p in subset]))
        pair_summ.append(ps)
        if name==winner:
            contrasts["pair_"+scope]={k:cluster([p[k]-base_pairs[p["id"]][k] for p in subset],
                                               [p["group_id"] for p in subset])
                                      for k in (["reversal","both_centers"] if "reversal" in subset[0] else ["shift_error","both_centers"])}
    if name==winner: locked_pairs=pairs
write_csv(out/"results_by_condition.csv",summaries)
write_csv(out/"paired_summary.csv",pair_summ)
write_csv(out/"locked_vs_native.csv",by_sample)
write_csv(out/"locked_pairs.csv",locked_pairs)
write(out/"paired_intervals.json",contrasts)
print("summary ready",winner,len(by_sample),len(locked_pairs))
