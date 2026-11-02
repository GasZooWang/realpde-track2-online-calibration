"""Student distillation."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"dev"))
from evaluate import build
from flow_model.calibration import INPUT_MEAN, INPUT_STD
from student import Student

p=argparse.ArgumentParser()
p.add_argument("--epochs",type=int,default=80)
p.add_argument("--width",type=int,default=48)
p.add_argument("--batch",type=int,default=32)
p.add_argument("--augment",action="store_true")
p.add_argument("--resume",type=Path)
args=p.parse_args()
torch.manual_seed(20260916); np.random.seed(20260916)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark=True
device="cuda"
out=ROOT/"dev/backbone/reports"
cache=ROOT/"dev/backbone"/("distill_aug_cache.pt" if args.augment else "distill_cache.pt")
split=json.loads((ROOT/"dev/backbone/split.json").read_text())
if not cache.exists():
    base=build("teacher",device).sps.accuracy.safe.base
    im,ist=[torch.tensor(v,device=device).reshape(1,1,1,1,3) for v in (INPUT_MEAN,INPUT_STD)]
    samples={}
    with torch.inference_mode():
        for group in ("train50","eval25"):
            xs=[];ys=[]
            for name in split[group]:
                uv=torch.from_numpy(np.load(ROOT/"dev/public_data"/(Path(name).stem+".npz"))["velocity"]).to(device)
                raw=torch.cat((uv,torch.zeros_like(uv[...,:1])),-1)
                stride=5 if args.augment and group=="train50" else 20
                for start in range(0,len(raw)-39,stride):
                    scale=float(np.random.uniform(.8,1.2)) if args.augment and group=="train50" and start%20 else 1.
                    x=(raw[start:start+20][None]*scale-im)/ist
                    y=base(x)
                    xs.append(x[...,:2].permute(0,1,4,2,3).flatten(1,2).cpu().half())
                    ys.append(y[...,:2].permute(0,1,4,2,3).flatten(1,2).cpu().half())
            samples[group]=(torch.cat(xs),torch.cat(ys))
            print("cache",group,len(xs),flush=True)
    torch.save(samples,cache)
    del base
else:samples=torch.load(cache,weights_only=True)
x,y=[a.to(device).contiguous(memory_format=torch.channels_last) for a in samples["train50"]]
vx,vy=[a.to(device).contiguous(memory_format=torch.channels_last) for a in samples["eval25"]]
model=Student(args.width).cuda().to(memory_format=torch.channels_last)
if args.resume:model.load_state_dict(torch.load(args.resume,map_location="cpu",weights_only=True)["state_dict"])
tag=f"student_{args.width}"+("_aug" if args.augment else "")
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
scaler=torch.cuda.amp.GradScaler()
best=float("inf"); history=[]; t0=time.perf_counter()
for epoch in range(args.epochs):
    model.train(); sums=[]
    lr=1e-5+(1e-3-1e-5)*.5*(1+math.cos(math.pi*epoch/args.epochs))
    for pg in optimizer.param_groups:pg["lr"]=lr
    order=torch.randperm(len(x),device=device)
    for chunk in order.split(args.batch):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.float16):q=model(x[chunk])
        e=q.float()-y[chunk].float()
        loss=e.square().mean()
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer);scaler.update();sums.append(float(loss.detach()))
    if (epoch+1)%5==0 or epoch==0:
        model.eval();errors=[]
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):
            for a,b in zip(vx.split(args.batch),vy.split(args.batch)):
                errors.append(float((model(a).float()-b.float()).square().mean())*len(a))
        mse=sum(errors)/len(vx)
        row=dict(epoch=epoch+1,train_mse=float(np.mean(sums)),eval_mse=mse,elapsed=time.perf_counter()-t0)
        history.append(row);print(json.dumps({k:row[k] for k in ('epoch','train_mse','elapsed')}),flush=True)
        state=dict(width=args.width,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch+1,eval_mse=mse,split="train50/eval25",objective='CNO teacher distillation')
        torch.save(state,out/f"{tag}_latest.pt")
        if mse<best:
            best=mse;torch.save(state,out/f"{tag}_best.pt")
        (out/f"{tag}_training.json").write_text(json.dumps(history,indent=2)+"\n")
