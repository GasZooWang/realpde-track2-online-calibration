"""Canonical interval head training."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from train_temporal_interval import load,batch,evaluate,WORK,SIGMA
from canonical_features import CanonicalTemporalHead
from accuracy_screen import ROOT,sha


def widen(original,width):
    factor=width//original['stem.weight'].shape[0]
    assert factor>=1 and width==original['stem.weight'].shape[0]*factor
    state={k:v.clone() for k,v in original.items()}
    for key,v in original.items():
        if key.startswith('stem.'):
            state[key]=v.repeat(factor,1,1,1) if v.ndim==4 else v.repeat(factor)
        elif key.startswith('global_encoder.2.'):
            a,b=v.chunk(2,0);repeats=[factor]+[1]*(v.ndim-1)
            state[key]=torch.cat([a.repeat(*repeats),b.repeat(*repeats)])
        elif key.startswith('blocks.'):
            if '.net.0.' in key:state[key]=v.repeat(factor,1,1,1) if v.ndim==4 else v.repeat(factor)
            elif '.net.2.' in key:state[key]=v.repeat(factor,factor,1,1)/factor if v.ndim==4 else v.repeat(factor)
        elif key=='out.weight':state[key]=v.repeat(1,factor,1,1)/factor
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=40);p.add_argument('--steps',type=int,default=120)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--width',type=int,default=96);p.add_argument('--tag',default='canonical96')
    p.add_argument('--seed',default='best_full48.pt');p.add_argument('--device',default='mps');args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(150909)
    meta=json.loads((WORK/'manifest.json').read_text());assert set(meta['train50']).isdisjoint(meta['eval25'])
    seed_path=WORK/args.seed;seed=torch.load(seed_path,weights_only=True)
    model=CanonicalTemporalHead(args.width);model.initialize_from_temporal(widen(seed['state_dict'],args.width))
    cpu=CanonicalTemporalHead(args.width).eval();cpu.load_state_dict(model.state_dict())
    train=load(meta['train50']);ev=load(meta['eval25']);rng=np.random.default_rng(150909)
    initial=evaluate(cpu,ev,meta['eval25'],thin=4);best=initial['delta_sps'];history=[];start=time.perf_counter()
    torch.save(dict(state_dict=cpu.state_dict(),width=args.width,architecture='canonical',epoch=0,training_names=meta['train50'],evidence=meta['evidence']),WORK/f'best_{args.tag}.pt')
    with torch.no_grad():model.stem.weight.add_(torch.randn_like(model.stem.weight)*.00005)
    model=model.to(args.device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
    print(json.dumps(dict(preflight='PASS',device=args.device,train50=len(train),eval25=len(ev),parameters=sum(p.numel() for p in model.parameters()))),flush=True)
    for epoch in range(args.epochs):
        temp=max(.00008,.00025*.93**epoch);losses=[]
        for step in range(args.steps):
            tr=rng.integers(0,len(train),size=args.batch)
            ix=[(int(t),int(rng.integers(1,len(train[t]['x'])))) for t in tr]
            d={k:v.to(args.device) for k,v in batch(train,ix).items()}
            shift,scale=model(d['x'],d['g']);h=d['half']*scale;r=d['residual']-d['half']*shift
            covered=torch.sigmoid((h-r)/temp)*torch.sigmoid((h+r)/temp)
            loss=-(covered*torch.exp(-2*h/SIGMA)*d['valid']*d['accuracy'][:,None,None,None]).sum()/d['valid'].sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
            if epoch==0 and step in (0,19):print(f'GPU benchmark step={step+1}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
        record=dict(epoch=epoch+1,loss=float(np.mean(losses)),temperature=temp,elapsed=time.perf_counter()-start)
        print(json.dumps(record),flush=True)
        if (epoch+1)%5==0 or epoch+1==args.epochs:
            cpu.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()});cpu.eval()
            result=evaluate(cpu,ev,meta['eval25'],thin=3);record['eval25']=result;history.append(record)
            if result['delta_sps']>best:
                best=result['delta_sps'];torch.save(dict(state_dict=cpu.state_dict(),width=args.width,architecture='canonical',
                    epoch=epoch+1,training_names=meta['train50'],evidence=meta['evidence'],seed_sha256=sha(seed_path),
                    source_sha256=sha(Path(__file__))),WORK/f'best_{args.tag}.pt')
            (ROOT/f'dev/reports/temporal_interval_{args.tag}_progress.json').write_text(json.dumps(dict(status='RUNNING',history=history),indent=2)+'\n')
    checkpoint=WORK/f'best_{args.tag}.pt'
    if checkpoint.exists():cpu.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict'])
    else:cpu.initialize_from_temporal(widen(seed['state_dict'],args.width))
    full=evaluate(cpu,ev,meta['eval25'])
    groups={key:evaluate(cpu,[ev[meta['eval25'].index(n)] for n in meta[key]],meta[key]) for key in ('d30_eval10','c45_eval15')}
    report=dict(status='COMPLETE',configuration=vars(args),history=history,eval25=full,**groups,seed_sha256=sha(seed_path),
        model_sha256=sha(checkpoint) if checkpoint.exists() else None,source_sha256=sha(Path(__file__)),evidence=meta['evidence'],
        fixed_prediction='base')
    (ROOT/f'dev/reports/temporal_interval_{args.tag}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
