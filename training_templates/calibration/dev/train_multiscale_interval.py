"""Multiscale interval head training."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from train_temporal_interval import load, batch, evaluate, WORK, SIGMA
from canonical_features import CanonicalTemporalHead
from multiscale_interval_head import MultiscaleIntervalHead
from accuracy_screen import ROOT, sha

DEST = ROOT/'dev/cache/multiscale_interval_cache'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--stop-after',type=int,default=None)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--batch',type=int,default=12)
    parser.add_argument('--device',default='mps')
    parser.add_argument('--tag',default='pyramid96')
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(160907)
    DEST.mkdir(parents=True,exist_ok=True)
    meta=json.loads((WORK/'manifest.json').read_text())
    assert set(meta['train50']).isdisjoint(meta['eval25'])
    seed_path=WORK/'best_canonical96.pt';seed=torch.load(seed_path,weights_only=True)
    model=MultiscaleIntervalHead(96);model.initialize_temporal(seed['state_dict'])
    reference=CanonicalTemporalHead(96).eval();reference.load_state_dict(seed['state_dict'])
    train=load(meta['train50']);ev=load(meta['eval25']);rng=np.random.default_rng(160907)
    with torch.inference_mode():
        example=batch(train,[(0,1),(1,3)])
        a=model(example['x'],example['g']);b=reference(example['x'],example['g'])
        assert all(torch.equal(v,w) for v,w in zip(a,b)), 'initialization drift'
    initial=evaluate(reference,ev,meta['eval25'],thin=3)
    cpu=MultiscaleIntervalHead(96).eval();cpu.load_state_dict(model.state_dict())
    best=initial['delta_sps'];history=[];start=time.perf_counter()
    model.to(args.device).train()
    backbone_parameters=[p for n,p in model.named_parameters() if not n.startswith('pyramid.')]
    pyramid_parameters=list(model.pyramid.parameters())
    optimizer=torch.optim.AdamW([dict(params=backbone_parameters,lr=.0003),dict(params=pyramid_parameters,lr=.0007)],weight_decay=.01)
    torch.save(dict(state_dict=cpu.state_dict(),width=96,architecture='pyramid',epoch=0,
        training_names=meta['train50'],evidence=meta['evidence']),DEST/f'best_{args.tag}.pt')
    print(json.dumps(dict(preflight='PASS',seed_exact=True,configuration=vars(args),
        parameters=sum(p.numel() for p in model.parameters()))),flush=True)
    for epoch in range(min(args.epochs, args.stop_after or args.epochs)):
        temp=max(.00008,.00025*.95**epoch);losses=[]
        lr_factor=.4+.6*.5*(1+math.cos(math.pi*epoch/args.epochs))
        for group,lr in zip(optimizer.param_groups,(.0003,.0007)):group['lr']=lr*lr_factor
        for step in range(args.steps):
            tr=rng.integers(0,len(train),size=args.batch)
            ix=[(int(t),int(rng.integers(1,len(train[t]['x'])))) for t in tr]
            d={k:v.to(args.device) for k,v in batch(train,ix).items()}
            shift,scale=model(d['x'],d['g']);h=d['half']*scale;r=d['residual']-d['half']*shift
            covered=torch.sigmoid((h-r)/temp)*torch.sigmoid((h+r)/temp)
            loss=-(covered*torch.exp(-2*h/SIGMA)*d['valid']*d['accuracy'][:,None,None,None]).sum()/d['valid'].sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
        record=dict(epoch=epoch+1,loss=float(np.mean(losses)),temperature=temp,elapsed=time.perf_counter()-start)
        print(json.dumps(record),flush=True)
        if (epoch+1)%5==0 or epoch+1==args.epochs:
            cpu.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()})
            result=evaluate(cpu,ev,meta['eval25'],thin=3);record['eval25']=result;history.append(record)
            if result['delta_sps']>best:
                best=result['delta_sps'];torch.save(dict(state_dict=cpu.state_dict(),width=96,architecture='pyramid',
                    epoch=epoch+1,training_names=meta['train50'],evidence=meta['evidence'],seed_sha256=sha(seed_path),
                    source_sha256=sha(Path(__file__))),DEST/f'best_{args.tag}.pt')
            (ROOT/f'dev/reports/multiscale_{args.tag}_progress.json').write_text(json.dumps(dict(status='RUNNING',history=history),indent=2)+'\n')
    checkpoint=DEST/f'best_{args.tag}.pt';cpu.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict'])
    full=evaluate(cpu,ev,meta['eval25'])
    baseline=evaluate(reference,ev,meta['eval25'])
    report=dict(status='COMPLETE',configuration=vars(args),history=history,eval25=full,temporal_baseline=baseline,
        delta_sps_vs_temporal=full['sps']-baseline['sps'],model_sha256=sha(checkpoint),source_sha256=sha(Path(__file__)),
        evidence=meta['evidence'],fixed_prediction='base')
    (ROOT/f'dev/reports/multiscale_{args.tag}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE')),flush=True)


if __name__=='__main__':main()
