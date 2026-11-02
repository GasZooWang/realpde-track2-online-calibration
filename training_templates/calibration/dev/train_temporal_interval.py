"""Temporal interval head training."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from temporal_interval_head import TemporalIntervalHead
from accuracy_screen import ROOT,sha

WORK=ROOT/'dev/cache/temporal_interval_head'
SIGMA=.0563870259


def load(names):
    return [torch.load(str(WORK/f'{Path(n).stem}.pt'),weights_only=True,mmap=True) for n in names]


def batch(shards,indices):
    return {k:torch.stack([shards[t][k][i] for t,i in indices]) for k in ('x','g','residual','half','valid','accuracy','first')}


def corrected(model,d):
    shift,scale=model(d['x'],d['g'])
    gate=(~d['first'])[:,None,None,None]
    return shift*gate,1+(scale-1)*gate


@torch.inference_mode()
def evaluate(model,shards,names,thin=1,step=8):
    rewards=np.zeros(len(names));baseline=rewards.copy();valid=rewards.copy();coverage=rewards.copy();width=rewards.copy()
    for t,s in enumerate(shards):
        for indices in torch.arange(0,len(s['x']),thin).split(step):
            d={k:s[k][indices].to(next(model.parameters()).device) for k in ('x','g','residual','half','valid','accuracy','first')}
            shift,scale=corrected(model,d)
            h=d['half']*scale;r=d['residual']-d['half']*shift
            inside=(r.abs()<=h)&d['valid']
            per=(inside*torch.exp(-2*h/SIGMA)).double().sum((1,2,3))
            rewards[t]+=float((per*d['accuracy'].double()).sum())
            baseline[t]+=float((s['kernel'][indices].double()*s['accuracy'][indices].double()).sum())
            valid[t]+=int(d['valid'].sum());coverage[t]+=int(inside.sum());width[t]+=float((2*h*d['valid']).double().sum())
    return dict(names=names,sps=float(100*rewards.sum()/valid.sum()),baseline_sps=float(100*baseline.sum()/valid.sum()),
        delta_sps=float(100*(rewards-baseline).sum()/valid.sum()),coverage=float(coverage.sum()/valid.sum()),
        width=float(width.sum()/valid.sum()),reward=rewards.tolist(),baseline_reward=baseline.tolist(),valid=valid.tolist(),
        trajectories={n:float(100*(rewards[t]-baseline[t])/valid[t]) for t,n in enumerate(names)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=30);p.add_argument('--width',type=int,default=48)
    p.add_argument('--steps',type=int,default=200);p.add_argument('--batch',type=int,default=4);p.add_argument('--tag',default='full48')
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--resume')
    p.add_argument('--temp',type=float,default=.0015);p.add_argument('--min-temp',type=float,default=.00015)
    p.add_argument('--eval-every',type=int,default=3);p.add_argument('--canonical',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(150908)
    meta=json.loads((WORK/'manifest.json').read_text());assert meta['status']=='COMPLETE'
    assert set(meta['train50']).isdisjoint(meta['eval25'])
    train=load(meta['train50']);ev=load(meta['eval25'])
    if args.canonical:
        from canonical_features import CanonicalTemporalHead
        model=CanonicalTemporalHead(args.width)
    else:model=TemporalIntervalHead(args.width)
    if args.resume:
        state=torch.load(args.resume,weights_only=True)['state_dict']
        if args.canonical and state['stem.weight'].shape[1]==167:model.initialize_from_temporal(state)
        else:model.load_state_dict(state)
    model.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    rng=np.random.default_rng(150908);started=time.perf_counter();best=-1e9;history=[]
    model.eval();initial=evaluate(model,ev,meta['eval25'],thin=4)
    if args.resume is None:assert abs(initial['delta_sps'])<2e-5,initial['delta_sps']
    print(f'Preflight PASS; fit50/eval25; elapsed={time.perf_counter()-started:.1f}s',flush=True)
    for epoch in range(args.epochs):
        model.train();losses=[];temp=max(args.min_temp,args.temp*.90**epoch)
        for step in range(args.steps):
            tr=rng.integers(0,len(train),size=args.batch)
            ix=[(int(t),int(rng.integers(1,len(train[t]['x'])))) for t in tr]
            d={k:v.cuda() for k,v in batch(train,ix).items()};shift,scale=corrected(model,d)
            h=d['half']*scale;r=d['residual']-d['half']*shift
            covered=torch.sigmoid((h-r)/temp)*torch.sigmoid((h+r)/temp)
            reward=covered*torch.exp(-2*h/SIGMA)
            weight=d['valid']*d['accuracy'][:,None,None,None]
            loss=-(reward*weight).sum()/d['valid'].sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
            if epoch==0 and step in (0,19):print(f'Benchmark step={step+1}, elapsed={time.perf_counter()-started:.1f}s',flush=True)
        record=dict(epoch=epoch+1,loss=float(np.mean(losses)),temperature=temp,elapsed=time.perf_counter()-started)
        print(json.dumps(record),flush=True)
        if (epoch+1)%args.eval_every==0 or epoch+1==args.epochs:
            model.eval();r=evaluate(model,ev,meta['eval25'],thin=3)
            record['eval25']=r;history.append(record)
            if r['delta_sps']>best:
                best=r['delta_sps'];torch.save(dict(state_dict=model.state_dict(),width=args.width,epoch=epoch+1,architecture='canonical' if args.canonical else 'temporal',
                    training_names=meta['train50'],evidence=meta['evidence'],source_sha256=sha(Path(__file__))),WORK/f'best_{args.tag}.pt')
    model.load_state_dict(torch.load(WORK/f'best_{args.tag}.pt',weights_only=True)['state_dict']);model.eval()
    full=evaluate(model,ev,meta['eval25'])
    groups={key:evaluate(model,[ev[meta['eval25'].index(n)] for n in meta[key]],meta[key]) for key in ('d30_eval10','c45_eval15')}
    report=dict(status='COMPLETE',configuration=vars(args),history=history,eval25=full,**groups,
        cache_manifest_sha256=sha(WORK/'manifest.json'),model_sha256=sha(WORK/f'best_{args.tag}.pt'),source_sha256=sha(Path(__file__)),
        evidence=meta['evidence'],fixed_prediction='base')
    (ROOT/f'dev/reports/temporal_interval_{args.tag}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
