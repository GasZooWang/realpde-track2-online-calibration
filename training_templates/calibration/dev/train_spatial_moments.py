"""Spatial moment head training."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from neural_moments import MomentHead
from spatial_moments import SpatialCorrection
from accuracy_screen import ROOT, TIME, proxy, sha

WORK=ROOT/'dev/cache/spatial_moments'
SOURCE=ROOT/'dev/cache/neural_moments_head'


def load(names, teacher, source=SOURCE, c45=False):
    vals={}
    for name in names:
        p=torch.load(str(source/f'{Path(name).stem}.pt'),weights_only=True,mmap=True)
        n=len(p['x'])//2048
        assert len(p['x'])==n*2048 and p['x'].shape[1:]==(2,22)
        assert (p['x'][...,0].abs()<8).all()
        cache=p if c45 else torch.load(str(ROOT/'dev/cache/screen_cache'/f'{Path(name).stem}.pt'),weights_only=True,mmap=True)
        if not c45:assert len(cache['indices'])==n and cache['trajectory']==name
        for k,v in p.items():
            if k in ('kernel','valid'):continue
            shaped=v.reshape(n,32,64,*v.shape[1:])
            if k.startswith('norm_'):shaped=shaped[:,0,0]
            vals.setdefault(k,[]).append(shaped.clone())
        vals.setdefault('kernel',[]).append(torch.as_tensor(cache['kernel'],dtype=torch.float64).clone())
        vals.setdefault('trajectory',[]).append(torch.full((n,),len(vals.get('names',[])),dtype=torch.long))
        vals.setdefault('names',[]).append(name)
    result={k:torch.cat(v) for k,v in vals.items() if k!='names'}
    result['names']=vals['names']
    with torch.inference_mode():
        mu,lv=zip(*[teacher(x) for x in result['x'].split(16)])
        result['mu']=torch.cat(mu);result['lv']=torch.cat(lv)
    return result


def errors(v,ix,mu,lv):
    x,delta,pv,tv,cov,nr,nk=[v[k][ix] for k in ('x','delta','pv','tv','cov','norm_rel','norm_tke')]
    scale=x.new_tensor([.1,.02]);amp=torch.exp(.5*lv)
    me=mu*scale-delta
    mse=(me.square()+amp.square()*pv+tv-2*amp*cov).sum(-1).mean((-2,-1))
    rel=(mse/nr).clamp_min(1e-12).sqrt()
    ke=.5*(torch.exp(lv)*pv-tv).sum(-1)
    tke=(ke.square().mean((-2,-1))/nk).clamp_min(1e-12).sqrt()
    truth=x[...,0]*scale+delta
    probe_err=me[:,8:25:2,[13,21,29,37],:].permute(0,2,1,3).flatten(-2)
    probe_true=truth[:,8:25:2,[13,21,29,37],:].permute(0,2,1,3).flatten(-2)
    mvpe=(torch.linalg.vector_norm(probe_err,dim=-1)/torch.linalg.vector_norm(probe_true,dim=-1).clamp_min(1e-8)).mean(-1)
    return torch.stack([rel,tke,mvpe],-1)


def objective(err,kernel,denominator):
    sc=100/(1+.5*err.mean(0))
    accuracy=(.5/(.5+err)*err.new_tensor([.5,.3,.2])).sum(-1)
    sps=100*(kernel*accuracy).sum()/denominator
    scores=torch.cat([sc,sps.new_tensor([TIME]),sps.reshape(1)])
    return (scores.pow(.6854)*scores.new_tensor([.5,.1,.1,.1,.2])).sum().pow(1/.6854)


@torch.inference_mode()
def evaluate(v,model,counts):
    es=[]
    for ix in torch.arange(len(v['x'])).split(16):
        mu,lv=v['mu'][ix],v['lv'][ix]
        if model is not None:mu,lv=model(v['x'][ix],mu,lv)
        es.append(errors(v,ix,mu,lv).double())
    err=torch.cat(es)
    def group(ix):
        e=err[ix]
        sc=100/(1+.5*e.mean(0))
        acc=(.5/(.5+e)*e.new_tensor([.5,.3,.2])).sum(-1)
        sps=100*(v['kernel'][ix]*acc).sum()/counts[ix].sum()
        return [*sc.tolist(),float(sps),proxy([*sc.tolist(),TIME,float(sps)])]
    return dict(global_metrics=group(torch.arange(len(err))),
        trajectories={n:group(torch.where(v['trajectory']==i)[0]) for i,n in enumerate(v['names'])})


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=40);p.add_argument('--width',type=int,default=32)
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(140907)
    WORK.mkdir(exist_ok=True,parents=True)
    meta=json.loads((SOURCE/'manifest.json').read_text())
    teacher=MomentHead().eval();teacher.load_state_dict(torch.load(SOURCE/'head.pt',weights_only=True)['state_dict'])
    train=load(meta['train'],teacher);evals=load(meta['held_out'],teacher)
    from cached_data import load_data
    data=load_data()
    def count(v):
        parts=[]
        for n in v['names']:
            ix=np.where(np.asarray(data.simulation_ids)==n)[0]
            target=data.targets[ix,...,:2]*data.std_target[...,:2]+data.mean_target[...,:2]
            parts.append((target!=0).sum((1,2,3,4)).double())
        return torch.cat(parts)
    tc,ec=count(train),count(evals)
    baseline=evaluate(evals,None,ec)
    recorded=json.loads((ROOT/'dev/reports/center_screen.json').read_text())['eval10']['metrics'][0]
    drift=max(abs(a-b) for a,b in zip(baseline['global_metrics'],recorded))
    assert drift<2e-5,(drift,baseline,recorded)
    model=SpatialCorrection(args.width)
    init=evaluate(evals,model,ec)
    assert max(abs(a-b) for a,b in zip(init['global_metrics'],baseline['global_metrics']))<1e-8
    print(f'Preflight PASS, baseline drift={drift:.3g}, parameters={sum(p.numel() for p in model.parameters())}',flush=True)
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    history=[];best=0.;begun=time.perf_counter()
    for epoch in range(args.epochs):
        model.train();losses=[]
        for ix in torch.randperm(len(train['x'])).split(16):
            mu,lv=model(train['x'][ix],train['mu'][ix],train['lv'][ix])
            err=errors(train,ix,mu,lv)
            val=objective(err,train['kernel'][ix],tc[ix].sum())
            penalty=.02*((mu-train['mu'][ix]).square()+(lv-train['lv'][ix]).square()).mean()
            loss=-val+penalty
            optim.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
            losses.append(float(loss.detach()))
        print(f'spatial epoch {epoch+1}/{args.epochs}, loss={np.mean(losses):.6f}, elapsed={time.perf_counter()-begun:.1f}s',flush=True)
        if (epoch+1)%5==0 or epoch+1==args.epochs:
            model.eval();ev=evaluate(evals,model,ec)
            delta=np.subtract(ev['global_metrics'],baseline['global_metrics']).tolist()
            td={n:ev['trajectories'][n][-1]-baseline['trajectories'][n][-1] for n in ev['trajectories']}
            record=dict(epoch=epoch+1,eval10=ev,delta=delta,wins=sum(v>0 for v in td.values()),worst=min(td.values()))
            history.append(record)
            if delta[-1]>best:
                best=delta[-1]
                torch.save(dict(state_dict=model.state_dict(),width=args.width,epoch=epoch+1,train=meta['train'],
                    eval10=meta['held_out'],evidence='D30 development, eval10'),WORK/'best.pt')
            report=dict(status='RUNNING' if epoch+1<args.epochs else 'COMPLETE',baseline=baseline,history=history,
                baseline_drift=drift,best_eval10_delta=best,source_sha256=sha(Path(__file__)),
                evidence='D30 development, train20/eval10')
            (ROOT/'dev/reports/spatial_moments.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
