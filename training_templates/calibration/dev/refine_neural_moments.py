"""Neural moment head training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cached_data import load_data
from refine_temporal_filter import load_or_build_anchor, ANCHOR_CACHE
from refine_neural_interval import splits, proxy, TIME_SCORE
from refine_joint_grid import vector_errors

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'dev/cache/neural_moments_head'
REPORT = ROOT/'dev/reports/neural_moments_head.json'
P = 22


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(P,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        z=self.net(x)
        return .15*torch.tanh(z[...,0]), .5*torch.tanh(z[...,1])


def smooth(x):
    return F.avg_pool2d(F.pad(x.permute(2,0,1)[None],(1,1,1,1),mode='replicate'),3,1)[0].permute(1,2,0)


def stream(data,anchor):
    std=data.std_target[0,...,:2]
    mean=data.mean_target[0,...,:2]
    channel_scale=torch.tensor([.1,.02])
    lag_ratio=lag_mean=ratio_state=mean_state=None
    old_pred=old_target=None
    for i in range(len(data.targets)):
        if data.first[i]:
            lag_ratio=lag_mean=ratio_state=mean_state=None
            old_pred=old_target=None
        if old_target is not None:
            lag_ratio=(torch.log(old_target.var(0,correction=0)+1e-9)-torch.log(old_pred.var(0,correction=0)+1e-9)).clamp(-2,2)
            lag_mean=(old_target-old_pred).mean(0)/channel_scale
            ratio_state=lag_ratio if ratio_state is None else .8*ratio_state+.2*lag_ratio
            mean_state=lag_mean if mean_state is None else .8*mean_state+.2*lag_mean
        pred=anchor['prediction_norm_uv'][i]*std+mean
        inp=(data.inputs[i:i+1]*data.std_input+data.mean_input)[0,...,:2]
        pm,pv=pred.mean(0),pred.var(0,correction=0)
        im,iv=inp.mean(0),inp.var(0,correction=0)
        ps=(pv+1e-9).sqrt()
        zeros=torch.zeros_like(pm)
        xv=torch.linspace(-1,1,64)[None,:,None].expand_as(pm)
        yv=torch.linspace(-1,1,32)[:,None,None].expand_as(pm)
        cv=torch.tensor([-1.,1.])[None,None].expand_as(pm)
        values=[pm/channel_scale,ps/channel_scale,(im-pm)/channel_scale,
                (inp[-1]-pm)/channel_scale,(inp[-5:].mean(0)-inp[:5].mean(0))/channel_scale,
                torch.log((iv+1e-9)/(pv+1e-9)),
                torch.log((inp[:10].var(0,correction=0)+1e-9)/(pv+1e-9)),
                torch.log((inp[-10:].var(0,correction=0)+1e-9)/(pv+1e-9)),
                (inp[1:]-inp[:-1]).square().mean(0).sqrt()/ps,
                torch.log((smooth(iv)+1e-9)/(pv+1e-9)),
                (smooth(im)-im)/channel_scale,
                zeros if lag_ratio is None else lag_ratio,
                zeros if ratio_state is None else ratio_state,
                zeros if lag_mean is None else lag_mean,
                zeros if mean_state is None else mean_state,
                (inp==0).float().mean(0),
                smooth((inp==0).float().mean(0)),
                pv.flip(-1)/(pv.sum(-1,keepdim=True)+1e-9),xv,yv,cv,
                torch.ones_like(pm)*float(old_pred is not None)]
        features=torch.stack(values,-1).clamp(-8,8)
        assert features.shape==(32,64,2,P)
        target=data.targets[i,...,:2]*std+mean
        yield i,features,pred,target
        old_pred,old_target=pred,target


def prepare(args):
    WORK.mkdir(parents=True,exist_ok=True)
    data=load_data()
    anchor=load_or_build_anchor(data,ANCHOR_CACHE,False,0)
    train,held=splits(data.trajectory_ids)
    buffers={}
    for i,x,p,t in stream(data,anchor):
        name=data.simulation_ids[i]
        pm,tm=p.mean(0),t.mean(0)
        pd,td=p-pm,t-tm
        pv,tv=pd.square().mean(0),td.square().mean(0)
        cov=(pd*td).mean(0)
        norm_rel=t.square().sum(-1).mean().clamp_min(1e-8)
        norm_tke=(.5*tv.sum(-1)).square().mean().clamp_min(1e-14)
        row=dict(x=x.reshape(-1,2,P),delta=(tm-pm).reshape(-1,2),
                 pv=pv.reshape(-1,2),tv=tv.reshape(-1,2),cov=cov.reshape(-1,2),
                 norm_rel=norm_rel.expand(2048),norm_tke=norm_tke.expand(2048))
        buffers.setdefault(name,[]).append(row)
        if i+1==len(data.targets) or data.simulation_ids[i+1]!=name:
            buf=buffers.pop(name)
            torch.save({k:torch.cat([v[k] for v in buf]) for k in row},WORK/f'{Path(name).stem}.pt')
            print(f'moments prepared {i+1}/{len(data.targets)}',flush=True)
    (WORK/'manifest.json').write_text(json.dumps(dict(train=train,held_out=held,features=P),indent=2)+'\n')


def train(args):
    meta=json.loads((WORK/'manifest.json').read_text())
    payloads=[torch.load(WORK/f'{Path(n).stem}.pt',weights_only=True) for n in meta['train']]
    vals={k:torch.cat([v[k] for v in payloads]).to(args.device) for k in payloads[0]}
    del payloads
    torch.manual_seed(130906)
    model=Head().to(args.device)
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    channel_scale=torch.tensor([.1,.02],device=args.device)
    for epoch in range(args.epochs):
        order=torch.randperm(len(vals['x']))
        losses=[]
        for ix in order.split(16384):
            x,delta,pv,tv,cov,nr,nk=[vals[k][ix] for k in ('x','delta','pv','tv','cov','norm_rel','norm_tke')]
            mu,logvar=model(x)
            amplitude=torch.exp(.5*logvar)
            err=(mu*channel_scale-delta).square()+amplitude.square()*pv+tv-2*amplitude*cov
            rel_loss=(err.sum(-1)/nr).mean()
            ke=.5*(torch.exp(logvar)*pv-tv).sum(-1)
            tke_loss=(ke.square()/nk).mean()
            loss=args.rel_weight*rel_loss+tke_loss+.001*(mu.square()+logvar.square()).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step()
            losses.append(float(loss.detach()))
        print(f'moments epoch {epoch+1}/{args.epochs} loss={np.mean(losses):.6f}',flush=True)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    **meta,epochs=args.epochs,device=args.device,rel_weight=args.rel_weight),WORK/f'head{args.tag}.pt')


def evaluate(args):
    payload=torch.load(WORK/f'head{args.tag}.pt',weights_only=True)
    model=Head().eval();model.load_state_dict(payload['state_dict'])
    data=load_data();anchor=load_or_build_anchor(data,ANCHOR_CACHE,False,0)
    configs=[('calibrated',0.,0.),('mean_half',.5,0.),('mean_full',1.,0.),
             ('energy_half',0.,.5),('energy_full',0.,1.),('joint_half',.5,.5),('joint_full',1.,1.)]
    nr=len(configs); ntraj=len(data.trajectory_ids)
    errors=np.zeros((nr,ntraj,3));reward=np.zeros((nr,ntraj));windows=np.zeros(ntraj);valid=np.zeros(ntraj)
    with torch.inference_mode():
        for i,x,p,t in stream(data,anchor):
            tr=data.trajectory_index[i];mu,logvar=model(x)
            pm=p.mean(0);dev=p-pm
            bank=torch.stack([p if j==0 else pm+a*mu*torch.tensor([.1,.02])+dev*torch.exp(.5*b*logvar)
                              for j,(_,a,b) in enumerate(configs)])
            e=vector_errors(bank,t[None]).numpy().astype(np.float64)
            errors[:,tr]+=e
            accuracy=(.5/(.5+np.maximum(e,0))*np.array([.5,.3,.2])).sum(-1)
            reward[:,tr]+=float(anchor['fixed_bounds_sps_kernel'][i])*accuracy
            windows[tr]+=1;valid[tr]+=float(anchor['valid_count'][i])
            if (i+1)%200==0:print(f'moments evaluated {i+1}/{len(data.targets)}',flush=True)
    def summary(names):
        ix=[data.trajectory_ids.index(n) for n in names]
        score=100/(1+.5*errors[:,ix].sum(1)/windows[ix].sum())
        sps=100*reward[:,ix].sum(1)/valid[ix].sum()
        final=np.array([proxy([*s,TIME_SCORE,q]) for s,q in zip(score,sps)])
        metrics=np.c_[score,sps,final]
        return dict(names=list(names),metrics=metrics.tolist(),delta=(metrics-metrics[0]).tolist())
    result=dict(evidence='D30 development, train20/eval10',
                columns=['Rel-L2','TKE','MVPE','SPS','fixed_Time_final_proxy'],configs=configs,
                train=summary(payload['train']),held_out=summary(payload['held_out']),
                held_out_by_angle={str(a):summary([n for n in payload['held_out'] if n.endswith(f'_{a}.h5')]) for a in (0,5,10,15,20)},
                trajectories={n:summary([n]) for n in data.trajectory_ids})
    REPORT.with_stem(REPORT.stem+args.tag).write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','train','evaluate'])
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--device',default='cpu')
    p.add_argument('--rel-weight',type=float,default=30.)
    p.add_argument('--tag',default='')
    args=p.parse_args();torch.set_num_threads(2)
    globals()[args.command](args)


if __name__=='__main__':main()
