"""Competition metrics."""
import torch

def errors(p,t):
    e=(p-t).flatten(1).norm(dim=1)/t.flatten(1).norm(dim=1).clamp_min(1e-8)
    pk=.5*p.var(1,correction=0).sum(1)
    tk=.5*t.var(1,correction=0).sum(1)
    ke=(pk-tk).flatten(1).norm(dim=1)/tk.flatten(1).norm(dim=1).clamp_min(1e-8)
    pm=p.mean(1);tm=t.mean(1)
    probes=[]
    for x in (13,21,29,37):
        a=pm[:,:,8:25:2,x].flatten(1);b=tm[:,:,8:25:2,x].flatten(1)
        probes.append((a-b).norm(dim=1)/b.norm(dim=1).clamp_min(1e-8))
    return torch.stack((e,ke,torch.stack(probes).mean(0)),-1)

def record(p,t,lo,hi):
    e=errors(p,t)
    acc=(.5/(.5+e)*e.new_tensor([.5,.3,.2])).sum(-1)
    valid=t!=0
    reward=(((t>=lo)&(t<=hi))*torch.exp(-(hi-lo)/.0563870259)*valid*acc[:,None,None,None,None]).double().sum()
    return dict(windows=len(p),error_sum=e.double().sum(0).cpu().tolist(),reward=float(reward),valid=int(valid.sum()))

def summary(rows,time_score=90.167643):
    import numpy as np
    n=sum(r["windows"] for r in rows)
    e=sum(np.array(r["error_sum"]) for r in rows)/n
    sps=100*sum(r["reward"] for r in rows)/sum(r["valid"] for r in rows)
    scores=[*(100/(1+.5*e)),time_score,sps]
    proxy=float((np.array(scores)**.6854@np.array([.5,.1,.1,.1,.2]))**(1/.6854))
    return dict(scores=scores,proxy=proxy)
