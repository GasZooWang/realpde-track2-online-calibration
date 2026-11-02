"""Energy head."""
import torch
from torch import nn
from torch.nn import functional as F
from temporal_interval_head import ResidualBlock


def canonical(x):
    n,_,h,w=x.shape
    inp=x[:,:40].reshape(n,20,2,h,w)
    center=x[:,40:80].reshape_as(inp)
    half=x[:,80:120].reshape_as(inp).exp().clamp_min(.0001)
    residual=x[:,120:160].reshape_as(inp)
    mean=inp.mean(1,keepdim=True);std=inp.std(1,correction=0,keepdim=True)
    extra=torch.cat([(inp-mean)/half,(center-mean)/half,residual/half,std.expand_as(half)/half],1)
    return torch.cat([x,extra.reshape(n,160,h,w).clamp(-8,8)],1)


class EnergyHead(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.width=width
        self.stem=nn.Conv2d(333,width,1)
        self.global_encoder=nn.Sequential(nn.Linear(256,32),nn.SiLU(),nn.Linear(32,2*width))
        self.fine=nn.Sequential(ResidualBlock(width,1),ResidualBlock(width,2))
        self.coarse=nn.Sequential(ResidualBlock(width,1),ResidualBlock(width,2))
        self.out=nn.Conv2d(width,4,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,x,g,p):
        return self.forward_prepared(canonical(x),x,g,p)

    def forward_prepared(self,features,x,g,p):
        n,_,h,w=p.shape
        scale=p.new_tensor([.1,.02])[None,None,:,None,None]
        pp=p.reshape(n,20,2,h,w)/scale
        pm=pp.mean(1);ps=pp.std(1,correction=0)
        ins=x[:,:40].reshape_as(pp).std(1,correction=0)
        z=torch.cat([features,pm,ps,(ins/ps.clamp_min(.0001)).clamp(0,8)],1)
        h=self.stem(z);gamma,beta=self.global_encoder(g).chunk(2,-1)
        h=F.silu(h*(1+.1*torch.tanh(gamma[:,:,None,None]))+.1*beta[:,:,None,None])
        z=self.fine(h)+F.interpolate(self.coarse(F.avg_pool2d(h,4)),size=h.shape[-2:],mode='bilinear',align_corners=False)
        mu,lv=self.out(z).chunk(2,1)
        return .1*torch.tanh(mu),1.5*torch.tanh(lv)


def phase_prediction(p,half,shift,alpha=.5):
    n,_,h,w=p.shape
    pp=p.reshape(n,20,2,h,w)
    delta=(half*shift).reshape_as(pp)
    pm=pp.mean(1,keepdim=True);pv=pp.var(1,correction=0,keepdim=True)
    dm=delta.mean(1,keepdim=True)
    q=pp-pm+alpha*(delta-dm);q=q-q.mean(1,keepdim=True);qv=q.var(1,correction=0,keepdim=True)
    q=torch.where((pv>1e-12)&(qv>1e-12),q*(pv/qv.clamp_min(1e-12)).sqrt(),pp-pm)
    return (pm+q+alpha*dm).flatten(1,2)


def apply_energy(p,mu,lv):
    n,_,h,w=p.shape;pp=p.reshape(n,20,2,h,w);pm=pp.mean(1,keepdim=True)
    return (pm+mu[:,None]*p.new_tensor([.1,.02])[None,None,:,None,None]+(pp-pm)*torch.exp(.5*lv[:,None])).flatten(1,2)


def errors(pred,target):
    n,_,h,w=pred.shape;p=pred.reshape(n,20,2,h,w);t=target.reshape_as(p)
    rel=torch.linalg.vector_norm((p-t).flatten(1),dim=1)/torch.linalg.vector_norm(t.flatten(1),dim=1).clamp_min(1e-8)
    pk=.5*(p-p.mean(1,keepdim=True)).square().mean(1).sum(1)
    tk=.5*(t-t.mean(1,keepdim=True)).square().mean(1).sum(1)
    tke=torch.linalg.vector_norm((pk-tk).flatten(1),dim=1)/torch.linalg.vector_norm(tk.flatten(1),dim=1).clamp_min(1e-8)
    mv=[]
    for px in (13,21,29,37):
        a=p[:,:,:,8:25:2,px].mean(1).flatten(1);b=t[:,:,:,8:25:2,px].mean(1).flatten(1)
        mv.append(torch.linalg.vector_norm(a-b,dim=1)/torch.linalg.vector_norm(b,dim=1).clamp_min(1e-8))
    return torch.stack([rel,tke,torch.stack(mv).mean(0)],-1)
