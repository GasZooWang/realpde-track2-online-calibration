"""Forecast head."""
import torch
from torch import nn
from torch.nn import functional as F
from energy_head import canonical
from moment_head import CausalMomentHead

class CausalForecastHead(CausalMomentHead):
    def __init__(self,width=32):
        super().__init__(width,True)
        self.pred_stem=nn.Conv2d(40,width,1);nn.init.zeros_(self.pred_stem.weight);nn.init.zeros_(self.pred_stem.bias)
        self.shape_out=nn.Conv2d(width,40,1);nn.init.zeros_(self.shape_out.weight);nn.init.zeros_(self.shape_out.bias)
        self.bound_stem=nn.Conv2d(80,width,1);nn.init.zeros_(self.bound_stem.weight);nn.init.zeros_(self.bound_stem.bias)
        self.bound_out=nn.Conv2d(width,80,1);nn.init.zeros_(self.bound_out.weight);nn.init.zeros_(self.bound_out.bias)

    def forward(self,x,g,p,history,first,shift16,scale16):
        units=p.new_tensor([.1,.02])[None,:,None,None]
        pp=p.reshape(-1,20,2,32,64);pm=pp.mean(1);pv=pp.var(1,correction=0);ps=pv.clamp_min(1e-12).sqrt()
        inp=x[:,:40].reshape_as(pp)
        extras=torch.cat([history,pm/units,ps/units,inp.mean(1),inp.std(1,correction=0)],1)
        z=torch.cat([canonical(x),extras.clamp(-8,8)],1)
        phase=((pp-pm[:,None])/ps[:,None]).clamp(-8,8).flatten(1,2)
        h=self.stem(z)+self.pred_stem(phase)+self.bound_stem(torch.cat([shift16,scale16.log()],1))
        gamma,beta=self.global_encoder(g).chunk(2,-1)
        a=self.fine(F.silu(h*(1+.1*torch.tanh(gamma[:,:,None,None]))+.1*beta[:,:,None,None]))
        b=self.down1(F.avg_pool2d(a,2));c=self.down2(F.avg_pool2d(b,2))
        b=self.merge1(b+F.interpolate(self.up1(c),size=b.shape[-2:],mode='bilinear',align_corners=False))
        a=self.merge0(a+F.interpolate(self.up0(b),size=a.shape[-2:],mode='bilinear',align_corners=False))
        out=self.out(a);mu=.2*units*torch.tanh(out[:,:2]);lv=1.5*torch.tanh(out[:,2:3])
        shape=2*ps[:,None]*torch.tanh(self.shape_out(a).reshape_as(pp))
        wave=pp-pm[:,None]+shape;wave=wave-wave.mean(1,keepdim=True);wv=wave.var(1,correction=0)
        wave=torch.where(((pv>1e-12)&(wv>1e-12))[:,None],wave*(pv.clamp_min(1e-12)/wv.clamp_min(1e-12)).sqrt()[:,None],pp-pm[:,None])
        q=(pm+mu)[:,None]+wave*torch.exp(.5*lv[:,None])
        bsh,bsc=self.bound_out(a).chunk(2,1);gate=(~first)[:,None,None,None]
        return torch.where(first[:,None,None,None],p,q.flatten(1,2)),.5*torch.tanh(bsh)*gate,torch.exp(.69314718056*torch.tanh(bsc)*gate)
