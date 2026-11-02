"""Moment head."""
import torch
from torch import nn
from torch.nn import functional as F
from energy_head import canonical

class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.SiLU(),nn.Conv2d(c,c,3,padding=1))
    def forward(self,x):return F.silu(x+.2*self.net(x))

class CausalMomentHead(nn.Module):
    def __init__(self,width=32,history=False):
        super().__init__();self.width=width;self.history=history
        self.stem=nn.Conv2d(375,width,1)
        self.global_encoder=nn.Sequential(nn.Linear(256,64),nn.SiLU(),nn.Linear(64,2*width))
        self.fine=Block(width);self.down1=nn.Sequential(nn.Conv2d(width,width*2,1),nn.SiLU(),Block(width*2))
        self.down2=nn.Sequential(nn.Conv2d(width*2,width*3,1),nn.SiLU(),Block(width*3),Block(width*3))
        self.up1=nn.Conv2d(width*3,width*2,1);self.merge1=Block(width*2)
        self.up0=nn.Conv2d(width*2,width,1);self.merge0=Block(width)
        self.out=nn.Conv2d(width,3,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,x,g,pm,pv,previous_target):
        units=x.new_tensor([.1,.02])[None,:,None,None]
        if self.history:
            hist=x[:,:40].reshape(-1,20,2,32,64)
            extras=torch.cat([previous_target,pm/units,pv.clamp_min(0).sqrt()/units,hist.mean(1),hist.std(1,correction=0)],1)
        else:
            hist=previous_target.reshape(-1,20,2,32,64)/units[:,None]
            extras=torch.cat([hist.flatten(1,2),pm/units,pv.clamp_min(0).sqrt()/units,hist.mean(1),hist.std(1,correction=0)],1)
        z=torch.cat([canonical(x),extras.clamp(-8,8)],1)
        h=self.stem(z);gamma,beta=self.global_encoder(g).chunk(2,-1)
        a=self.fine(F.silu(h*(1+.1*torch.tanh(gamma[:,:,None,None]))+.1*beta[:,:,None,None]))
        b=self.down1(F.avg_pool2d(a,2));c=self.down2(F.avg_pool2d(b,2))
        b=self.merge1(b+F.interpolate(self.up1(c),size=b.shape[-2:],mode='bilinear',align_corners=False))
        a=self.merge0(a+F.interpolate(self.up0(b),size=a.shape[-2:],mode='bilinear',align_corners=False))
        out=self.out(a)
        return .2*units*torch.tanh(out[:,:2]),1.5*torch.tanh(out[:,2:3]).expand(-1,2,-1,-1)

def apply_moments(p,mu,lv):
    pp=p.reshape(-1,20,2,32,64);pm=pp.mean(1,keepdim=True)
    return (pm+mu[:,None]+(pp-pm)*torch.exp(.5*lv[:,None])).flatten(1,2)
