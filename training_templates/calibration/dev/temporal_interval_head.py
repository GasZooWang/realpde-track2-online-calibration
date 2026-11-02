"""Temporal interval head."""
import torch
from torch import nn

CHANNELS=167
GLOBAL_SIZE=256


def image(value):
    return value.permute(0,3,1,2).flatten(0,1)


class IntervalContext:
    def __init__(self):self.reset()

    def reset(self):
        self.previous_center=None
        self.residual=self.ema_abs=self.initialized=None

    def observe(self,previous_target):
        if previous_target is None:return
        if self.previous_center is None:raise ValueError('revealed target without previous center')
        valid=previous_target!=0
        error=previous_target-self.previous_center
        prior=torch.zeros_like(error) if self.residual is None else self.residual
        self.residual=torch.where(valid,error,prior)
        count=valid.sum(0).clamp_min(1)
        absolute=(error.abs()*valid).sum(0)/count
        seen=valid.any(0)
        if self.ema_abs is None:
            self.ema_abs=torch.zeros_like(absolute);self.initialized=torch.zeros_like(seen)
        updated=torch.where(self.initialized,.8*self.ema_abs+.2*absolute,absolute)
        self.ema_abs=torch.where(seen,updated,self.ema_abs)
        self.initialized|=seen

    def features(self,inp,center,half):
        scale=inp.new_tensor([.1,.02]);has_previous=float(self.previous_center is not None)
        residual=torch.zeros_like(inp) if self.residual is None else self.residual
        absolute=torch.zeros_like(inp[0]) if self.ema_abs is None else self.ema_abs
        yy=torch.linspace(-1,1,32,device=inp.device)[:,None].expand(32,64)
        xx=torch.linspace(-1,1,64,device=inp.device)[None].expand(32,64)
        x=torch.cat([image(inp/scale),image(center/scale),image(torch.log(half.clamp_min(1e-6)/scale)),
            image(residual/scale),(absolute/scale).permute(2,0,1),(inp==0).float().mean(0).permute(2,0,1),
            yy[None],xx[None],torch.full_like(xx[None],has_previous)],0).clamp(-8,8)
        assert x.shape==(CHANNELS,32,64)
        global_context=(inp/scale).reshape(4,5,4,8,8,8,2).mean((1,3,5)).flatten().clamp(-8,8)
        self.previous_center=center.detach().clone()
        return x,global_context


class ResidualBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(width,width,5,padding=2*dilation,dilation=dilation,
            padding_mode='replicate',groups=width),nn.SiLU(),nn.Conv2d(width,width,1),nn.SiLU())

    def forward(self,x):return x+self.net(x)


class TemporalIntervalHead(nn.Module):
    def __init__(self,width=48):
        super().__init__();self.width=width
        self.stem=nn.Conv2d(CHANNELS,width,1)
        self.global_encoder=nn.Sequential(nn.Linear(GLOBAL_SIZE,32),nn.SiLU(),nn.Linear(32,2*width))
        self.blocks=nn.Sequential(ResidualBlock(width,1),ResidualBlock(width,2))
        self.out=nn.Conv2d(width,80,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,x,global_context):
        h=self.stem(x)
        gamma,beta=self.global_encoder(global_context).chunk(2,-1)
        h=torch.nn.functional.silu(h*(1+.1*torch.tanh(gamma[:,:,None,None]))+.1*beta[:,:,None,None])
        raw=self.out(self.blocks(h));shift,logwidth=raw.chunk(2,1)
        return torch.tanh(shift),torch.exp(.69314718056*torch.tanh(logwidth))
