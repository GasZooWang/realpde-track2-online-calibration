"""Multiscale interval head."""
import torch
from torch import nn
import torch.nn.functional as F
from neural_interval_rich import Context as RichContext, FEATURE_NAMES as RICH_NAMES

EXTRA_NAMES=('input_mad','input_skew','input_kurtosis','input_first_half_std','input_last_half_std',
    'local3_lag_absolute','local7_lag_absolute','local3_ema_absolute','local7_ema_absolute',
    'local3_lag_signed','local7_lag_signed','local7_input_energy','local7_input_mean_disagreement',
    'input_frame_is_zero','local3_input_zero_fraction','local7_input_zero_fraction')
FEATURE_NAMES=(*RICH_NAMES,*EXTRA_NAMES)


def pool(v,k):
    v=v.permute(0,3,1,2)
    return F.avg_pool2d(F.pad(v,(k//2,)*4,mode='replicate'),k,1).permute(0,2,3,1)


class Context(RichContext):
    def features(self,inp,pred,center,half):
        x=super().features(inp,pred,center,half)
        h=half.clamp_min(1e-5)
        im=inp.mean(0,keepdim=True);iv=inp.var(0,correction=0,keepdim=True)
        z=(inp-im)/(iv+1e-9).sqrt()
        ap=torch.zeros_like(center) if self.pointwise is None else self.pointwise.abs()
        ae=torch.zeros_like(center) if self.absolute_state is None else self.absolute_state
        sp=torch.zeros_like(center) if self.pointwise is None else self.pointwise
        iz=(inp==0).float().mean(0,keepdim=True)
        local3=pool(torch.cat([ap,ae,sp],-1),3).split(2,-1)
        local7=pool(torch.cat([ap,ae,sp],-1),7).split(2,-1)
        iv7,im7,iz7=pool(torch.cat([iv,im,iz],-1),7).split(2,-1)
        extra=[(inp-im).abs().mean(0,keepdim=True)/h,z.pow(3).mean(0,keepdim=True),
            z.pow(4).mean(0,keepdim=True),inp[:10].std(0,correction=0,keepdim=True)/h,
            inp[-10:].std(0,correction=0,keepdim=True)/h,
            local3[0]/h,local7[0]/h,local3[1]/h,local7[1]/h,
            local3[2]/h,local7[2]/h,iv7.clamp_min(0).sqrt()/h,
            (im7-center)/h,(inp==0).float(),pool(iz,3),iz7]
        return torch.cat([x,torch.stack([v.expand_as(center).clamp(-8,8) for v in extra],-1)],-1)


class IntervalHead(nn.Module):
    def __init__(self,width=32,feature_count=len(FEATURE_NAMES)):
        super().__init__()
        self.width=width
        self.feature_count=feature_count
        self.network=nn.Sequential(nn.Linear(feature_count,width),nn.SiLU(),
            nn.Linear(width,width),nn.SiLU(),nn.Linear(width,2))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,x,damping=1.):
        raw=self.network(x[...,:self.feature_count])
        return damping*torch.tanh(raw[...,0]),torch.exp(damping*.69314718056*torch.tanh(raw[...,1]))
