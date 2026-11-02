"""Moment features."""
import torch
import torch.nn.functional as F
from torch import nn

FEATURE_COUNT = 22


class MomentHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_COUNT,64),nn.SiLU(),
                                 nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        z=self.net(x)
        return .15*torch.tanh(z[...,0]), .5*torch.tanh(z[...,1])


def smooth(x):
    v=F.pad(x.permute(2,0,1)[None],(1,1,1,1),mode='replicate')
    return F.avg_pool2d(v,3,1)[0].permute(1,2,0)


class MomentContext:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_prediction=None
        self.lag_ratio=self.lag_mean=self.ratio_state=self.mean_state=None

    def observe(self, matured_raw):
        if matured_raw is None:
            return
        if self.previous_prediction is None:
            raise ValueError('matured target without previous safe prediction')
        previous=self.previous_prediction
        channel_scale=previous.new_tensor([.1,.02])
        self.lag_ratio=(torch.log(matured_raw.var(0,correction=0)+1e-9)
                        -torch.log(previous.var(0,correction=0)+1e-9)).clamp(-2,2)
        self.lag_mean=(matured_raw-previous).mean(0)/channel_scale
        self.ratio_state=(self.lag_ratio if self.ratio_state is None else
                          .8*self.ratio_state+.2*self.lag_ratio)
        self.mean_state=(self.lag_mean if self.mean_state is None else
                         .8*self.mean_state+.2*self.lag_mean)

    def features(self,inp,pred):
        channel_scale=pred.new_tensor([.1,.02])
        pm,pv=pred.mean(0),pred.var(0,correction=0)
        im,iv=inp.mean(0),inp.var(0,correction=0)
        ps=(pv+1e-9).sqrt();zeros=torch.zeros_like(pm)
        xv=torch.linspace(-1,1,64,device=pred.device)[None,:,None].expand_as(pm)
        yv=torch.linspace(-1,1,32,device=pred.device)[:,None,None].expand_as(pm)
        cv=pred.new_tensor([-1.,1.])[None,None].expand_as(pm)
        values=[pm/channel_scale,ps/channel_scale,(im-pm)/channel_scale,
                (inp[-1]-pm)/channel_scale,(inp[-5:].mean(0)-inp[:5].mean(0))/channel_scale,
                torch.log((iv+1e-9)/(pv+1e-9)),
                torch.log((inp[:10].var(0,correction=0)+1e-9)/(pv+1e-9)),
                torch.log((inp[-10:].var(0,correction=0)+1e-9)/(pv+1e-9)),
                (inp[1:]-inp[:-1]).square().mean(0).sqrt()/ps,
                torch.log((smooth(iv)+1e-9)/(pv+1e-9)),(smooth(im)-im)/channel_scale,
                zeros if self.lag_ratio is None else self.lag_ratio,
                zeros if self.ratio_state is None else self.ratio_state,
                zeros if self.lag_mean is None else self.lag_mean,
                zeros if self.mean_state is None else self.mean_state,
                (inp==0).float().mean(0),smooth((inp==0).float().mean(0)),
                pv.flip(-1)/(pv.sum(-1,keepdim=True)+1e-9),xv,yv,cv,
                torch.ones_like(pm)*float(self.previous_prediction is not None)]
        self.previous_prediction=pred.detach().clone()
        return torch.stack(values,-1).clamp(-8,8)
