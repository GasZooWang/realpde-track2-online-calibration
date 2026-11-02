"""Distribution head."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from energy_head import canonical
from temporal_interval_head import ResidualBlock


class DistributionHead(nn.Module):
    def __init__(self,width=96,components=3):
        super().__init__();self.width=width;self.components=components
        self.stem=nn.Conv2d(407,width,1)
        self.global_encoder=nn.Sequential(nn.Linear(256,32),nn.SiLU(),nn.Linear(32,width*2))
        self.blocks=nn.Sequential(ResidualBlock(width,1),ResidualBlock(width,2),ResidualBlock(width,1))
        self.out=nn.Conv2d(width,40*components*3,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,x,g,shift,scale,spatial_slice=None):
        z=torch.cat([canonical(x),shift,scale.log()],1)
        h=self.stem(z);gamma,beta=self.global_encoder(g).chunk(2,-1)
        h=F.silu(h*(1+.1*torch.tanh(gamma[:,:,None,None]))+.1*beta[:,:,None,None])
        h=self.blocks(h)
        if spatial_slice is not None:h=h[:,:,spatial_slice[0],spatial_slice[1]]
        raw=self.out(h).reshape(len(x),40,3,self.components,*h.shape[-2:])
        logits,mu,ls=raw.unbind(2)
        offsets=raw.new_tensor([-.7,0,.7])[None,None,:,None,None]
        mu=(mu+offsets).clamp(-12,12)
        ls=(ls+math.log(.6)).clamp(-3.5,2.5)
        return F.log_softmax(logits,2),mu,ls


def nll(params,y):
    lw,mu,ls=params
    logp=torch.logsumexp(lw-ls-.5*((y[:,:,None]-mu)*torch.exp(-ls)).square()-.5*math.log(2*math.pi),2)
    tail=-.5*(y/12).square()-math.log(12)-.5*math.log(2*math.pi)
    return -torch.logaddexp(logp+math.log(.99),tail+math.log(.01))


def interval_mass(params,lo,hi):
    lw,mu,ls=params;factor=math.sqrt(.5)*torch.exp(-ls)
    return (.5*(torch.erf((hi[:,:,None]-mu)*factor)-torch.erf((lo[:,:,None]-mu)*factor))*lw.exp()).sum(2)
