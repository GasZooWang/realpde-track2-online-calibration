"""Phase head."""
import torch
from torch import nn
from multiscale_interval_head import MultiscaleIntervalHead
from energy_head import canonical


class PhaseIntervalHead(MultiscaleIntervalHead):
    def __init__(self,width=96):
        super().__init__(width);self.phase=nn.Conv2d(width+80,40,1)
        nn.init.zeros_(self.phase.weight);nn.init.zeros_(self.phase.bias)

    def latent(self,x,g):
        h=self.features_prepared(canonical(x),g);fine=self.blocks(h);coarse=self.pyramid(h)
        return torch.cat([fine,coarse],1)

    def forward_full_prepared(self,features,g):
        h=self.features_prepared(features,g);fine=self.blocks(h);coarse=self.pyramid(h)
        raw=self.out(fine)+coarse;a,b=raw.chunk(2,1)
        shape=torch.tanh(self.phase(torch.cat([fine,coarse],1)))
        return torch.tanh(a),torch.exp(.69314718056*torch.tanh(b)),shape


def shape_prediction(p,half,shift,shape):
    n,_,h,w=p.shape;pp=p.reshape(n,20,2,h,w);delta=(half*shift).reshape_as(pp)
    pm=pp.mean(1,keepdim=True);pv=pp.var(1,correction=0,keepdim=True);dm=delta.mean(1,keepdim=True)
    q=pp-pm+.5*(delta-dm)+(half*shape).reshape_as(pp);q=q-q.mean(1,keepdim=True);qv=q.var(1,correction=0,keepdim=True)
    q=torch.where((pv>1e-12)&(qv>1e-12),q*(pv/qv.clamp_min(1e-12)).sqrt(),pp-pm)
    return (pm+q+.5*dm).flatten(1,2)
