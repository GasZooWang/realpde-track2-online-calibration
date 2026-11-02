"""Spatial moment head."""
import torch
from torch import nn
import torch.nn.functional as F


class SpatialCorrection(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.width=width
        self.net=nn.Sequential(nn.Conv2d(48,width,1),nn.SiLU(),
            nn.Conv2d(width,width,5,padding=2,padding_mode='replicate',groups=width),nn.SiLU(),
            nn.Conv2d(width,width,1),nn.SiLU(),
            nn.Conv2d(width,width,5,padding=4,dilation=2,padding_mode='replicate',groups=width),nn.SiLU(),
            nn.Conv2d(width,4,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,mu,lv):
        h=torch.cat([x.flatten(-2),mu,lv],-1).permute(0,3,1,2)
        out=self.net(h).permute(0,2,3,1)
        dm=.15*torch.tanh(out[...,:2])
        dl=torch.tanh(out[...,2:])
        return mu+dm,lv+dl
