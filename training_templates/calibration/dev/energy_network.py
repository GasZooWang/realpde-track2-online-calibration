"""Energy head."""
import torch
from torch import nn
from torch.nn import functional as F
from temporal_interval_head import ResidualBlock
class DirectEnergyHead(nn.Module):
 def __init__(self,width=32):
  super().__init__();self.stem=nn.Conv2d(78,width,1)
  self.fine=nn.Sequential(ResidualBlock(width,1),ResidualBlock(width,2))
  self.coarse=nn.Sequential(ResidualBlock(width,1),ResidualBlock(width,2))
  self.out=nn.Conv2d(width,1,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,f):
  z=F.silu(self.stem(f));h=self.fine(z)+F.interpolate(self.coarse(F.avg_pool2d(z,4)),size=z.shape[-2:],mode='bilinear',align_corners=False)
  return 4*torch.tanh(self.out(h))
