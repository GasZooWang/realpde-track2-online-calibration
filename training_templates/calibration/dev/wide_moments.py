"""Wide moment head."""
import torch
from neural_moments import MomentHead


class WideMomentHead(MomentHead):
    def __init__(self,response_range=1.5):
        super().__init__()
        self.response_range=float(response_range)

    def forward(self,x):
        z=self.net(x)
        return .15*torch.tanh(z[...,0]),self.response_range*torch.tanh(z[...,1]*(.5/self.response_range))
