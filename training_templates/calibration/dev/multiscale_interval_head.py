"""Multiscale interval head."""
import torch
from torch import nn
from torch.nn import functional as F
from canonical_features import CanonicalTemporalHead
from temporal_interval_head import ResidualBlock


class SpatialPyramid(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(width, 64, 1), nn.SiLU(), ResidualBlock(64, 1))
        self.down2 = nn.Sequential(nn.Conv2d(64, 96, 1), nn.SiLU(), ResidualBlock(96, 1))
        self.down3 = nn.Sequential(nn.Conv2d(96, 128, 1), nn.SiLU(), ResidualBlock(128, 1))
        self.up2 = nn.Conv2d(128, 96, 1)
        self.up1 = nn.Conv2d(96, 64, 1)
        self.refine2 = ResidualBlock(96, 1)
        self.refine1 = ResidualBlock(64, 1)
        self.out = nn.Conv2d(64, 80, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h):
        a = self.down1(F.avg_pool2d(h, 2))
        b = self.down2(F.avg_pool2d(a, 2))
        c = self.down3(F.avg_pool2d(b, 2))
        b = self.refine2(b + F.interpolate(self.up2(c), size=b.shape[-2:], mode='bilinear', align_corners=False))
        a = self.refine1(a + F.interpolate(self.up1(b), size=a.shape[-2:], mode='bilinear', align_corners=False))
        return F.interpolate(self.out(a), size=h.shape[-2:], mode='bilinear', align_corners=False)


class MultiscaleIntervalHead(CanonicalTemporalHead):
    def __init__(self, width=96):
        super().__init__(width)
        self.pyramid = SpatialPyramid(width)

    def features(self, x, global_context):
        n, _, height, width = x.shape
        inp = x[:, :40].reshape(n, 20, 2, height, width)
        center = x[:, 40:80].reshape_as(inp)
        half = x[:, 80:120].reshape_as(inp).exp().clamp_min(.0001)
        residual = x[:, 120:160].reshape_as(inp)
        mean = inp.mean(1, keepdim=True)
        std = inp.std(1, correction=0, keepdim=True)
        extra = torch.cat([(inp-mean)/half, (center-mean)/half, residual/half, std.expand_as(half)/half], 1)
        return self.features_prepared(torch.cat([x, extra.reshape(n, 160, height, width).clamp(-8, 8)], 1), global_context)

    def features_prepared(self, features, global_context):
        h = self.stem(features)
        gamma, beta = self.global_encoder(global_context).chunk(2, -1)
        return F.silu(h*(1+.1*torch.tanh(gamma[:, :, None, None])) + .1*beta[:, :, None, None])

    def forward(self, x, global_context):
        h = self.features(x, global_context)
        raw = self.out(self.blocks(h)) + self.pyramid(h)
        shift, logwidth = raw.chunk(2, 1)
        return torch.tanh(shift), torch.exp(.69314718056*torch.tanh(logwidth))

    def forward_prepared(self, features, global_context):
        h = self.features_prepared(features, global_context)
        raw = self.out(self.blocks(h)) + self.pyramid(h)
        shift, logwidth = raw.chunk(2, 1)
        return torch.tanh(shift), torch.exp(.69314718056*torch.tanh(logwidth))

    def initialize_temporal(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith('pyramid.') for k in missing)
