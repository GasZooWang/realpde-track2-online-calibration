"""CNO teacher."""

import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def fuse_backbone_(model):
    model.eval()
    pairs = []
    for block in model.modules():
        for conv_name, norm_name in (
            ("convolution", "batch_norm"),
            ("convolution1", "batch_norm1"),
            ("convolution2", "batch_norm2"),
        ):
            norm = getattr(block, norm_name, None)
            if isinstance(norm, nn.BatchNorm3d):
                conv = getattr(block, conv_name)
                pairs.append((block, conv_name, norm_name, fuse_conv_bn_eval(conv, norm)))
    for block, conv_name, norm_name, fused in pairs:
        setattr(block, conv_name, fused)
        setattr(block, norm_name, nn.Identity())
    return len(pairs)


class InferenceBackbone(nn.Module):

    def __init__(self, model, device, *, half=None):
        super().__init__()
        self.fused_pairs = fuse_backbone_(model)
        self.use_half = torch.device(device).type == "cuda" if half is None else half
        self.channels_last = torch.device(device).type == "cuda"
        self.model = model.to(device).eval().requires_grad_(False)
        if self.use_half:
            self.model.half()
        if self.channels_last:
            for layer in self.model.modules():
                if isinstance(layer, nn.Conv3d):
                    layer.weight = nn.Parameter(
                        layer.weight.detach().contiguous(memory_format=torch.channels_last_3d),
                        requires_grad=False,
                    )

    def forward(self, x):
        z = x.permute(0, 4, 1, 2, 3)
        if self.use_half:
            z = z.to(dtype=torch.float16)
        if self.channels_last:
            z = z.contiguous(memory_format=torch.channels_last_3d)
        return self.model(z).permute(0, 2, 3, 4, 1).float()
