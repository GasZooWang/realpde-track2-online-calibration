"""Extended neural interval head."""
import torch
from torch import nn
import torch.nn.functional as F
from neural_interval import Context as BaseContext, FEATURE_NAMES as BASE_NAMES

FEATURE_NAMES = (*BASE_NAMES,'lag_pointwise','lag_absolute_pointwise',
                 'ema_absolute_pointwise','input_neighbor_delta','prediction_neighbor_delta',
                 'input_spatial_roughness','other_channel_input_std','other_channel_lag')


def spatial_mean(value):
    return F.avg_pool2d(F.pad(value.permute(0,3,1,2),(1,1,1,1),mode='replicate'),3,1).permute(0,2,3,1)


class Context(BaseContext):
    def reset(self):
        super().reset()
        self.pointwise=self.absolute_state=None
        self.point_initialized=None

    def observe(self,matured_raw):
        if matured_raw is not None:
            if self.previous_center is None:
                raise ValueError('matured target without previous center')
            valid=matured_raw!=0
            residual=matured_raw-self.previous_center
            if self.pointwise is None:
                self.pointwise=torch.zeros_like(residual)
                self.absolute_state=torch.zeros_like(residual)
                self.point_initialized=torch.zeros_like(valid)
            self.pointwise=torch.where(valid,residual,self.pointwise)
            updated=torch.where(self.point_initialized,.8*self.absolute_state+.2*residual.abs(),residual.abs())
            self.absolute_state=torch.where(valid,updated,self.absolute_state)
            self.point_initialized|=valid
        super().observe(matured_raw)

    def features(self,input_raw,prediction_raw,center_raw,half_raw):
        baseline=super().features(input_raw,prediction_raw,center_raw,half_raw)
        half=half_raw.clamp_min(1e-5)
        point=torch.zeros_like(center_raw) if self.pointwise is None else self.pointwise
        absolute=torch.zeros_like(center_raw) if self.absolute_state is None else self.absolute_state
        local=spatial_mean(input_raw)
        rough=(spatial_mean(input_raw.square())-local.square()).clamp_min(0).sqrt()
        scale=center_raw.new_tensor([.1,.02])
        input_std=input_raw.std(0,correction=0,keepdim=True)/scale
        lag=torch.zeros_like(center_raw[0]) if self.lag is None else self.lag/scale
        extra=[point/half,point.abs()/half,absolute/half,(local-input_raw)/half,
               (spatial_mean(prediction_raw)-prediction_raw)/half,rough/half,
               input_std.flip(-1).expand_as(center_raw),lag.flip(-1).expand_as(center_raw)]
        return torch.cat([baseline,torch.stack(extra,-1).clamp(-8,8)],dim=-1)


class IntervalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(len(FEATURE_NAMES),64),nn.SiLU(),
                                   nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,features,damping=1.):
        raw=self.network(features)
        return damping*torch.tanh(raw[...,0]),torch.exp(damping*.69314718056*torch.tanh(raw[...,1]))
