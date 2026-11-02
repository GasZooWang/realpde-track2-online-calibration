"""Official CNO checkpoint loader."""
import torch
from flow_model.cno import CNO3d

def load_baseline(path, device="cpu", **kwargs):
    value=torch.load(path,map_location="cpu",weights_only=False)
    model=CNO3d(in_dim=3,out_dim=3,out_dim_mult=1,in_size=32,N_layers=3)
    model.load_state_dict(value.get("model_state_dict",value))
    return model.to(device).eval(), {"type":"cno"}
