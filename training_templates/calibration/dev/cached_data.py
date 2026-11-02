"""Cached training data."""
import numpy as np
import torch
from refine_learned_interval import ROOT,ReplayData


def load_data():
    cache=ROOT/'dev/cache'
    base=torch.load(str(cache/'train_real_balanced30.pt'),map_location='cpu',weights_only=False,mmap=True)
    inp=torch.load(str(cache/'train_real_balanced30_inputs.pt'),map_location='cpu',weights_only=False,mmap=True)
    assert base['format_version']==2 and inp['format_version']==1
    assert base['simulation_id']==inp['simulation_id']
    assert torch.equal(base['time_id'],inp['time_id'])
    ids=tuple(str(v) for v in base['simulation_id'])
    names=tuple(dict.fromkeys(ids));mapping={n:i for i,n in enumerate(names)}
    assert len(names)==30 and len(ids)==1241 and int(base['trajectory_first'].sum())==30
    for field,payload in (('base_prediction_norm',base),('target_norm',base),('input_norm',inp)):
        value=payload[field]
        assert value.dtype==torch.float32 and value.shape==(1241,20,32,64,3)
    for name in names:
        ix=np.flatnonzero(np.asarray(ids)==name)
        assert np.array_equal(ix,np.arange(ix[0],ix[-1]+1))
        assert torch.equal(base['time_id'][ix],torch.arange(len(ix))*20)
        assert bool(base['trajectory_first'][ix[0]])
    return ReplayData(base=base['base_prediction_norm'],inputs=inp['input_norm'],targets=base['target_norm'],
                      first=base['trajectory_first'].numpy().astype(bool),simulation_ids=ids,trajectory_ids=names,
                      trajectory_index=np.array([mapping[n] for n in ids],dtype=np.int64),
                      mean_input=inp['mean_input'].reshape(1,1,1,1,3),std_input=inp['std_input'].reshape(1,1,1,1,3),
                      mean_target=base['mean_target'].reshape(1,1,1,1,3),std_target=base['std_target'].reshape(1,1,1,1,3))
