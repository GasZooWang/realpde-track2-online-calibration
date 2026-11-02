"""C45 feature cache."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from cached_data import load_data
from refine_learned_interval import ReplayData
from refine_temporal_filter import anchor_model
from compact_interval import Context,FEATURE_NAMES
from neural_interval_rich import IntervalHead
from neural_moments import MomentHead,MomentContext
from refine_joint_grid import vector_errors
from accuracy_screen import ROOT,sha
from neural_interval import SIGMA

WORK=ROOT/'dev/cache/c45_neural'


def as_data(p):
    n=len(p['simulation_id']);names=tuple(dict.fromkeys(p['simulation_id']));assert len(names)==1
    return ReplayData(base=p['base_prediction_norm'],inputs=p['input_norm'],targets=p['target_norm'],
        first=p['trajectory_first'].numpy().astype(bool),simulation_ids=tuple(p['simulation_id']),trajectory_ids=names,
        trajectory_index=np.zeros(n,dtype=np.int64),**{k:p[v].reshape(1,1,1,1,3) for k,v in
        [('mean_input','mean_input'),('std_input','std_input'),('mean_target','mean_target'),('std_target','std_target')]})


def main():
    torch.set_num_threads(2);WORK.mkdir(exist_ok=True,parents=True)
    (WORK/'interval').mkdir(exist_ok=True);(WORK/'moments').mkdir(exist_ok=True)
    d30=load_data();files=sorted((ROOT/'dev/cache/train_real_c45_shards').glob('*.pt'));assert len(files)==45
    sources={};groups={}
    for file in files:
        p=torch.load(str(file),weights_only=False,mmap=True,map_location='cpu')
        assert p['format_version']==3
        name=file.stem+'.h5';n=len(p['simulation_id'])
        assert name not in d30.trajectory_ids and name!='7575_0.h5'
        assert p['simulation_id']==[name]*n
        assert torch.equal(p['time_id'],torch.arange(n)*20)
        assert bool(p['trajectory_first'][0]) and int(p['trajectory_first'].sum())==1
        for k in ('input_norm','target_norm','base_prediction_norm'):
            assert p[k].shape==(n,20,32,64,3) and p[k].dtype==torch.float32
        for k in ('mean_input','std_input','mean_target','std_target'):
            assert torch.equal(p[k].reshape(-1),getattr(d30,k).reshape(-1)),(name,k)
        groups.setdefault(int(file.stem.rsplit('_',1)[1]),[]).append(name)
        sources[name]=dict(file=str(file),windows=n,bytes=file.stat().st_size)
    assert set(groups)=={0,5,10,15,20} and all(len(v)>=3 for v in groups.values()),groups
    held=[]
    for a in sorted(groups):
        ordered=sorted(groups[a],key=lambda n:int(n.split('_')[0]))
        held.extend(ordered[i] for i in (0,len(ordered)//2,len(ordered)-1))
    held=sorted(held)
    train=sorted(set(sources)-set(held))
    manifest=dict(source='official C45 public caches',train30=train,eval15=held,features=list(FEATURE_NAMES),
        sources=sources,samples_per_window=2048,seed=140907,source_script_sha256=sha(Path(__file__)),
        evidence='C45 development split',
        schema_preflight='PASS',status='BUILDING')
    (WORK/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('C45 preflight passed.',flush=True)
    head=IntervalHead().eval();head.load_state_dict(torch.load(ROOT/'dev/cache/neural_interval_rich/head.pt',weights_only=True)['state_dict'])
    moment=MomentHead().eval();moment.load_state_dict(torch.load(ROOT/'dev/cache/neural_moments_head/head.pt',weights_only=True)['state_dict'])
    gen=torch.Generator().manual_seed(140907);start=time.perf_counter()
    with torch.inference_mode():
        for j,file in enumerate(files):
            p=torch.load(str(file),weights_only=False,mmap=True,map_location='cpu');data=as_data(p)
            base,safe=anchor_model(data);ctx=Context();mctx=MomentContext();prev=None;ir=[];mr=[]
            for i in range(len(data.targets)):
                old=None if prev is None else (prev*data.std_target+data.mean_target)[0,...,:2]
                ctx.observe(old);mctx.observe(old)
                pred,info=safe.ttt_step(data.inputs[i:i+1],prev)
                std,mean=data.std_target[0,...,:2],data.mean_target[0,...,:2]
                q=pred[0,...,:2]*std+mean
                lo=info['lower'][0,...,:2]*std+mean;hi=info['upper'][0,...,:2]*std+mean
                center,half=.5*(lo+hi),.5*(hi-lo)
                inp=(data.inputs[i:i+1]*data.std_input+data.mean_input)[0,...,:2]
                x=ctx.features(inp,q,center,half);mx=mctx.features(inp,q)
                shift,scale=head(x[...,:28]);mu,lv=moment(mx)
                pm,pv=q.mean(0),q.var(0,correction=0)
                q13=pm+mu*q.new_tensor([.1,.02])+(q-pm)*torch.exp(.5*lv)
                nlo=center+half*shift-half*scale;nhi=center+half*shift+half*scale
                target=(data.targets[i:i+1]*data.std_target+data.mean_target)[0,...,:2]
                valid=target!=0
                err=vector_errors(q[None],target[None])[0]
                accuracy=float((.5/(.5+err)*err.new_tensor([.5,.3,.2])).sum())
                ix=torch.randperm(target.numel(),generator=gen)[:2048]
                ir.append(dict(x=x.reshape(-1,len(FEATURE_NAMES))[ix].clone(),residual=(target-center).flatten()[ix],
                    half=half.flatten()[ix],weight=valid.flatten()[ix].float()*accuracy))
                tm,tv=target.mean(0),target.var(0,correction=0);cov=((q-pm)*(target-tm)).mean(0)
                kernel=float((((target>=nlo)&(target<=nhi)&valid)*torch.exp(-(nhi-nlo)/SIGMA)).double().sum())
                mr.append(dict(x=mx.reshape(-1,2,22),delta=(tm-pm).reshape(-1,2),pv=pv.reshape(-1,2),tv=tv.reshape(-1,2),
                    cov=cov.reshape(-1,2),norm_rel=target.square().sum(-1).mean().clamp_min(1e-8).expand(2048),
                    norm_tke=(.5*tv.sum(-1)).square().mean().clamp_min(1e-14).expand(2048),
                    kernel=kernel,valid=int(valid.sum())))
                prev=data.targets[i:i+1]
            assert base.index==len(data.targets)
            ip={k:torch.cat([r[k] for r in ir]) for k in ir[0]}
            mp={k:torch.cat([r[k] for r in mr]) for k in mr[0] if k not in ('kernel','valid')}
            mp.update(kernel=torch.tensor([r['kernel'] for r in mr],dtype=torch.float64),valid=torch.tensor([r['valid'] for r in mr],dtype=torch.float64))
            for kind,obj in [('interval',ip),('moments',mp)]:
                dest=WORK/kind/file.name;assert not dest.exists(),dest
                tmp=dest.with_suffix('.partial');torch.save(obj,tmp);tmp.replace(dest)
            print(f'C45 features {j+1}/45 {file.stem}, windows={len(data.targets)}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
    manifest['status']='COMPLETE';(WORK/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
