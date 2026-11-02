"""Temporal interval cache."""
import json
import time
from pathlib import Path
import torch
from cached_data import load_data
from prepare_c45 import as_data
from refine_temporal_filter import anchor_model
from accuracy_wrapper import AccuracyTTTModel
from temporal_interval_head import IntervalContext,image
from refine_joint_grid import vector_errors
from accuracy_screen import ROOT,sha

WORK=ROOT/'dev/cache/temporal_interval_head'


def main():
    torch.set_num_threads(2);WORK.mkdir(exist_ok=True,parents=True)
    source=ROOT/'ridge_compact'
    for name in ('accuracy_wrapper.py','compact_interval.py','wide_moments.py'):
        assert sha(ROOT/'dev'/name)==sha(source/name),name
    weights=torch.load(source/'neural_heads.pt',weights_only=True)
    meta=json.loads((ROOT/'dev/cache/neural_moments_head/manifest.json').read_text())
    extra=json.loads((ROOT/'dev/cache/c45_neural/manifest.json').read_text())
    train=meta['train']+extra['train30'];evals=meta['held_out']+extra['eval15']
    assert len(train)==50 and len(evals)==25 and set(train).isdisjoint(evals)
    manifest=dict(status='BUILDING',train50=train,eval25=evals,d30_eval10=meta['held_out'],c45_eval15=extra['eval15'],
        evidence='train50 fit, eval25 selection',
        compact_zip_sha256=sha(ROOT/'ridge_compact_heads.zip'),
        feature_source_sha256=sha(ROOT/'dev/temporal_interval_head.py'),driver_sha256=sha(Path(__file__)),format_version=1)
    (WORK/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    jobs=[None]+sorted(extra['sources']);start=time.perf_counter();total=0
    with torch.inference_mode():
        for job in jobs:
            data=load_data() if job is None else as_data(torch.load(str(ROOT/'dev/cache/train_real_c45_shards'/f'{Path(job).stem}.pt'),weights_only=False,mmap=True))
            base,safe=anchor_model(data)
            model=AccuracyTTTModel(safe,weights['interval_state_dict'],weights['moment_state_dict'],
                weights['width'],weights['response_range'],weights['feature_count'])
            context=IntervalContext();previous=None;rows=[]
            for i in range(len(data.targets)):
                if data.first[i]:model.reset_ttt_state();context.reset();previous=None
                old=None if previous is None else (previous*data.std_target+data.mean_target)[0,...,:2]
                context.observe(old)
                pred,info=model.ttt_step(data.inputs[i:i+1],previous)
                std,mean=data.std_target[0,...,:2],data.mean_target[0,...,:2]
                p=pred[0,...,:2]*std+mean
                lo=info['lower'][0,...,:2]*std+mean;hi=info['upper'][0,...,:2]*std+mean
                center=.5*(lo+hi);half=.5*(hi-lo)
                inp=(data.inputs[i:i+1]*data.std_input+data.mean_input)[0,...,:2]
                x,g=context.features(inp,center,half)
                target=(data.targets[i:i+1]*data.std_target+data.mean_target)[0,...,:2]
                valid=target!=0;error=vector_errors(p[None],target[None])[0]
                accuracy=float((.5/(.5+error)*error.new_tensor([.5,.3,.2])).sum())
                kernel=float((((target>=lo)&(target<=hi)&valid)*torch.exp(-(hi-lo)/.0563870259)).double().sum())
                rows.append(dict(x=x,g=g,residual=image(target-center),half=image(half),valid=image(valid),
                    accuracy=accuracy,kernel=kernel,first=bool(data.first[i])))
                previous=data.targets[i:i+1];total+=1
                if i+1==len(data.targets) or data.first[i+1]:
                    name=data.simulation_ids[i];dest=WORK/f'{Path(name).stem}.pt'
                    assert not dest.exists(),dest
                    payload={k:torch.stack([r[k] for r in rows]) for k in ('x','g','residual','half','valid')}
                    payload.update({k:torch.tensor([r[k] for r in rows]) for k in ('accuracy','kernel','first')})
                    payload.update(format_version=1,trajectory=name)
                    tmp=dest.with_suffix('.partial');torch.save(payload,tmp);tmp.replace(dest);rows=[]
                    print(f'temporal cache {name}, total={total}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
            assert base.index==len(data.targets)
    manifest['status']='COMPLETE';manifest['windows']=total
    (WORK/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
