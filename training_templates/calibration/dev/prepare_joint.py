"""Joint prediction cache."""
import json
import time
from pathlib import Path
import torch
from cached_data import load_data
from prepare_c45 import as_data
from refine_temporal_filter import anchor_model
from accuracy_wrapper import AccuracyTTTModel
from temporal_interval_head import IntervalContext,image
from accuracy_screen import ROOT,sha

WORK=ROOT/'dev/cache/joint_cache'
FEATURES=ROOT/'dev/cache/temporal_interval_head'


def main():
    torch.set_num_threads(2);WORK.mkdir(parents=True,exist_ok=True)
    source=ROOT/'ridge_compact'
    weights=torch.load(source/'neural_heads.pt',weights_only=True)
    meta=json.loads((FEATURES/'manifest.json').read_text())
    extra=json.loads((ROOT/'dev/cache/c45_neural/manifest.json').read_text())
    manifest={**meta,'status':'BUILDING','format_version':1,'feature_manifest_sha256':sha(FEATURES/'manifest.json'),
        'driver_sha256':sha(Path(__file__)),'purpose':'physical predictions and targets'}
    (WORK/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    start=time.perf_counter();total=0;checked=0
    with torch.inference_mode():
        for job in [None]+sorted(extra['sources']):
            data=load_data() if job is None else as_data(torch.load(str(ROOT/'dev/cache/train_real_c45_shards'/f'{Path(job).stem}.pt'),weights_only=False,mmap=True))
            base,safe=anchor_model(data)
            model=AccuracyTTTModel(safe,weights['interval_state_dict'],weights['moment_state_dict'],weights['width'],weights['response_range'],weights['feature_count'])
            context=IntervalContext();previous=None;rows=[]
            for i in range(len(data.targets)):
                if data.first[i]:
                    model.reset_ttt_state();context.reset();previous=None;local=0
                    cached=torch.load(str(FEATURES/f'{Path(data.simulation_ids[i]).stem}.pt'),weights_only=True,mmap=True)
                old=None if previous is None else (previous*data.std_target+data.mean_target)[0,...,:2]
                context.observe(old)
                pred,info=model.ttt_step(data.inputs[i:i+1],previous)
                std,mean=data.std_target[0,...,:2],data.mean_target[0,...,:2]
                p=pred[0,...,:2]*std+mean
                lo=info['lower'][0,...,:2]*std+mean;hi=info['upper'][0,...,:2]*std+mean
                inp=(data.inputs[i:i+1]*data.std_input+data.mean_input)[0,...,:2]
                x,g=context.features(inp,.5*(lo+hi),.5*(hi-lo))
                assert torch.equal(x,cached['x'][local]) and torch.equal(g,cached['g'][local]),(job,i,'feature drift')
                target=(data.targets[i:i+1]*data.std_target+data.mean_target)[0,...,:2]
                assert torch.equal(image(target-.5*(lo+hi)),cached['residual'][local])
                assert torch.equal(image(target!=0),cached['valid'][local])
                rows.append(dict(prediction=image(p),target=image(target)))
                total+=1;checked+=1;local+=1;previous=data.targets[i:i+1]
                if i+1==len(data.targets) or data.first[i+1]:
                    name=data.simulation_ids[i];dest=WORK/f'{Path(name).stem}.pt';assert not dest.exists(),dest
                    payload={k:torch.stack([r[k] for r in rows]) for k in ('prediction','target')}
                    payload.update(format_version=1,trajectory=name,feature_shard_sha256=sha(FEATURES/dest.name))
                    tmp=dest.with_suffix('.partial');torch.save(payload,tmp);tmp.replace(dest);rows=[]
                    print(f'joint cache {name}, windows={total}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
            assert base.index==len(data.targets)
    manifest.update(status='COMPLETE',windows=total,exact_feature_checks=checked)
    (WORK/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
