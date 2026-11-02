"""Interval output cache."""
import json,time
from pathlib import Path
import numpy as np
import torch
from train_energy_head import joint_load,JOINT
from train_temporal_interval import corrected,SIGMA
from multiscale_interval_head import MultiscaleIntervalHead
from energy_head import EnergyHead,phase_prediction,apply_energy,errors
from accuracy_screen import ROOT,sha

DEST=ROOT/'dev/cache/distribution_cache'
INTERVAL=ROOT/'dev/cache/multiscale_interval_cache/best_pyramid96.pt'
ENERGY=ROOT/'dev/cache/energy_head_cache/best_energy64.pt'


def main():
    torch.set_num_threads(2);DEST.mkdir(parents=True,exist_ok=True)
    meta=json.loads((JOINT/'manifest.json').read_text());assert meta['status']=='COMPLETE'
    net=MultiscaleIntervalHead(96).eval();net.load_state_dict(torch.load(INTERVAL,weights_only=True)['state_dict'])
    en=EnergyHead(64).eval();en.load_state_dict(torch.load(ENERGY,weights_only=True)['state_dict'])
    records={};start=time.perf_counter()
    with torch.inference_mode():
        for name in meta['train50']+meta['eval25']:
            path=DEST/f'{Path(name).stem}.pt'
            if path.exists():
                r=torch.load(str(path),weights_only=True,mmap=True)
                assert r['interval_sha256']==sha(INTERVAL) and r['energy_sha256']==sha(ENERGY)
            else:
                row=joint_load([name])[0];out={k:[] for k in ('shift','scale','accuracy16','errors16','kernel16')}
                for idx in torch.arange(len(row['x'])).split(8):
                    d={k:row[k][idx] for k in ('x','g','prediction','target','half','residual','valid','first')}
                    sh,sc=corrected(net,d);p=phase_prediction(d['prediction'],d['half'],sh)
                    mu,lv=en(d['x'],d['g'],d['prediction']);gate=(~d['first'])[:,None,None,None]
                    p=apply_energy(p,mu*gate*.5,lv*gate*.5);e=errors(p,d['target']).double()
                    acc=(.5/(.5+e)*e.new_tensor([.5,.3,.2])).sum(-1)
                    inside=((d['residual']-d['half']*sh).abs()<=d['half']*sc)&d['valid']
                    kernel=(inside*torch.exp(-2*d['half']*sc/SIGMA)).double().sum((1,2,3))
                    for k,v in zip(out,(sh,sc,acc,e,kernel)):out[k].append(v)
                r={k:torch.cat(v) for k,v in out.items()}
                r.update(format_version=1,trajectory=name,interval_sha256=sha(INTERVAL),energy_sha256=sha(ENERGY),
                         feature_sha256=sha(ROOT/'dev/cache/temporal_interval_head'/path.name),valid=int(row['valid'].sum()))
                torch.save(r,path)
            records[name]=dict(windows=len(r['shift']),valid=r['valid'],reward=float((r['kernel16']*r['accuracy16']).sum()),errors=r['errors16'].sum(0).tolist())
            print(f'teacher {len(records)}/75 {name} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    def summary(names):
        rows=[records[n] for n in names];nv=sum(r['valid'] for r in rows);nw=sum(r['windows'] for r in rows)
        return dict(sps=100*sum(r['reward'] for r in rows)/nv,scores=(100/(1+.5*sum(np.array(r['errors']) for r in rows)/nw)).tolist(),windows=nw)
    out=dict(status='COMPLETE',format_version=1,**{k:meta[k] for k in ('train50','eval25','d30_eval10','c45_eval15','evidence')},
             interval_sha256=sha(INTERVAL),energy_sha256=sha(ENERGY),source_sha256=sha(Path(__file__)),records=records,
             baseline={k:summary(meta[k]) for k in ('train50','eval25','d30_eval10','c45_eval15')})
    (DEST/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
