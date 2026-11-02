"""Moment cache."""
import json,time
from pathlib import Path
import torch
from train_phase_head import phase_load
from prepare_distribution import DEST as SOURCE, ENERGY
from energy_head import EnergyHead,phase_prediction,apply_energy,errors
from accuracy_screen import ROOT,sha

DEST=ROOT/'dev/cache/moment_head_cache'

def moment_errors(pm,pv,tm,tv,cov,mu,lv):
    qm=pm+mu; amp=torch.exp(.5*lv);qv=pv*amp.square()
    mse=(qm-tm).square()+qv+tv-2*amp*cov
    rel=(mse.sum((1,2,3)).clamp_min(1e-16)/(tm.square()+tv).sum((1,2,3)).clamp_min(1e-16)).sqrt()
    pk=.5*qv.sum(1);tk=.5*tv.sum(1)
    tke=((pk-tk).square().sum((1,2))/tk.square().sum((1,2)).clamp_min(1e-16)).sqrt()
    mv=[]
    for px in (13,21,29,37):
        a=(qm-tm)[:,:,8:25:2,px].flatten(1);b=tm[:,:,8:25:2,px].flatten(1)
        mv.append(torch.linalg.vector_norm(a,dim=1)/torch.linalg.vector_norm(b,dim=1).clamp_min(1e-8))
    return torch.stack([rel,tke,torch.stack(mv).mean(0)],-1)

@torch.inference_mode()
def main():
    torch.set_num_threads(2);DEST.mkdir(exist_ok=True,parents=True)
    meta=json.loads((SOURCE/'manifest.json').read_text());assert set(meta['train50']).isdisjoint(meta['eval25'])
    energy=EnergyHead().eval();energy.load_state_dict(torch.load(ENERGY,weights_only=True)['state_dict'])
    stats={};max_drift=0.;start=time.perf_counter()
    for name in meta['train50']+meta['eval25']:
        path=DEST/f'{Path(name).stem}.pt'
        row=phase_load([name])[0];out={k:[] for k in ('pm','pv','tm','tv','cov')}
        for ix in torch.arange(len(row['x'])).split(8):
            x,g,p,tar,half,sh,first=[row[k][ix] for k in ('x','g','prediction','target','half','shift','first')]
            mu,lv=energy(x,g,p);gate=(~first)[:,None,None,None]
            q=apply_energy(phase_prediction(p,half,sh),mu*gate*.5,lv*gate*.5)
            pp=q.reshape(-1,20,2,32,64);tt=tar.reshape_as(pp);pm=pp.mean(1);tm=tt.mean(1)
            pv=(pp-pm[:,None]).square().mean(1);tv=(tt-tm[:,None]).square().mean(1)
            cov=((pp-pm[:,None])*(tt-tm[:,None])).mean(1)
            analytic=moment_errors(pm,pv,tm,tv,cov,torch.zeros_like(pm),torch.zeros_like(pm))
            direct=errors(q,tar);drift=float((analytic-direct).abs().max());max_drift=max(drift,max_drift)
            assert drift<2e-5,(name,drift)
            assert float((direct.double()-row['errors16'][ix]).abs().max())<2e-5
            for k,v in zip(out,(pm,pv,tm,tv,cov)):out[k].append(v)
        result={k:torch.cat(v) for k,v in out.items()}
        result.update(trajectory=name,format_version=1,energy_sha256=sha(ENERGY),source_sha256=sha(Path(__file__)))
        torch.save(result,path);stats[name]={'windows':len(row['x']),'file_sha256':sha(path)}
        print(json.dumps(dict(prepared=len(stats),trajectory=name,elapsed=time.perf_counter()-start,max_error_drift=max_drift)),flush=True)
    report=dict(status='COMPLETE',**{k:meta[k] for k in ('train50','eval25','d30_eval10','c45_eval15','evidence')},
        records=stats,windows=sum(r['windows'] for r in stats.values()),max_analytic_error_drift=max_drift,
        source_manifest_sha256=sha(SOURCE/'manifest.json'),source_sha256=sha(Path(__file__)),energy_sha256=sha(ENERGY),
        features='x/g from the causal cache; pm/pv from the multiscale head; previous target shifted by one window',
        labels='tm/tv/cov are training and evaluation targets')
    (DEST/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE',windows=report['windows'],max_drift=max_drift)),flush=True)

if __name__=='__main__':main()
