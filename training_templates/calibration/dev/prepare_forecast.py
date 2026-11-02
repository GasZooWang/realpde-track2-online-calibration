"""Forecast cache."""
import json,time
from pathlib import Path
import torch
from train_phase_head import phase_load
from prepare_distribution import DEST as SOURCE,ENERGY
from energy_head import EnergyHead,phase_prediction,apply_energy,errors
from accuracy_screen import ROOT,sha

DEST=ROOT/'dev/cache/forecast_head_cache'

@torch.inference_mode()
def main():
    torch.set_num_threads(2);DEST.mkdir(parents=True,exist_ok=True)
    meta=json.loads((SOURCE/'manifest.json').read_text());energy=EnergyHead().eval()
    energy.load_state_dict(torch.load(ENERGY,weights_only=True)['state_dict']);start=time.perf_counter();records={}
    for name in meta['train50']+meta['eval25']:
        r=phase_load([name])[0];parts=[];worst=0.
        for ix in torch.arange(len(r['x'])).split(8):
            x,g,p,t,half,sh,first=[r[k][ix] for k in ('x','g','prediction','target','half','shift','first')]
            mu,lv=energy(x,g,p);gate=(~first)[:,None,None,None]
            q=apply_energy(phase_prediction(p,half,sh),mu*gate*.5,lv*gate*.5)
            drift=float((errors(q,t).double()-r['errors16'][ix]).abs().max());worst=max(worst,drift);assert drift<2e-5
            parts.append(q)
        path=DEST/f'{Path(name).stem}.pt';torch.save(dict(prediction16=torch.cat(parts),trajectory=name),path)
        records[name]=dict(windows=len(r['x']),sha256=sha(path),drift=worst)
        print(json.dumps(dict(prepared=len(records),elapsed=time.perf_counter()-start)),flush=True)
    out=dict(status='COMPLETE',records=records,**{k:meta[k] for k in ('train50','eval25','d30_eval10','c45_eval15','evidence')},
      source_sha256=sha(Path(__file__)),energy_sha256=sha(ENERGY),parent_manifest_sha256=sha(SOURCE/'manifest.json'))
    (DEST/'manifest.json').write_text(json.dumps(out,indent=2)+'\n');print('COMPLETE',flush=True)

if __name__=='__main__':main()
