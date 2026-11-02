"""Energy feature cache."""
import argparse,json,time
from pathlib import Path
import torch
from train_forecast_head import load,batch,forward,kernel
from prepare_forecast import DEST as FORECAST_CACHE
from prepare_moments import moment_errors
from forecast_head import CausalForecastHead
from energy_head import errors
from energy_network_features import energy_features
from accuracy_screen import ROOT,sha
DEST=ROOT/'dev/cache/energy_network_cache';CK=FORECAST_CACHE/'best_joint32.pt'
@torch.inference_mode()
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--device',default='mps');pa.add_argument('--max-seconds',type=float,default=600);a=pa.parse_args();torch.set_num_threads(2)
 DEST.mkdir(exist_ok=True);meta=json.loads((FORECAST_CACHE/'manifest.json').read_text())
 expected=json.loads((ROOT/'dev/reports/forecast_joint32.json').read_text())['checkpoint_sha256']
 assert sha(CK)==expected, 'unexpected forecast checkpoint'
 ck=torch.load(CK,weights_only=True);net=CausalForecastHead(ck['width']).eval().to(a.device);net.load_state_dict(ck['state_dict']);records={};start=time.perf_counter();maxdrift=0.
 for name in meta['train50']+meta['eval25']:
  assert time.perf_counter()-start<a.max_seconds,'duration cap reached'
  rows=load([name]);r=rows[0];out={k:[] for k in ('features','reference','pm','pv','tm','tv','cov','errors16','errors18','kernel16','kernel18','valid_count','first')}
  for inds in torch.arange(len(r['x'])).split(8):
   d=batch(rows,[(0,int(i)) for i in inds],a.device);q,sh,sc=forward(net,d);pp=q.reshape(-1,20,2,32,64);tt=d['target'].reshape_as(pp)
   pm=pp.mean(1);pv=pp.var(1,correction=0);tm=tt.mean(1);tv=tt.var(1,correction=0);cov=((pp-pm[:,None])*(tt-tm[:,None])).mean(1)
   f,ref=energy_features(d['x'],d['history'],pm,pv);e=errors(q,d['target']);analytic=moment_errors(pm,pv,tm,tv,cov,torch.zeros_like(pm),torch.zeros_like(pm))
   drift=float((e-analytic).abs().max());maxdrift=max(maxdrift,drift);assert drift<2e-5
   vals=(f,ref,pm,pv,tm,tv,cov,d['errors16'],e,d['kernel16'],kernel(d,sh,sc),d['valid'].sum((1,2,3)),d['first'])
   for k,v in zip(out,vals):out[k].append(v.cpu())
  payload={k:torch.cat(v) for k,v in out.items()};payload.update(trajectory=name,teacher_sha256=sha(CK),feature_sha256=sha(ROOT/'dev/energy_network_features.py'))
  path=DEST/f'{Path(name).stem}.pt';assert not path.exists(),path;torch.save(payload,path);records[name]=dict(windows=len(r['x']),sha256=sha(path))
  print(json.dumps(dict(prepared=len(records),trajectory=name,elapsed=time.perf_counter()-start,max_error_drift=maxdrift)),flush=True)
 manifest=dict(status='COMPLETE',**{k:meta[k] for k in ('train50','eval25','d30_eval10','c45_eval15','evidence')},records=records,max_error_drift=maxdrift,
   teacher_sha256=sha(CK),source_sha256=sha(Path(__file__)),feature_sha256=sha(ROOT/'dev/energy_network_features.py'),
   feature_boundary='features from inputs, revealed targets, and forecast-head moments; tm/tv/cov are training targets')
 (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('COMPLETE',flush=True)
if __name__=='__main__':main()
