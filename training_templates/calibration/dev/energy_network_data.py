"""Energy evaluation data."""
import json
from pathlib import Path
import numpy as np
import torch
from prepare_energy import DEST
from prepare_moments import moment_errors
from train_moment_head import metrics
from energy_network_features import energy_logscale

def load(names):return [torch.load(str(DEST/f'{Path(n).stem}.pt'),weights_only=True,mmap=True) for n in names]
def split(meta):
 hold=[]
 for angle in (0,5,10,15,20):
  names=sorted([n for n in meta['train50'] if n.endswith(f'_{angle}.h5')],key=lambda x:int(x.split('_')[0]));assert len(names)>=6
  hold.extend(names[int(q*(len(names)-1))] for q in (.2,.5,.8))
 assert len(hold)==len(set(hold))==15
 return [n for n in meta['train50'] if n not in hold],hold

def logscale(raw,r,mode,strength):
 if mode=='delta':return energy_logscale(raw,r['pm'],r['pv'],r['reference'],strength)
 return (raw*strength).clamp(-3,3).expand_as(r['pm'])

@torch.inference_mode()
def evaluate(predict,rows,names,mode='delta',strengths=(.25,.5,1.,1.5),thin=3):
 records={}
 for r,n in zip(rows,names):
  ix=torch.arange(0,len(r['features']),thin);d={k:v[ix] for k,v in r.items() if isinstance(v,torch.Tensor)}
  raw=predict(d['features']).cpu();assert raw.shape==d['reference'].shape and torch.isfinite(raw).all()
  es=[d['errors16'],d['errors18']]
  for strength in strengths:
   lv=logscale(raw,d,mode,strength)*(~d['first'])[:,None,None,None]
   es.append(moment_errors(*(d[k] for k in ('pm','pv','tm','tv','cov')),torch.zeros_like(d['pm']),lv))
  es=torch.stack(es).double();acc=(.5/(.5+es)*es.new_tensor([.5,.3,.2])).sum(-1)
  kernels=torch.stack([d['kernel16']]+[d['kernel18']]*(len(es)-1)).double()
  records[n]=dict(errors=es.sum(1).tolist(),reward=(acc*kernels).sum(1).tolist(),valid=int(d['valid_count'].sum()),windows=len(ix))
 result=metrics(records,names);result['records']=records
 result['by_angle']={str(a):metrics(records,[n for n in names if n.endswith(f'_{a}.h5')]) for a in (0,5,10,15,20)}
 return result

def select(result,strengths):
 values=np.array(result['metrics']);gain=values[:, -1]-values[1,-1];idx=int(np.argmax(gain[1:]))+1
 return dict(index=idx,strength=0. if idx==1 else strengths[idx-2],gain_over_forecast=float(gain[idx]),gain_over_multiscale=float(result['delta'][idx][-1]))

def samples(rows,mode,pixels=192,thin=2,seed=190911):
 rng=np.random.default_rng(seed);xx=[];yy=[];ww=[]
 for r in rows:
  for i in range(1,len(r['features']),thin):
   ix=rng.integers(0,2048,size=pixels);f=r['features'][i].flatten(1).T.numpy()[ix]
   pk=r['pv'][i].sum(0).numpy().flatten();tk=r['tv'][i].sum(0).numpy().flatten();ref=r['reference'][i,0].numpy().flatten()
   if mode=='delta':y=((tk-pk)/ref).clip(-8,8);w=ref**2/max(float((tk**2).sum()),1e-16)
   else:y=np.log(np.maximum(tk,1e-12)/np.maximum(pk,1e-12)).clip(-3,3);w=pk**2/max(float((tk**2).sum()),1e-16)
   xx.append(f);yy.append(y[ix]);ww.append(w[ix])
 x=np.concatenate(xx).astype('float32');y=np.concatenate(yy).astype('float32');w=np.concatenate(ww).astype('float64');w/=w.mean()
 assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all()
 return x,y,w
