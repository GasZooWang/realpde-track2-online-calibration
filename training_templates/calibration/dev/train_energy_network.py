"""Energy network training."""
import argparse,json,time,math,shutil
from pathlib import Path
import torch
import numpy as np
from energy_network_data import load,split,evaluate,select,logscale
from energy_network import DirectEnergyHead
from prepare_energy import DEST
from prepare_moments import moment_errors
from accuracy_screen import ROOT,sha
KEYS=('features','reference','pm','pv','tm','tv','cov','first')
def batch(rows,ix,device):return {k:torch.stack([rows[t][k][i] for t,i in ix]).to(device) for k in KEYS}
@torch.inference_mode()
def predict_fn(model,device):
 def run(f):return torch.cat([model(part.to(device)).cpu() for part in f.split(8)])
 return run

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--device',default='mps');pa.add_argument('--epochs',type=int,default=30);pa.add_argument('--steps',type=int,default=100);pa.add_argument('--batch',type=int,default=16);pa.add_argument('--max-seconds',type=float,default=1500);a=pa.parse_args();torch.set_num_threads(2)
 meta=json.loads((DEST/'manifest.json').read_text());trainnames,holdnames=split(meta);train=load(trainnames);held=load(holdnames)
 sources=['train_energy_network.py','energy_network.py','energy_network_features.py','energy_network_data.py'];snapshot=ROOT/'dev/reports/energy_network_sources';snapshot.mkdir(exist_ok=True)
 for n in sources:shutil.copy2(ROOT/'dev'/n,snapshot/n)
 hashes={n:sha(snapshot/n) for n in sources};start=time.perf_counter();history=[];best=0.;best_epoch=0;best_strength=0.;strengths=[.5]
 def init():
  torch.manual_seed(190911);net=DirectEnergyHead().to(a.device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.05);return net,opt,np.random.default_rng(190911)
 def epoch_run(net,opt,rng,rows,epoch):
  net.train();losses=[]
  for g in opt.param_groups:g['lr']=.001*(.25+.75*.5*(1+math.cos(math.pi*epoch/a.epochs)))
  for step in range(a.steps):
   assert time.perf_counter()-start<a.max_seconds,'hard duration cap'
   ix=[(int(t),int(rng.integers(1,len(rows[t]['features'])))) for t in rng.integers(0,len(rows),size=a.batch)];d=batch(rows,ix,a.device)
   raw=net(d['features']);lv=logscale(raw,d,'delta',1.);e=moment_errors(*(d[k] for k in ('pm','pv','tm','tv','cov')),torch.zeros_like(d['pm']),lv)
   loss=(e[:,1].square()+10*e[:,0].square()).mean();assert torch.isfinite(loss)
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();losses.append(float(loss.detach()))
  return float(np.mean(losses))
 net,opt,rng=init();initial=evaluate(predict_fn(net,a.device),held,holdnames,'delta',[1.],thin=3)
 assert max(abs(x-y) for x,y in zip(initial['metrics'][1],initial['metrics'][2]))<2e-5
 print(json.dumps(dict(preflight='PASS',parameters=sum(p.numel() for p in net.parameters()))),flush=True)
 for epoch in range(a.epochs):
  loss=epoch_run(net,opt,rng,train,epoch);print(json.dumps(dict(phase='inner35',epoch=epoch+1,loss=loss,elapsed=time.perf_counter()-start)),flush=True)
  if (epoch+1)%5==0 or epoch+1==a.epochs:
   net.eval();res=evaluate(predict_fn(net,a.device),held,holdnames,'delta',strengths,thin=3);choice=select(res,strengths);rec=dict(epoch=epoch+1,choice=choice,inner15=res);history.append(rec)
   print(json.dumps(dict(validation=epoch+1,strength=choice['strength'])),flush=True)
   if choice['gain_over_forecast']>best:
    best=choice['gain_over_forecast'];best_epoch=epoch+1;best_strength=choice['strength'];torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},epoch=best_epoch,strength=best_strength),DEST/'inner_energy_nn.pt')
   (ROOT/'dev/reports/energy_network_progress.json').write_text(json.dumps(dict(status='RUNNING',history=history,best_epoch=best_epoch),indent=2)+'\n')
 if best_epoch==0:
  net,opt,rng=init();torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},width=32,epoch=0,strength=.5,training_names=meta['train50']),DEST/'energy_nn.pt')
  report=dict(status='COMPLETE_INNER_SCREEN_NO_GAIN',selected='neutral direct-energy head')
 else:
  train=load(meta['train50']);net,opt,rng=init()
  for epoch in range(best_epoch):
   loss=epoch_run(net,opt,rng,train,epoch);print(json.dumps(dict(phase='refit50',epoch=epoch+1,loss=loss,elapsed=time.perf_counter()-start)),flush=True)
  path=DEST/'energy_nn.pt';assert not path.exists();torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},width=32,epoch=best_epoch,strength=best_strength,training_names=meta['train50'],source_hashes=hashes,teacher_sha256=meta['teacher_sha256']),path)
  net.eval();res=evaluate(predict_fn(net,a.device),load(meta['eval25']),meta['eval25'],'delta',[best_strength],thin=1)
  report=dict(status='COMPLETE',selected_epoch=best_epoch,strength=best_strength,eval25=res,model_path=str(path),checkpoint_sha256=sha(path))
 report.update(configuration=vars(a),history=history,training_names=meta['train50'],inner_train35=trainnames,inner15=holdnames,source_hashes=hashes,
   evidence='fit50 training, inner15 selection, eval25 evaluation')
 (ROOT/'dev/reports/energy_network_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'])),flush=True)
if __name__=='__main__':main()
