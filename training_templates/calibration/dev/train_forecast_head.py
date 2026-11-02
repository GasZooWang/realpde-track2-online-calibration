"""Forecast head training."""
import argparse,json,time,math,os,shutil
from pathlib import Path
import torch
import numpy as np
from train_moment_head import load_rows,metrics,objective
from prepare_forecast import DEST
from forecast_head import CausalForecastHead
from energy_head import errors
from accuracy_screen import ROOT,TIME,sha
from train_temporal_interval import SIGMA

KEYS=('x','g','history','prediction16','target','first','kernel16','errors16','valid','half','residual','shift','scale')

def load(names):
    rows=load_rows(names,True)
    for r,n in zip(rows,names):
        data=torch.load(str(DEST/f'{Path(n).stem}.pt'),weights_only=True,mmap=True);assert data['trajectory']==n
        r['prediction16']=data['prediction16']
    return rows

def batch(rows,ix,device):
    return {k:torch.stack([rows[t][k][i] for t,i in ix]).to(device=device,dtype=torch.float32 if rows[0][k].dtype==torch.float64 else rows[0][k].dtype) for k in KEYS}

def forward(model,d):return model(*(d[k] for k in ('x','g','prediction16','history','first','shift','scale')))

def kernel(d,shift,scale,temp=None):
    h16=d['half']*d['scale'];h=h16*scale;r=d['residual']-d['half']*d['shift']-h16*shift
    inside=(r.abs()<=h) if temp is None else torch.sigmoid((h-r)/temp)*torch.sigmoid((h+r)/temp)
    return (inside*d['valid']*torch.exp(-2*h/SIGMA)).sum((1,2,3))

@torch.inference_mode()
def evaluate(model,rows,names,thin=3):
    model.eval();device=next(model.parameters()).device;records={}
    for t,(r,n) in enumerate(zip(rows,names)):
        en=np.zeros((2,3));rr=np.zeros(2);nv=nw=0
        for inds in torch.arange(0,len(r['x']),thin).split(8):
            d=batch(rows,[(t,int(i)) for i in inds],device);q,sh,sc=forward(model,d)
            e=errors(q,d['target']).cpu().double();base=d['errors16'].cpu().double()
            en+=torch.stack([base.sum(0),e.sum(0)]).numpy()
            acc=(.5/(.5+torch.stack([base,e]))*e.new_tensor([.5,.3,.2])).sum(-1)
            kernels=torch.stack([d['kernel16'],kernel(d,sh,sc)]).cpu().double()
            rr+=(acc*kernels).sum(1).numpy();nv+=int(d['valid'].sum());nw+=len(inds)
        records[n]=dict(errors=en.tolist(),reward=rr.tolist(),valid=nv,windows=nw)
    result=metrics(records,names);result['records']=records
    result['trajectory_delta']={n:metrics(records,[n])['delta'][1] for n in names}
    result['by_angle']={str(a):metrics(records,[n for n in names if n.endswith(f'_{a}.h5')]) for a in (0,5,10,15,20)}
    return result

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--tag',default='forecast32');pa.add_argument('--epochs',type=int,default=60)
    pa.add_argument('--steps',type=int,default=120);pa.add_argument('--batch',type=int,default=8);pa.add_argument('--width',type=int,default=32)
    pa.add_argument('--lr',type=float,default=.0004);pa.add_argument('--device',default='mps');pa.add_argument('--seed',type=int,default=180912)
    pa.add_argument('--resume',type=Path);pa.add_argument('--max-seconds',type=float,default=2400);pa.add_argument('--eval-only',action='store_true')
    args=pa.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    meta=json.loads((DEST/'manifest.json').read_text());assert meta['status']=='COMPLETE';assert set(meta['train50']).isdisjoint(meta['eval25'])
    sources=['train_forecast_head.py','forecast_head.py','moment_head.py','history_features.py','train_moment_head.py','prepare_moments.py']
    snapshot=ROOT/'dev/reports'/f'forecast_{args.tag}_sources';snapshot.mkdir(exist_ok=True)
    for n in sources:shutil.copy2(ROOT/'dev'/n,snapshot/n)
    hashes={n:sha(snapshot/n) for n in sources}
    train=load(meta['train50']);ev=load(meta['eval25']);model=CausalForecastHead(args.width).to(args.device)
    if args.resume:
        ck=torch.load(args.resume,weights_only=True);assert ck['training_names']==meta['train50']
        missing,unexpected=model.load_state_dict(ck['state_dict'],strict=False)
        assert not unexpected and all(k.startswith(('pred_stem.','shape_out.','bound_stem.','bound_out.')) for k in missing)
    start=time.perf_counter();history=[];initial=evaluate(model,ev,meta['eval25'],thin=3)
    if not args.resume:assert max(abs(v) for v in initial['delta'][1])<2e-5,initial['delta'][1]
    print(json.dumps(dict(preflight='PASS',parameters=sum(p.numel() for p in model.parameters()))),flush=True)
    if args.eval_only:
        result=evaluate(model,ev,meta['eval25'],thin=1)
        (ROOT/f'dev/reports/forecast_{args.tag}_eval.json').write_text(json.dumps(result,indent=2)+'\n');return
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01);best=initial['delta'][1][-1];best_epoch=0
    path=DEST/f'best_{args.tag}.pt'
    def save(epoch):
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},width=args.width,epoch=epoch,
          training_names=meta['train50'],source_hashes=hashes,manifest_sha256=sha(DEST/'manifest.json'),evidence=meta['evidence'],
          seed=args.seed,resume=None if args.resume is None else str(args.resume)),path)
    save(0);status='COMPLETE'
    for epoch in range(args.epochs):
        model.train();losses=[]
        for group in optimizer.param_groups:group['lr']=args.lr*(.25+.75*.5*(1+math.cos(math.pi*epoch/args.epochs)))
        for step in range(args.steps):
            if time.perf_counter()-start>args.max_seconds:status='STOPPED_MAX_DURATION';break
            ix=[(int(t),int(rng.integers(1,len(train[t]['x'])))) for t in rng.integers(0,len(train),size=args.batch)]
            d=batch(train,ix,args.device);q,sh,sc=forward(model,d);e=errors(q,d['target'])
            kd=dict(d);kd['kernel16']=kernel(d,sh,sc,temp=max(.00008,.0003*.96**epoch));loss=objective(e,kd)
            assert torch.isfinite(loss),('nonfinite loss',epoch,step)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
        record=dict(epoch=epoch+1,loss=float(np.mean(losses)),elapsed=time.perf_counter()-start);print(json.dumps(record),flush=True)
        if (epoch+1)%5==0 or epoch+1==args.epochs or status!='COMPLETE':
            res=evaluate(model,ev,meta['eval25'],thin=3);record['eval25']=res
            if res['delta'][1][-1]>best:best=res['delta'][1][-1];best_epoch=epoch+1;save(epoch+1)
            history.append(record)
            (ROOT/f'dev/reports/forecast_{args.tag}_progress.json').write_text(json.dumps(dict(status='RUNNING',history=history,best_epoch=best_epoch),indent=2)+'\n')
        if status!='COMPLETE':break
    model.load_state_dict(torch.load(path,weights_only=True)['state_dict']);full=evaluate(model,ev,meta['eval25'],thin=1)
    report=dict(status=status,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},history=history,
        eval25=full,best_epoch=best_epoch,checkpoint_sha256=sha(path),public_proxy_target_met=full['delta'][1][-1]>=.5,
        evidence=meta['evidence'],columns=['Rel-L2','TKE','MVPE','SPS','fixed_Time_proxy'],fixed_time=TIME,source_hashes=hashes)
    (ROOT/f'dev/reports/forecast_{args.tag}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=status,best_epoch=best_epoch)),flush=True)

if __name__=='__main__':main()
