"""Moment head training."""
import argparse,json,time,math,os
from pathlib import Path
import numpy as np
import torch
from prepare_moments import DEST,moment_errors
from train_phase_head import phase_load
from moment_head import CausalMomentHead
from accuracy_screen import ROOT,TIME,proxy,sha

KEYS=('x','g','pm','pv','tm','tv','cov','first','errors16','kernel16','valid')

def load_rows(names,history=False):
    rows=phase_load(names)
    for r,n in zip(rows,names):
        d=torch.load(str(DEST/f'{Path(n).stem}.pt'),weights_only=True,mmap=True)
        assert d['trajectory']==n
        r.update({k:d[k] for k in ('pm','pv','tm','tv','cov')})
        if history:
            from history_features import make_history
            r['history']=make_history(r)
    return rows

def batch(rows,ix,device):
    data={k:torch.stack([rows[t][k][i] for t,i in ix]).to(device=device,dtype=torch.float32 if rows[0][k].dtype==torch.float64 else rows[0][k].dtype) for k in KEYS}
    data['previous_target']=torch.stack([rows[t]['target'][i-1] if i>0 else torch.zeros_like(rows[t]['target'][i]) for t,i in ix]).to(device)
    if 'history' in rows[0]:data['history']=torch.stack([rows[t]['history'][i] for t,i in ix]).to(device)
    return data

def get_errors(model,d,strength=1.):
    mu,lv=model(d['x'],d['g'],d['pm'],d['pv'],d['history'] if model.history else d['previous_target']);gate=(~d['first'])[:,None,None,None]
    return moment_errors(*(d[k] for k in ('pm','pv','tm','tv','cov')),mu*gate*strength,lv*gate*strength)

def metrics(records,names):
    r=[records[n] for n in names];nw=sum(v['windows'] for v in r);nv=sum(v['valid'] for v in r)
    scores=100/(1+.5*sum(np.array(v['errors']) for v in r)/nw)
    sps=100*sum(np.array(v['reward']) for v in r)/nv
    values=np.c_[scores,sps,[proxy([*s,TIME,p]) for s,p in zip(scores,sps)]]
    return dict(metrics=values.tolist(),delta=(values-values[0]).tolist(),windows=nw)

@torch.inference_mode()
def evaluate(model,rows,names,thin=3,strength=1.):
    model.eval();device=next(model.parameters()).device;records={}
    for t,(r,n) in enumerate(zip(rows,names)):
        en=np.zeros((2,3));rr=np.zeros(2);nv=nw=0
        for inds in torch.arange(0,len(r['x']),thin).split(8):
            d=batch(rows,[(t,int(i)) for i in inds],device)
            e=get_errors(model,d,strength).cpu().double();base=d['errors16'].cpu().double()
            en+=torch.stack([base.sum(0),e.sum(0)]).numpy()
            acc=(.5/(.5+torch.stack([base,e]))*e.new_tensor([.5,.3,.2])).sum(-1)
            rr+=(acc*d['kernel16'].cpu().double()[None]).sum(1).numpy()
            nv+=int(d['valid'].sum());nw+=len(inds)
        records[n]=dict(errors=en.tolist(),reward=rr.tolist(),valid=nv,windows=nw)
    result=metrics(records,names);result['records']=records
    result['trajectory_delta']={n:metrics(records,[n])['delta'][1] for n in names}
    result['by_angle']={str(a):metrics(records,[n for n in names if n.endswith(f'_{a}.h5')]) for a in (0,5,10,15,20)}
    return result

def objective(e,d):
    scores=100/(1+.5*e.mean(0))
    accuracy=(.5/(.5+e)*e.new_tensor([.5,.3,.2])).sum(-1)
    sps=100*(d['kernel16']*accuracy).sum()/d['valid'].sum().clamp_min(1)
    scores=torch.cat([scores,e.new_tensor([TIME]),sps[None]])
    return -(scores.pow(.6854)*e.new_tensor([.5,.1,.1,.1,.2])).sum().pow(1/.6854)/100

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--tag',default='moments32');pa.add_argument('--epochs',type=int,default=30)
    pa.add_argument('--steps',type=int,default=100);pa.add_argument('--batch',type=int,default=8)
    pa.add_argument('--width',type=int,default=32);pa.add_argument('--lr',type=float,default=.0004)
    pa.add_argument('--device',default='mps');pa.add_argument('--seed',type=int,default=180911);pa.add_argument('--resume',type=Path)
    pa.add_argument('--eval-only',action='store_true');pa.add_argument('--max-seconds',type=float,default=3600)
    pa.add_argument('--history',action='store_true')
    args=pa.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    meta=json.loads((DEST/'manifest.json').read_text());assert meta['status']=='COMPLETE'
    assert set(meta['train50']).isdisjoint(meta['eval25'])
    train=load_rows(meta['train50'],args.history);ev=load_rows(meta['eval25'],args.history);model=CausalMomentHead(args.width,args.history).to(args.device)
    if args.resume:
        ck=torch.load(args.resume,weights_only=True);assert ck['training_names']==meta['train50'];model.load_state_dict(ck['state_dict'])
    start=time.perf_counter();history=[];initial=evaluate(model,ev,meta['eval25'],thin=3)
    if not args.resume:assert max(abs(v) for v in initial['delta'][1])<2e-5,initial['delta'][1]
    print(json.dumps(dict(preflight='PASS',parameters=sum(p.numel() for p in model.parameters()))),flush=True)
    if args.eval_only:
        out=evaluate(model,ev,meta['eval25'],thin=1)
        (ROOT/f'dev/reports/forecast_{args.tag}_eval.json').write_text(json.dumps(out,indent=2)+'\n');return
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01);best=initial['delta'][1][-1];best_epoch=0
    path=DEST/f'best_{args.tag}.pt'
    def save(epoch):
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},width=args.width,epoch=epoch,
          training_names=meta['train50'],source_sha256=sha(Path(__file__)),model_source_sha256=sha(ROOT/'dev/moment_head.py'),
          manifest_sha256=sha(DEST/'manifest.json'),evidence=meta['evidence'],seed=args.seed,history=args.history),path)
    save(0);status='COMPLETE'
    for epoch in range(args.epochs):
        model.train();losses=[]
        for group in optimizer.param_groups:group['lr']=args.lr*(.25+.75*.5*(1+math.cos(math.pi*epoch/args.epochs)))
        for step in range(args.steps):
            if time.perf_counter()-start>args.max_seconds:status='STOPPED_MAX_DURATION';break
            ix=[(int(t),int(rng.integers(1,len(train[t]['x'])))) for t in rng.integers(0,len(train),size=args.batch)]
            d=batch(train,ix,args.device);e=get_errors(model,d);loss=objective(e,d)
            assert torch.isfinite(loss),('nonfinite loss',epoch,step)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
        record=dict(epoch=epoch+1,loss=float(np.mean(losses)),elapsed=time.perf_counter()-start)
        print(json.dumps(record),flush=True)
        if (epoch+1)%5==0 or epoch+1==args.epochs or status!='COMPLETE':
            res=evaluate(model,ev,meta['eval25'],thin=3);record['eval25']=res
            if res['delta'][1][-1]>best:best=res['delta'][1][-1];best_epoch=epoch+1;save(epoch+1)
            history.append(record)
            (ROOT/f'dev/reports/forecast_{args.tag}_progress.json').write_text(json.dumps(dict(status='RUNNING',history=history,best_epoch=best_epoch),indent=2)+'\n')
        if status!='COMPLETE':break
    model.load_state_dict(torch.load(path,weights_only=True)['state_dict']);full=evaluate(model,ev,meta['eval25'],thin=1)
    report=dict(status=status,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},history=history,
        eval25=full,best_epoch=best_epoch,checkpoint_sha256=sha(path),public_proxy_target_met=full['delta'][1][-1]>=.5,
        evidence=meta['evidence'],columns=['Rel-L2','TKE','MVPE','SPS','fixed_Time_proxy'],fixed_time=TIME)
    (ROOT/f'dev/reports/forecast_{args.tag}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=status,best_epoch=best_epoch)),flush=True)

if __name__=='__main__':main()
