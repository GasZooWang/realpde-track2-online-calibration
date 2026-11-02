"""Energy head training."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from train_temporal_interval import WORK,load,batch,SIGMA,corrected
from canonical_features import CanonicalTemporalHead
from energy_head import EnergyHead,phase_prediction,apply_energy,errors
from accuracy_screen import ROOT,sha,proxy,TIME

JOINT=ROOT/'dev/cache/joint_cache'
DEST=ROOT/'dev/cache/energy_head_cache'


def joint_load(names):
    rows=load(names)
    for row,name in zip(rows,names):
        d=torch.load(str(JOINT/f'{Path(name).stem}.pt'),weights_only=True,mmap=True)
        assert d['trajectory']==name and d['prediction'].shape==row['half'].shape
        row.update(prediction=d['prediction'],target=d['target'])
    return rows


def joint_batch(rows,ix):
    d=batch(rows,ix)
    for k in ('prediction','target'):d[k]=torch.stack([rows[t][k][i] for t,i in ix])
    return d


@torch.inference_mode()
def evaluate(model,interval,rows,names,thin=3):
    records={}
    for row,name in zip(rows,names):
        en=np.zeros((2,3));reward=np.zeros(2);nv=nw=0
        for idx in torch.arange(0,len(row['x']),thin).split(8):
            d={k:row[k][idx] for k in ('x','g','prediction','target','half','residual','valid','first')}
            sh,sc=corrected(interval,d);p=phase_prediction(d['prediction'],d['half'],sh)
            mu,lv=model(d['x'],d['g'],d['prediction'])
            gate=(~d['first'])[:,None,None,None];q=apply_energy(p,mu*gate,lv*gate)
            kernel=(((d['residual']-d['half']*sh).abs()<=d['half']*sc)*d['valid']*torch.exp(-2*d['half']*sc/SIGMA)).double().sum((1,2,3))
            for j,v in enumerate((p,q)):
                e=errors(v,d['target']).double();en[j]+=e.sum(0).numpy()
                accuracy=(.5/(.5+e)*e.new_tensor([.5,.3,.2])).sum(-1)
                reward[j]+=float((kernel*accuracy).sum())
            nv+=int(d['valid'].sum());nw+=len(idx)
        records[name]=dict(errors=en.tolist(),reward=reward.tolist(),valid=nv,windows=nw)
    def summary(group):
        r=[records[n] for n in group];windows=sum(v['windows'] for v in r);valid=sum(v['valid'] for v in r)
        scores=100/(1+.5*sum(np.array(v['errors']) for v in r)/windows)
        sps=100*sum(np.array(v['reward']) for v in r)/valid
        metrics=np.c_[scores,sps,[proxy([*s,TIME,q]) for s,q in zip(scores,sps)]]
        return dict(metrics=metrics.tolist(),delta=(metrics[1]-metrics[0]).tolist())
    return dict(**summary(names),trajectories={n:summary([n]) for n in names},records=records)


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--epochs',type=int,default=40);pa.add_argument('--steps',type=int,default=100)
    pa.add_argument('--stop-after',type=int,default=None)
    pa.add_argument('--batch',type=int,default=12);pa.add_argument('--device',default='mps');pa.add_argument('--tag',default='energy64')
    args=pa.parse_args();torch.set_num_threads(2);torch.manual_seed(160908);rng=np.random.default_rng(160908)
    DEST.mkdir(parents=True,exist_ok=True);meta=json.loads((JOINT/'manifest.json').read_text());assert meta['status']=='COMPLETE'
    assert set(meta['train50']).isdisjoint(meta['eval25'])
    train=joint_load(meta['train50']);ev=joint_load(meta['eval25'])
    seed=WORK/'best_canonical96.pt';interval=CanonicalTemporalHead(96).eval();interval.load_state_dict(torch.load(seed,weights_only=True)['state_dict'])
    cpu=EnergyHead(64).eval();model=EnergyHead(64);model.load_state_dict(cpu.state_dict())
    initial=evaluate(cpu,interval,ev,meta['eval25'],thin=4);assert max(abs(v) for v in initial['delta'])<2e-5
    interval.to(args.device).requires_grad_(False);model.to(args.device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
    best=0.;history=[];start=time.perf_counter()
    torch.save(dict(state_dict=cpu.state_dict(),width=64,epoch=0,training_names=meta['train50']),DEST/f'best_{args.tag}.pt')
    print(json.dumps(dict(preflight='PASS',configuration=vars(args),parameters=sum(p.numel() for p in model.parameters()))),flush=True)
    eval_interval=CanonicalTemporalHead(96).eval();eval_interval.load_state_dict(torch.load(seed,weights_only=True)['state_dict'])
    for epoch in range(min(args.epochs, args.stop_after or args.epochs)):
        losses=[]
        for group in optimizer.param_groups:group['lr']=.0005*(.3+.7*.5*(1+math.cos(math.pi*epoch/args.epochs)))
        for step in range(args.steps):
            tr=rng.integers(0,len(train),size=args.batch)
            ix=[(int(t),int(rng.integers(1,len(train[t]['x'])))) for t in tr]
            d={k:v.to(args.device) for k,v in joint_batch(train,ix).items()}
            with torch.no_grad():
                sh,sc=interval(d['x'],d['g']);p=phase_prediction(d['prediction'],d['half'],sh)
                kernel=(((d['residual']-d['half']*sh).abs()<=d['half']*sc)*d['valid']*torch.exp(-2*d['half']*sc/SIGMA)).sum((1,2,3))
            mu,lv=model(d['x'],d['g'],d['prediction']);q=apply_energy(p,mu,lv);e=errors(q,d['target'])
            accuracy=(.5/(.5+e)*e.new_tensor([.5,.3,.2])).sum(-1)
            scores=1/(1+.5*e)
            loss=-(kernel*accuracy).sum()/d['valid'].sum().clamp_min(1)-.2*(scores*e.new_tensor([.5,.1,.1])).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
        record=dict(epoch=epoch+1,loss=float(np.mean(losses)),elapsed=time.perf_counter()-start)
        print(json.dumps(record),flush=True)
        if (epoch+1)%5==0 or epoch+1==args.epochs:
            cpu.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()})
            result=evaluate(cpu,eval_interval,ev,meta['eval25'],thin=3);record['eval25']=result;history.append(record)
            if result['delta'][-1]>best:
                best=result['delta'][-1];torch.save(dict(state_dict=cpu.state_dict(),width=64,epoch=epoch+1,training_names=meta['train50'],
                    evidence=meta['evidence'],source_sha256=sha(Path(__file__)),seed_sha256=sha(seed)),DEST/f'best_{args.tag}.pt')
            (ROOT/f'dev/reports/multiscale_{args.tag}_progress.json').write_text(json.dumps(dict(status='RUNNING',history=history),indent=2)+'\n')
    ck=DEST/f'best_{args.tag}.pt';cpu.load_state_dict(torch.load(ck,weights_only=True)['state_dict'])
    full=evaluate(cpu,eval_interval,ev,meta['eval25'],thin=1)
    report=dict(status='COMPLETE',configuration=vars(args),eval25=full,history=history,model_sha256=sha(ck),
        evidence=meta['evidence'],source_sha256=sha(Path(__file__)))
    (ROOT/f'dev/reports/multiscale_{args.tag}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE')),flush=True)


if __name__=='__main__':main()
