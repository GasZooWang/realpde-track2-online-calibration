"""Phase head training."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from train_distribution_head import density_load,density_batch
from train_energy_head import JOINT
from prepare_distribution import DEST,INTERVAL,ENERGY
from phase_head import PhaseIntervalHead,shape_prediction
from energy_head import EnergyHead,apply_energy,errors
from accuracy_screen import ROOT,TIME,proxy,sha


def phase_load(names):
    rows=density_load(names)
    for r,n in zip(rows,names):
        d=torch.load(str(JOINT/f'{Path(n).stem}.pt'),weights_only=True,mmap=True);r.update(prediction=d['prediction'],target=d['target'])
    return rows


def phase_batch(rows,ix):
    d=density_batch(rows,ix)
    for k in ('prediction','target','kernel16'):d[k]=torch.stack([rows[t][k][i] for t,i in ix])
    return d


@torch.inference_mode()
def evaluate(model,energy,rows,names,thin=3,oracle=False):
    records={}
    for row,name in zip(rows,names):
        count=4 if oracle else 2;en=np.zeros((count,3));rr=np.zeros(count);nv=nw=0
        for idx in torch.arange(0,len(row['x']),thin).split(8):
            d={k:row[k][idx] for k in ('x','g','half','shift','first','prediction','target','valid','kernel16')}
            shape=torch.tanh(model.phase(model.latent(d['x'],d['g'])));gate=(~d['first'])[:,None,None,None];shape*=gate
            mu,lv=energy(d['x'],d['g'],d['prediction']);mu*=gate*.5;lv*=gate*.5
            base=apply_energy(shape_prediction(d['prediction'],d['half'],d['shift'],torch.zeros_like(shape)),mu,lv)
            cand=apply_energy(shape_prediction(d['prediction'],d['half'],d['shift'],shape),mu,lv);bank=[base,cand]
            if oracle:
                pp=base.reshape(-1,20,2,32,64);pm=pp.mean(1,keepdim=True);ps=pp.std(1,keepdim=True,correction=0)
                for source in (d['target'],d['x'][:,:40]*d['x'].new_tensor([.1,.02]).repeat(20)[None,:,None,None]):
                    t=source.reshape_as(pp);tm=t.mean(1,keepdim=True);ts=t.std(1,keepdim=True,correction=0)
                    new=torch.where(ts>1e-6,(t-tm)*ps/ts.clamp_min(1e-6),pp-pm)
                    bank.append((pm+new).flatten(1,2))
            for j,p in enumerate(bank):
                e=errors(p,d['target']).double();en[j]+=e.sum(0).numpy();accuracy=(.5/(.5+e)*e.new_tensor([.5,.3,.2])).sum(-1)
                rr[j]+=float((d['kernel16']*accuracy).sum())
            nv+=int(d['valid'].sum());nw+=len(idx)
        records[name]=dict(errors=en.tolist(),rewards=rr.tolist(),valid=nv,windows=nw)
    scores=100/(1+.5*sum(np.array(r['errors']) for r in records.values())/sum(r['windows'] for r in records.values()))
    sps=100*sum(np.array(r['rewards']) for r in records.values())/sum(r['valid'] for r in records.values())
    metrics=np.c_[scores,sps,[proxy([*s,TIME,p]) for s,p in zip(scores,sps)]]
    return dict(metrics=metrics.tolist(),delta=(metrics-metrics[0]).tolist(),records=records)


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--epochs',type=int,default=30);pa.add_argument('--steps',type=int,default=120)
    pa.add_argument('--batch',type=int,default=8);pa.add_argument('--device',default='mps');pa.add_argument('--tag',default='phase_linear')
    pa.add_argument('--oracle-only',action='store_true');args=pa.parse_args();torch.set_num_threads(2);torch.manual_seed(170911);rng=np.random.default_rng(170911)
    meta=json.loads((DEST/'manifest.json').read_text());assert set(meta['train50']).isdisjoint(meta['eval25'])
    train=phase_load(meta['train50']);ev=phase_load(meta['eval25']);seed=torch.load(INTERVAL,weights_only=True)
    model=PhaseIntervalHead();missing,unexpected=model.load_state_dict(seed['state_dict'],strict=False)
    assert not unexpected and all(k.startswith('phase.') for k in missing)
    cpu=PhaseIntervalHead().eval();cpu.load_state_dict(model.state_dict());energy=EnergyHead().eval();energy.load_state_dict(torch.load(ENERGY,weights_only=True)['state_dict'])
    initial=evaluate(cpu,energy,ev,meta['eval25'],thin=8,oracle=True);assert max(abs(v) for v in initial['delta'][1])<2e-5
    preflight=dict(status='PASS',configs=['multiscale_head','zero_head','target_phase','current_input_phase'],eval25_subset=initial)
    (ROOT/'dev/reports/distribution_phase_oracle.json').write_text(json.dumps(preflight,indent=2)+'\n')
    print(json.dumps(dict(preflight='PASS')),flush=True)
    if args.oracle_only:return
    cpu_energy=EnergyHead().eval();cpu_energy.load_state_dict(energy.state_dict())
    model.requires_grad_(False);model.phase.requires_grad_(True);model.to(args.device).train();energy.requires_grad_(False).to(args.device)
    optimizer=torch.optim.AdamW(model.phase.parameters(),lr=.001,weight_decay=.01);best=0.;history=[];start=time.perf_counter()
    for epoch in range(args.epochs):
        losses=[]
        for group in optimizer.param_groups:group['lr']=.001*(.3+.7*.5*(1+math.cos(math.pi*epoch/args.epochs)))
        for step in range(args.steps):
            ix=[(int(t),int(rng.integers(1,len(train[t]['x'])))) for t in rng.integers(0,len(train),size=args.batch)]
            d={k:v.float().to(args.device) if v.dtype==torch.float64 else v.to(args.device) for k,v in phase_batch(train,ix).items()}
            with torch.no_grad():z=model.latent(d['x'],d['g']);mu,lv=energy(d['x'],d['g'],d['prediction'])
            shape=torch.tanh(model.phase(z));p=apply_energy(shape_prediction(d['prediction'],d['half'],d['shift'],shape),mu*.5,lv*.5)
            rel=torch.linalg.vector_norm((p-d['target']).flatten(1),dim=1)/torch.linalg.vector_norm(d['target'].flatten(1),dim=1).clamp_min(1e-8)
            loss=-(d['kernel16']*.5*.5/(.5+rel)).sum()/d['valid'].sum().clamp_min(1)-.1*(1/(1+.5*rel)).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.phase.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        record=dict(epoch=epoch+1,loss=float(np.mean(losses)),elapsed=time.perf_counter()-start);print(json.dumps(record),flush=True)
        if (epoch+1)%5==0 or epoch+1==args.epochs:
            cpu.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()});result=evaluate(cpu,cpu_energy,ev,meta['eval25'],thin=3)
            history.append(dict(**record,eval25=result))
            if result['delta'][1][-1]>best:
                best=result['delta'][1][-1];torch.save(dict(state_dict=cpu.state_dict(),width=96,architecture='phase_linear',epoch=epoch+1,
                  training_names=meta['train50'],evidence=meta['evidence'],seed_sha256=sha(INTERVAL),source_sha256=sha(Path(__file__))),DEST/f'best_{args.tag}.pt')
            (ROOT/f'dev/reports/distribution_{args.tag}_progress.json').write_text(json.dumps(dict(status='RUNNING',history=history),indent=2)+'\n')
    path=DEST/f'best_{args.tag}.pt'
    if path.exists():cpu.load_state_dict(torch.load(path,weights_only=True)['state_dict'])
    result=evaluate(cpu,cpu_energy,ev,meta['eval25'],thin=1)
    report=dict(status='COMPLETE',configuration=vars(args),eval25=result,history=history,model_sha256=sha(path) if path.exists() else None,
       evidence=meta['evidence'])
    (ROOT/f'dev/reports/distribution_{args.tag}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='COMPLETE')),flush=True)


if __name__=='__main__':main()
