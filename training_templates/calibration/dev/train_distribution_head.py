"""Distribution head training."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from train_temporal_interval import load,batch,SIGMA
from prepare_distribution import DEST
from distribution_head import DistributionHead,nll
from accuracy_screen import ROOT,sha


def density_load(names):
    rows=load(names)
    for row,name in zip(rows,names):
        d=torch.load(str(DEST/f'{Path(name).stem}.pt'),weights_only=True,mmap=True)
        assert d['trajectory']==name
        row.update({k:d[k] for k in ('shift','scale','accuracy16','errors16','kernel16')})
    return rows


def density_batch(rows,ix):
    d=batch(rows,ix)
    for k in ('shift','scale','accuracy16'):d[k]=torch.stack([rows[t][k][i] for t,i in ix])
    return d


@torch.inference_mode()
def evaluate(model,rows,thin=6):
    total=count=0.
    for row in rows:
        for ix in torch.arange(1,len(row['x']),thin).split(6):
            p=model(row['x'][ix],row['g'][ix],row['shift'][ix],row['scale'][ix])
            y=(row['residual'][ix]/row['half'][ix].clamp_min(1e-7)-row['shift'][ix])/row['scale'][ix]
            v=row['valid'][ix];total+=float((nll(p,y)*v).double().sum());count+=int(v.sum())
    return total/count


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--epochs',type=int,default=30);pa.add_argument('--steps',type=int,default=120)
    pa.add_argument('--batch',type=int,default=6);pa.add_argument('--device',default='mps');pa.add_argument('--tag',default='density96')
    args=pa.parse_args();torch.set_num_threads(2);torch.manual_seed(170907);rng=np.random.default_rng(170907)
    meta=json.loads((DEST/'manifest.json').read_text());assert meta['status']=='COMPLETE';assert set(meta['train50']).isdisjoint(meta['eval25'])
    train=density_load(meta['train50']);ev=density_load(meta['eval25'])
    model=DistributionHead();cpu=DistributionHead().eval();cpu.load_state_dict(model.state_dict())
    initial=evaluate(cpu,ev);print(json.dumps(dict(preflight='PASS',configuration=vars(args))),flush=True)
    model.to(args.device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.01)
    best=initial;history=[];start=time.perf_counter()
    for epoch in range(args.epochs):
        losses=[]
        for group in optimizer.param_groups:group['lr']=.0008*(.25+.75*.5*(1+math.cos(math.pi*epoch/args.epochs)))
        for step in range(args.steps):
            ix=[(int(t),int(rng.integers(1,len(train[t]['x'])))) for t in rng.integers(0,len(train),size=args.batch)]
            d={k:v.float().to(args.device) if v.dtype==torch.float64 else v.to(args.device) for k,v in density_batch(train,ix).items()}
            sy=slice(int(rng.integers(0,2)),None,2);sx=slice(int(rng.integers(0,2)),None,2)
            p=model(d['x'],d['g'],d['shift'],d['scale'],(sy,sx))
            y=((d['residual']/d['half'].clamp_min(1e-7)-d['shift'])/d['scale'])[:,:,sy,sx]
            valid=d['valid'][:,:,sy,sx]
            loss=(nll(p,y)*valid).sum()/valid.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
            if epoch==0 and step in (0,19):print(f'Density step {step+1} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        record=dict(epoch=epoch+1,loss=float(np.mean(losses)),elapsed=time.perf_counter()-start)
        print(json.dumps(record),flush=True)
        if (epoch+1)%5==0 or epoch+1==args.epochs:
            cpu.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()});score=evaluate(cpu,ev)
            record['eval_nll']=score;history.append(record)
            if score<best:
                best=score;torch.save(dict(state_dict=cpu.state_dict(),width=96,components=3,epoch=epoch+1,training_names=meta['train50'],
                  evidence=meta['evidence'],source_sha256=sha(Path(__file__))),DEST/f'best_{args.tag}.pt')
            (ROOT/f'dev/reports/distribution_{args.tag}_progress.json').write_text(json.dumps(dict(status='RUNNING',history=history),indent=2)+'\n')
    path=DEST/f'best_{args.tag}.pt'
    report=dict(status='COMPLETE',configuration=vars(args),initial_nll=initial,best_nll=best,history=history,
      model_sha256=sha(path) if path.exists() else None,evidence=meta['evidence'])
    (ROOT/f'dev/reports/distribution_{args.tag}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE')),flush=True)


if __name__=='__main__':main()
