"""Multiscale interval head training."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from neural_interval_rich import IntervalHead as RichIntervalHead
from compact_interval import IntervalHead
from accuracy_screen import ROOT,sha
from neural_interval import SIGMA


def load(work,names):
    rows=[torch.load(str(work/f'{Path(n).stem}.pt'),weights_only=True,mmap=True) for n in names]
    out={k:torch.cat([row[k] for row in rows]) for k in rows[0]}
    out['trajectory']=torch.cat([torch.full((len(row['x']),),i,dtype=torch.long) for i,row in enumerate(rows)])
    return out


@torch.inference_mode()
def evaluate(model,v,names,baseline=False):
    total=[]
    shifts=[];scales=[]
    for ix in torch.arange(len(v['x'])).split(32768):
        x=v['x'][ix]
        shift,scale=model(x[:,:28] if baseline else x)
        r=v['residual'][ix]-shift*v['half'][ix];h=scale*v['half'][ix]
        total.append(((r.abs()<=h)*torch.exp(-2*h/SIGMA)*v['weight'][ix]).double())
    reward=torch.cat(total)
    valid=v['weight']>0
    def group(mask):return float(100*reward[mask].sum()/valid[mask].sum())
    return dict(sampled_sps=group(torch.ones(len(valid),dtype=torch.bool)),
        trajectories={n:group(v['trajectory']==i) for i,n in enumerate(names)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--width',type=int,default=32);p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--extra-c45',action='store_true')
    p.add_argument('--compact28',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(140907)
    work=ROOT/'dev/cache/compact_interval'
    meta=json.loads((work/'manifest.json').read_text())
    train=load(work,meta['train']);ev=load(work,meta['held_out'])
    suffix='_fit50' if args.extra_c45 else ''
    if args.compact28:suffix+='_f28'
    extra=None
    if args.extra_c45:
        extra=json.loads((ROOT/'dev/cache/c45_neural/manifest.json').read_text())
        assert extra['status']=='COMPLETE'
        more=load(ROOT/'dev/cache/c45_neural/interval',extra['train30'])
        train={k:torch.cat([v,more[k]]) for k,v in train.items()}
    if args.compact28:
        train['x']=train['x'][:,:28]
        ev['x']=ev['x'][:,:28]
    reference_head=RichIntervalHead();reference_head.load_state_dict(torch.load(ROOT/'dev/cache/neural_interval_rich/head.pt',weights_only=True)['state_dict']);reference_head.eval()
    reference_cache_path=ROOT/'dev/cache/neural_interval_rich'/f"{Path(meta['held_out'][0]).stem}.pt"
    a=torch.load(reference_cache_path,weights_only=True);b=torch.load(work/reference_cache_path.name,weights_only=True)
    for key in ('residual','half','weight'):assert torch.equal(a[key],b[key]),key
    assert torch.equal(a['x'],b['x'][:,:28])
    baseline=evaluate(reference_head,ev,meta['held_out'],True)
    model=IntervalHead(args.width,28 if args.compact28 else 44)
    if args.width==64:
        model_state=model.state_dict();reference_state=reference_head.state_dict()
        for k,v in reference_state.items():
            if k=='network.0.weight':model_state[k].zero_();model_state[k][:,:28].copy_(v)
            else:model_state[k].copy_(v)
        model.load_state_dict(model_state)
    if args.extra_c45:
        state=torch.load(work/f'best{args.width}.pt',weights_only=True)['state_dict']
        if args.compact28:state['network.0.weight']=state['network.0.weight'][:,:28].clone()
        model.load_state_dict(state)
    model.cuda()
    train={k:v.cuda() for k,v in train.items()}
    optim=torch.optim.AdamW(model.parameters(),lr=.0015 if args.width==32 else .0005,weight_decay=.01)
    history=[];best=-1e9;begun=time.perf_counter()
    print(f'Preflight PASS; rows={len(train["x"])}, features=44, width={args.width}',flush=True)
    for epoch in range(args.epochs):
        model.train();totals=[]
        for pg in optim.param_groups:pg['lr']=(.0015 if args.width==32 else .0005)*(.15+.85*.5*(1+math.cos(math.pi*epoch/args.epochs)))
        for ix in torch.randperm(len(train['x'])).split(16384):
            shift,scale=model(train['x'][ix]);half=train['half'][ix]
            r=train['residual'][ix]-shift*half;h=half*scale
            temp=max(.0001,.001*(.75**epoch))
            cover=torch.sigmoid((h-r)/temp)*torch.sigmoid((h+r)/temp)
            loss=-(train['weight'][ix]*cover*torch.exp(-2*h/SIGMA)).mean()+.0002*(shift.square()+(scale-1).square()).mean()
            optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
            totals.append(float(loss.detach()))
        print(f'interval width{args.width} epoch {epoch+1}/{args.epochs}, loss={np.mean(totals):.6f}, elapsed={time.perf_counter()-begun:.1f}s',flush=True)
        if (epoch+1)%10==0 or epoch+1==args.epochs:
            model.eval();result=evaluate(model.cpu(),ev,meta['held_out']);model.cuda()
            delta=result['sampled_sps']-baseline['sampled_sps']
            per={n:result['trajectories'][n]-baseline['trajectories'][n] for n in meta['held_out']}
            row=dict(epoch=epoch+1,eval10=result,delta_sampled_sps=delta,wins=sum(v>0 for v in per.values()),worst=min(per.values()))
            history.append(row)
            if delta>best:
                best=delta
                torch.save(dict(state_dict=model.state_dict(),width=args.width,features=meta['features'][:model.feature_count],feature_count=model.feature_count,epoch=epoch+1,
                    train=meta['train']+(extra['train30'] if extra else []),eval10=meta['held_out'],evidence='D30 development, eval10'),work/f'best{args.width}{suffix}.pt')
            report=dict(status='COMPLETE' if epoch+1==args.epochs else 'RUNNING',width=args.width,baseline=baseline,history=history,
                best_delta_sampled_sps=best,source_sha256=sha(Path(__file__)),
                evidence='D30 eval10 interval development')
            (ROOT/f'dev/reports/compact_interval_{args.width}{suffix}.json').write_text(json.dumps(report,indent=2)+'\n')
    if extra:
        model.load_state_dict(torch.load(work/f'best{args.width}{suffix}.pt',weights_only=True)['state_dict'])
        check=load(ROOT/'dev/cache/c45_neural/interval',extra['eval15'])
        cb=evaluate(reference_head,check,extra['eval15'],True);ce=evaluate(model.cpu(),check,extra['eval15'])
        report['c45_eval15']=dict(baseline=cb,candidate=ce,delta_sampled_sps=ce['sampled_sps']-cb['sampled_sps'])
        report['training_names']=meta['train']+extra['train30']
        report['c45_data_role']=extra['evidence']
        (ROOT/f'dev/reports/compact_interval_{args.width}{suffix}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
