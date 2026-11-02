"""Wide moment head training."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from neural_moments import MomentHead
from wide_moments import WideMomentHead
from train_spatial_moments import load,evaluate
from accuracy_screen import ROOT,sha

SOURCE=ROOT/'dev/cache/neural_moments_head'
EXTRA=ROOT/'dev/cache/c45_neural'
WORK=ROOT/'dev/cache/wide_moments'


class EvalAdapter(torch.nn.Module):
    def __init__(self,head):super().__init__();self.head=head
    def forward(self,x,mu,lv):return self.head(x)


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--response-range',type=float,default=1.5);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(140908);WORK.mkdir(exist_ok=True,parents=True)
    meta=json.loads((SOURCE/'manifest.json').read_text());extra=json.loads((EXTRA/'manifest.json').read_text())
    assert extra['status']=='COMPLETE'
    fields=('x','delta','pv','tv','cov','norm_rel','norm_tke');parts={k:[] for k in fields}
    names=meta['train']+extra['train30'];assert not set(names)&set(meta['held_out']+extra['eval15'])
    for source,group in ((SOURCE,meta['train']),(EXTRA/'moments',extra['train30'])):
        for n in group:
            item=torch.load(str(source/f'{Path(n).stem}.pt'),weights_only=True,mmap=True)
            for k in fields:parts[k].append(item[k])
    vals={k:torch.cat(v) for k,v in parts.items()};del parts
    teacher=MomentHead().eval();teacher.load_state_dict(torch.load(SOURCE/'head.pt',weights_only=True)['state_dict'])
    ev=load(meta['held_out'],teacher)
    from cached_data import load_data
    data=load_data();counts=[]
    for n in meta['held_out']:
        ix=np.where(np.asarray(data.simulation_ids)==n)[0]
        t=data.targets[ix,...,:2]*data.std_target[...,:2]+data.mean_target[...,:2]
        counts.append((t!=0).sum((1,2,3,4)).double())
    counts=torch.cat(counts);baseline=evaluate(ev,None,counts)
    expected=json.loads((ROOT/'dev/reports/center_screen.json').read_text())['eval10']['metrics'][0]
    assert max(abs(a-b) for a,b in zip(expected,baseline['global_metrics']))<2e-5
    model=WideMomentHead(args.response_range);model.load_state_dict(teacher.state_dict())
    model.cuda()
    vals={k:v.cuda() for k,v in vals.items()}
    optim=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
    history=[];best=-1e9;start=time.perf_counter()
    print(f'Preflight PASS, train50 rows={len(vals["x"])}, response_range={args.response_range}',flush=True)
    for epoch in range(args.epochs):
        model.train();losses=[]
        for pg in optim.param_groups:pg['lr']=.0005*(.2+.8*.5*(1+math.cos(math.pi*epoch/args.epochs)))
        for ix in torch.randperm(len(vals['x'])).split(16384):
            x,delta,pv,tv,cov,nr,nk=[vals[k][ix] for k in fields]
            mu,lv=model(x);amp=torch.exp(.5*lv)
            err=(mu*x.new_tensor([.1,.02])-delta).square()+amp.square()*pv+tv-2*amp*cov
            rel=(err.sum(-1)/nr).mean();ke=.5*(torch.exp(lv)*pv-tv).sum(-1)
            loss=30*rel+(ke.square()/nk).mean()+.001*(mu.square()+lv.square()).mean()
            optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
            losses.append(float(loss.detach()))
        print(f'wide moment epoch {epoch+1}/{args.epochs}, loss={np.mean(losses):.6f}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
        if (epoch+1)%5==0 or epoch+1==args.epochs:
            model.eval();res=evaluate(ev,EvalAdapter(model.cpu()),counts);model.cuda()
            delta=np.subtract(res['global_metrics'],baseline['global_metrics']).tolist()
            per={n:res['trajectories'][n][-1]-baseline['trajectories'][n][-1] for n in meta['held_out']}
            record=dict(epoch=epoch+1,eval10=res,delta=delta,wins=sum(v>0 for v in per.values()),worst=min(per.values()))
            history.append(record)
            if delta[-1]>best:
                best=delta[-1];torch.save(dict(state_dict=model.state_dict(),response_range=args.response_range,epoch=epoch+1,
                    train=names,eval10=meta['held_out'],eval15=extra['eval15']),WORK/'best.pt')
            report=dict(status='RUNNING',baseline=baseline,history=history,best_delta=best,
                evidence='D30 development; eval10 and C45 eval15 for evaluation',
                source_sha256=sha(Path(__file__)))
            (ROOT/'dev/reports/wide_moments.json').write_text(json.dumps(report,indent=2)+'\n')
    model.load_state_dict(torch.load(WORK/'best.pt',weights_only=True)['state_dict']);model.cpu().eval()
    check=load(extra['eval15'],teacher,EXTRA/'moments',True)
    cc=torch.cat([torch.load(str(EXTRA/'moments'/f'{Path(n).stem}.pt'),weights_only=True,mmap=True)['valid'] for n in extra['eval15']])
    cb=evaluate(check,None,cc);ce=evaluate(check,EvalAdapter(model),cc)
    report.update(status='COMPLETE',c45_eval15=dict(baseline=cb,candidate=ce,
        delta=np.subtract(ce['global_metrics'],cb['global_metrics']).tolist()),training_names=names,c45_data_role=extra['evidence'])
    (ROOT/'dev/reports/wide_moments.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
