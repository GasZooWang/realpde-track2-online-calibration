"""Component energy training."""
import argparse
import json
import math
import time
import torch
from model import ROOT, load_native, prepare, redistribute, SplitEnergy
from metrics import errors, record, summary


def load(names, base):
    data=[]
    for name in names:
        d=prepare(ROOT/'dev/shared/cache'/name.replace('.h5','.pt'),base)
        del d['input']
        valid=d['target']!=0
        d['valid_count']=valid.flatten(1).sum(1)
        d['reward']=(((d['target']>=d['lower'])&(d['target']<=d['upper']))
                     *torch.exp(-(d['upper']-d['lower'])/.0563870259)*valid).flatten(1).sum(1)
        d['errors']=errors(d['prediction'],d['target'])
        data.append(d)
    return data


def loss(q,target,reference_errors,reward,counts,d):
    rel=(q-target).flatten(1).norm(dim=1)/target.flatten(1).norm(dim=1).clamp_min(1e-8)
    e=torch.cat((rel[:,None],reference_errors[:,1:]),1)
    acc=(.5/(.5+e)*e.new_tensor([.5,.3,.2])).sum(1)
    scores=torch.cat((100/(1+.5*e.mean(0)),e.new_tensor([90.167643]),
                      (100*(reward*acc).sum()/counts.sum().clamp_min(1)).reshape(1)))
    proxy=(scores.pow(.6854)@scores.new_tensor([.5,.1,.1,.1,.2])).pow(1/.6854)
    return -proxy+.05*d.square().mean()


@torch.inference_mode()
def evaluate(head,data):
    rows={a:[] for a in (1.,)}
    head.eval()
    for d in data:
        with torch.autocast('cuda',dtype=torch.float16):
            delta=torch.cat([head(z) for z in d['latent'].split(32)])
        for a in rows:
            q=redistribute(d['prediction'],delta*a)
            rows[a].append(record(q,d['target'],d['lower'],d['upper']))
    return {a:summary(r) for a,r in rows.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=20)
    args=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(181)
    folder=ROOT/'dev/component_energy/reports';folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'training.json').exists()
    split=json.loads((ROOT/'dev/backbone/split.json').read_text())
    base=load_native();training=load(split['train50'],base);val=load(split['eval25'],base)
    incumbent=summary([record(d['prediction'],d['target'],d['lower'],d['upper']) for d in val])
    keys=('latent','prediction','target','errors','reward','valid_count')
    tr={k:torch.cat([d.pop(k) for d in training]) for k in keys};del training
    head=SplitEnergy().cuda();opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler();n=len(tr['latent']);best=incumbent['proxy'];start=time.perf_counter()
    report=dict(status='RUNNING',baseline_native=incumbent,windows=n,parameters=289,epochs=args.epochs,
                note='train50 training, eval25 selection; total-TKE and mean-preserving component split',history=[])
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()},strength=1.,epoch=0),folder/'best.pt')
    print(json.dumps(dict(windows=n,epochs=args.epochs)),flush=True)
    for epoch in range(1,args.epochs+1):
        head.train();total=0.
        for g in opt.param_groups:g['lr']=3e-4*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/args.epochs)))
        for ids in torch.randperm(n,device='cuda').split(16):
            z,p,t,e,r,v=(tr[k][ids] for k in keys)
            with torch.autocast('cuda',dtype=torch.float16):delta=head(z)
            q=redistribute(p,delta);obj=loss(q,t,e,r,v,delta)
            opt.zero_grad(set_to_none=True);scaler.scale(obj).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            scaler.step(opt);scaler.update();total+=float(obj.detach())*len(ids)
        if epoch==1 or epoch%2==0 or epoch==args.epochs:
            scores=evaluate(head,val);strength=max(scores,key=lambda a:scores[a]['proxy'])
            row=dict(epoch=epoch,scores=scores,strength=strength,gain_vs_native=scores[strength]['proxy']-incumbent['proxy'],loss=total/n,seconds=time.perf_counter()-start)
            report['history'].append(row);print(json.dumps({k:row[k] for k in ('epoch','loss','seconds')}),flush=True)
            if scores[strength]['proxy']>best:
                best=scores[strength]['proxy'];torch.save(dict(state_dict=head.state_dict(),strength=strength,epoch=epoch,scores=scores[strength],baseline_native=incumbent),folder/'best.pt')
            report['status']='COMPLETE' if epoch==args.epochs else 'RUNNING'
            (folder/'training.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
