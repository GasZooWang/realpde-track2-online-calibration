"""Residual refiner training."""
import argparse,json,sys,time,math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'dev/shared'),str(ROOT/'submission')]
from metrics import errors,record,summary
from flow_model.student import Student

def ema(x,rate):
    out=[];v=x[0]
    for a in x:v=v*(1-rate)+a*rate;out.append(v)
    return torch.stack(out)

def features(d):
    x,p,lo,hi=[d[k] for k in ('input','prediction','lower','upper')]
    pm=p.mean(1);im=x.mean(1);iv=x.var(1,correction=0);pv=p.var(1,correction=0)
    scale=(.5*(iv+pv)).clamp_min(2.5e-7).sqrt()
    center=(lo+hi)*.5;half=(hi-lo)*.5
    units=x.new_tensor([.1,.02])[None,:,None,None]
    f=[((x-pm[:,None])/scale[:,None]).flatten(1,2),
       ((p-pm[:,None])/scale[:,None]).flatten(1,2),
       ((center-pm[:,None])/scale[:,None]).flatten(1,2),
       (half.clamp_min(1e-7)/scale[:,None]).log().flatten(1,2)]
    old_mean=torch.cat([torch.zeros_like(pm[:1]),d['target'][:-1].mean(1)-pm[:-1]])
    old_k=torch.cat([torch.zeros_like(pv[:1]),((d['target'][:-1].var(1,correction=0)+1e-8)/(pv[:-1]+1e-8)).log().clamp(-3,3)])
    for rate in (.1,.3,.7):
        f += [(ema(im,rate)-pm)/scale,(ema(iv,rate)+1e-8).sqrt()/scale,
              ema(old_mean,rate)/scale,ema(old_k,rate)]
    f += [(x[:,-5:].mean(1)-x[:,:5].mean(1))/scale,
          (x==0).float().mean(1),pv.sqrt()/scale]
    f += [pm/units,im/units,(scale/units).log()]
    return torch.cat(f,1).clamp(-10,10),scale

class Refiner(nn.Module):
    def __init__(self,channels,width=48,mode='all'):
        super().__init__();self.mode=mode
        self.net=Student(width)
        self.net.stem=nn.Conv2d(channels,width,3,padding=1)
        self.net.out=nn.Conv2d(width,83,3,padding=1)
        nn.init.zeros_(self.net.out.weight);nn.init.zeros_(self.net.out.bias)
    def forward(self,x):
        n=self.net;a=n.b0(F.silu(n.stem(x)));b=n.b1(F.silu(n.d1(a)));c=n.b2(F.silu(n.d2(b)))
        b=n.r1(b+F.interpolate(n.u1(c),size=b.shape[-2:],mode='bilinear',align_corners=False))
        a=n.r0(a+F.interpolate(n.u0(b),size=a.shape[-2:],mode='bilinear',align_corners=False))
        return n.out(a).float()

def apply(raw,p,lo,hi,scale,mode='all',strength=1.):
    z=raw*strength;pm=p.mean(1,keepdim=True)
    if mode=='interval':q=p
    else:
        mu=.5*torch.tanh(z[:,:2])*scale
        lv=1.5*torch.tanh(z[:,2:3])
        q=pm+mu[:,None]+(p-pm)*(.5*lv[:,None]).exp()
    if mode=='accuracy':return q,lo,hi
    center=(hi+lo)*.5;half=(hi-lo)*.5
    shift=.7*torch.tanh(z[:,3:43]).reshape_as(p)
    width=(.7*torch.tanh(z[:,43:])).exp().reshape_as(p)
    return q,center+half*(shift-width),center+half*(shift+width)

def loss(q,t,lo,hi,temp,phase_average=1):
    e=errors(q,t)
    acc=(.5/(.5+e)*e.new_tensor([.5,.3,.2])).sum(-1)
    valid=t!=0
    rewards=[]
    for shift in range(0,20,20//phase_average):
        shifted=t.roll(shift,1)
        smooth=torch.sigmoid((shifted-lo)/temp)*torch.sigmoid((hi-shifted)/temp)
        rewards.append((smooth*torch.exp(-(hi-lo)/.0563870259)*(shifted!=0)).flatten(1).sum(1)/valid.flatten(1).sum(1))
    reward=torch.stack(rewards).mean(0)
    scores=torch.cat([100/(1+.5*e),q.new_full((len(q),1),90.167643),100*(acc*reward)[:,None]],1)
    proxy=(scores.clamp_min(1e-4)**.6854@q.new_tensor([.5,.1,.1,.1,.2]))**(1/.6854)
    return -proxy.mean()

@torch.inference_mode()
def evaluate(model,data,strengths=(1.,)):
    rows={a:[] for a in strengths}
    for d in data:
        f,scale=d['features'],d['scale'];zs=[]
        for i in range(0,len(f),16):
            with torch.autocast('cuda',dtype=torch.float16):zs.append(model(f[i:i+16]))
        raw=torch.cat(zs)
        for a in strengths:
            q,l,h=apply(raw,d['prediction'],d['lower'],d['upper'],scale,model.mode,a)
            rows[a].append(record(q,d['target'],l,h))
    return {a:summary(v) for a,v in rows.items()}

def load(names):
    data=[]
    for name in names:
        d=torch.load(ROOT/'dev/shared/cache'/f'{Path(name).stem}.pt',map_location='cuda')
        d['features'],d['scale']=features(d);del d['input'];d['features']=d['features'].half()
        data.append(d)
    return data

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',default='all');ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--width',type=int,default=48);ap.add_argument('--seed',type=int,default=31);ap.add_argument('--phase-average',type=int,default=1);args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    split=json.loads((ROOT/'dev/backbone/split.json').read_text())
    train=load(split['train50']);val=load(split['eval25'])
    base=summary([record(d['prediction'],d['target'],d['lower'],d['upper']) for d in val])
    keys=['features','scale','prediction','target','lower','upper'];tr={k:torch.cat([d[k] for d in train]) for k in keys};del train
    n=len(tr['prediction']);channels=tr['features'].shape[1]
    model=Refiner(channels,args.width,args.mode).cuda().to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.02)
    scaler=torch.cuda.amp.GradScaler();t0=time.perf_counter();best=base['proxy'];history=[]
    name=f'refine_{args.mode}_{args.width}_{args.seed}_phase{args.phase_average}';print(name,'windows',n,'channels',channels,flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();order=torch.randperm(n,device='cuda');total=0
        lr=2e-4*(.2+.8*.5*(1+math.cos(math.pi*(epoch-1)/args.epochs)))
        for group in opt.param_groups:group['lr']=lr
        for ids in order.split(16):
            f,s,p,t,l,h=[tr[k][ids] for k in keys]
            factor=torch.empty(len(ids),1,1,1,1,device='cuda').uniform_(.85,1.15)
            f=f.clone();f[:,-6:-2]*=factor[:,0];f[:,-2:]+=factor[:,0].log()
            s=s*factor[:,0];p,t,l,h=[v*factor for v in (p,t,l,h)]
            with torch.autocast('cuda',dtype=torch.float16):raw=model(f)
            q,l,h=apply(raw,p,l,h,s,args.mode)
            temp=max(.0002,.001*(.95**epoch))
            obj=loss(q,t,l,h,temp,args.phase_average)+.002*raw.square().mean()
            opt.zero_grad(set_to_none=True);scaler.scale(obj).backward();scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update()
            total+=float(obj.detach())*len(ids)
        if epoch%2==0 or epoch==1:
            model.eval();scores=evaluate(model,val,(.25,.5,1.))
            a=max(scores,key=lambda a:scores[a]['proxy']);current=scores[a]['proxy']
            row=dict(epoch=epoch,train_loss=total/n,seconds=time.perf_counter()-t0,scores=scores,best_strength=a,gain=current-base['proxy'])
            history.append(row);print({k:row[k] for k in ('epoch','train_loss','seconds')},flush=True)
            if current>best:
                best=current;torch.save(dict(state_dict=model.state_dict(),channels=channels,width=args.width,mode=args.mode,strength=a,epoch=epoch,baseline=base,scores=scores[a]),ROOT/f'dev/shared/reports/{name}_best.pt')
            (ROOT/f'dev/shared/reports/{name}.json').write_text(json.dumps(dict(baseline=base,history=history),indent=2))
if __name__=='__main__':main()
