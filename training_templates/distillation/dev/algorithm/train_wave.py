"""Temporal-shape head training."""
import argparse,json,sys,time,math
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'dev/shared'),str(ROOT/'submission')]
from train_refiner import Refiner,load,loss,features
from metrics import record,summary,errors

class Wave(Refiner):
    def __init__(self,channels,width=48,mode='wave'):
        super().__init__(channels,width,mode)
        self.net.out=nn.Conv2d(width,123 if mode in ('joint','interval') else 43,3,padding=1)
        nn.init.zeros_(self.net.out.weight);nn.init.zeros_(self.net.out.bias)

def apply(raw,p,l,h,s,mode,a=1.):
    raw=raw*a;mu=p.mean(1,keepdim=True);var=p.var(1,correction=0,keepdim=True)
    wave=p-mu+s[:,None]*2*torch.tanh(raw[:,3:43]).reshape_as(p)
    wave=wave-wave.mean(1,keepdim=True)
    wv=wave.var(1,correction=0,keepdim=True)
    wave=torch.where((wv>1e-12)&(var>1e-12),wave*(var/wv.clamp_min(1e-12)).sqrt(),p-mu)
    if mode=='interval':q=p
    elif mode=='wave':q=mu+wave
    else:q=mu+.5*s[:,None]*torch.tanh(raw[:,:2])[:,None]+wave*(.75*torch.tanh(raw[:,2:3])[:,None]).exp()
    if mode not in ('joint','interval'):return q,l,h
    c=(l+h)*.5;r=(h-l)*.5
    shift=.7*torch.tanh(raw[:,43:83]).reshape_as(p)
    scale=(.7*torch.tanh(raw[:,83:])).exp().reshape_as(p)
    return q,c+r*(shift-scale),c+r*(shift+scale)

def hard_loss(q,t,l,h,temp):
    e=errors(q,t);acc=(.5/(.5+e)*e.new_tensor([.5,.3,.2])).sum(-1)
    valid=t!=0
    soft=torch.sigmoid((t-l)/temp)*torch.sigmoid((h-t)/temp)
    hard=((t>=l)&(t<=h)).to(soft.dtype)
    coverage=hard+soft-soft.detach()
    reward=(coverage*torch.exp(-(h-l)/.0563870259)*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)
    scores=torch.cat([100/(1+.5*e),q.new_full((len(q),1),90.167643),100*(acc*reward)[:,None]],1)
    return -((scores.clamp_min(1e-4)**.6854@q.new_tensor([.5,.1,.1,.1,.2]))**(1/.6854)).mean()

@torch.inference_mode()
def evaluate(model,data,strengths):
    rows={a:[] for a in strengths}
    for d in data:
        zs=[]
        for f in d['features'].split(16):
            with torch.autocast('cuda',dtype=torch.float16):zs.append(model(f))
        raw=torch.cat(zs)
        for a in strengths:
            q,l,h=apply(raw,d['prediction'],d['lower'],d['upper'],d['scale'],model.mode,a)
            rows[a].append(record(q,d['target'],l,h))
    return {a:summary(v) for a,v in rows.items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['wave','moments','joint','interval'],default='wave');ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--width',type=int,default=48);ap.add_argument('--seed',type=int,default=41);ap.add_argument('--dense',action='store_true');ap.add_argument('--flip',action='store_true');ap.add_argument('--hard',action='store_true');args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    split=json.loads((ROOT/'dev/backbone/split.json').read_text())
    train=load(split['train50']);val=load(split['eval25'])
    if args.dense:
        for name in split['train50']:
            for offset in (5,10,15):
                d=torch.load(ROOT/'dev/algorithm/cache_dense'/f'{Path(name).stem}_o{offset:02d}.pt',map_location='cuda')
                d['features'],d['scale']=features(d);del d['input'];d['features']=d['features'].half();train.append(d)
    base=summary([record(d['prediction'],d['target'],d['lower'],d['upper']) for d in val])
    keys=['features','scale','prediction','target','lower','upper'];tr={k:torch.cat([d[k] for d in train]) for k in keys};del train
    n=len(tr['prediction']);channels=tr['features'].shape[1]
    model=Wave(channels,args.width,args.mode).cuda().to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.03)
    scaler=torch.cuda.amp.GradScaler();beg=time.perf_counter();best=-float('inf');history=[]
    name=f'{args.mode}_{args.width}_{args.seed}'+('_dense' if args.dense else '')+('_flip' if args.flip else '')+('_hard' if args.hard else '');out=ROOT/'dev/algorithm/reports'
    print(name,'windows',n,flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();order=torch.randperm(n,device='cuda');total=0
        lr=3e-4*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/args.epochs)))
        for group in opt.param_groups:group['lr']=lr
        for ids in order.split(16):
            f,s,p,t,l,h=[tr[k][ids] for k in keys]
            if args.flip:
                for dim, component in [(-1,0),(-2,1)]:
                    if bool(torch.rand((),device='cuda')<.5):
                        f,s,p,t,l,h=[v.flip(dim) for v in (f,s,p,t,l,h)]
                        sign=p.new_ones(2);sign[component]=-1
                        fs=f.new_ones(196)
                        fs[:120]=sign.repeat(60)
                        for rate in range(3):
                            fs[160+8*rate:162+8*rate]=sign
                            fs[164+8*rate:166+8*rate]=sign
                        fs[184:186]=sign;fs[190:194]=sign.repeat(2)
                        f=f*fs[None,:,None,None]
                        p,t=[v*sign[None,None,:,None,None] for v in (p,t)]
                        a,b=l*sign[None,None,:,None,None],h*sign[None,None,:,None,None]
                        l,h=torch.minimum(a,b),torch.maximum(a,b)
            fac=torch.empty(len(ids),1,1,1,1,device='cuda').uniform_(.8,1.2)
            f=f.clone();f[:,-6:-2]*=fac[:,0];f[:,-2:]+=fac[:,0].log()
            s=s*fac[:,0];p,t,l,h=[v*fac for v in (p,t,l,h)]
            with torch.autocast('cuda',dtype=torch.float16):raw=model(f)
            q,l,h=apply(raw,p,l,h,s,args.mode)
            obj=(hard_loss(q,t,l,h,max(.000025,.0002*.94**epoch)) if args.hard else loss(q,t,l,h,max(.0002,.001*.97**epoch)))+.002*raw.square().mean()
            opt.zero_grad(set_to_none=True);scaler.scale(obj).backward();scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();total+=float(obj.detach())*len(ids)
        if epoch%2==0 or epoch==1:
            model.eval();scores=evaluate(model,val,(.25,.5,1.))
            a=max(scores,key=lambda a:scores[a]['proxy']);current=scores[a]['proxy']
            row=dict(epoch=epoch,train_loss=total/n,seconds=time.perf_counter()-beg,scores=scores,best_strength=a,gain=current-base['proxy'])
            history.append(row);print({k:row[k] for k in ('epoch','train_loss','seconds')},flush=True)
            if current>best:
                best=current;torch.save(dict(state_dict=model.state_dict(),channels=channels,width=args.width,mode=args.mode,strength=a,epoch=epoch,baseline=base,scores=scores[a]),out/f'{name}_best.pt')
            (out/f'{name}.json').write_text(json.dumps(dict(baseline=base,history=history),indent=2))
if __name__=='__main__':main()
