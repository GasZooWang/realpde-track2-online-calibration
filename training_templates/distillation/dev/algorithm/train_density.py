"""Mixture-density training."""
import argparse,json,sys,time,math
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'dev/shared'),str(ROOT/'dev/algorithm'),str(ROOT/'submission')]
from train_refiner import Refiner,load,features
from metrics import record,summary

class Density(Refiner):
    def __init__(self,channels,width=24):
        super().__init__(channels,width,'density');self.net.out=nn.Conv2d(width,360,3,padding=1)
        nn.init.zeros_(self.net.out.weight);nn.init.zeros_(self.net.out.bias)

def distribution(raw):
    n,_,h,w=raw.shape;z=raw.reshape(n,9,20,2,h,w)
    base=z.new_tensor([-.5,0.,.5])[None,:,None,None,None,None]
    mu=base+2*torch.tanh(z[:,:3])
    sd=.75*torch.exp(1.5*torch.tanh(z[:,3:6]))
    logw=(z[:,6:9]+z.new_tensor([-1.,0.,-1.])[None,:,None,None,None,None]).log_softmax(1)
    return mu,sd,logw

def objective(raw,t,l,h):
    c=(l+h)/2;r=((h-l)/2).clamp_min(1e-5)
    z=((t-c)/r).clamp(-20,20)
    mu,sd,logw=distribution(raw)
    logp=logw-sd.log()-.5*((z[:,None]-mu)/sd).square()-.9189385332
    value=-torch.logsumexp(logp,1);valid=t!=0
    return (value*valid).sum()/valid.sum().clamp_min(1)+.002*raw.square().mean()

@torch.inference_mode()
def choose(raw,l,h):
    mu,sd,logw=distribution(raw);weight=logw.exp();c=(l+h)/2;r=(h-l)/2
    centers=[torch.zeros_like(mu[:,0]),*(mu[:,k] for k in range(3)),(mu*weight).sum(1)]
    best=torch.full_like(c,-1);blo=l.clone();bhi=h.clone()
    for mid in centers:
        for half in (.4,.7,1.,1.4,2.,3.):
            lo,hi=mid-half,mid+half
            coverage=(.5*(torch.erf((hi[:,None]-mu)/(sd*math.sqrt(2)))-torch.erf((lo[:,None]-mu)/(sd*math.sqrt(2))))*weight).sum(1)
            utility=coverage*torch.exp(-2*half*r/.0563870259)
            take=utility>best;best=torch.maximum(best,utility)
            blo=torch.where(take,c+r*lo,blo);bhi=torch.where(take,c+r*hi,bhi)
    return blo,bhi

@torch.inference_mode()
def evaluate(model,data):
    rows={a:[] for a in (.25,.5,1.)}
    for d in data:
        ls=[];hs=[]
        for i in range(0,len(d['features']),8):
            with torch.autocast('cuda',dtype=torch.float16):raw=model(d['features'][i:i+8])
            l,h=choose(raw,d['lower'][i:i+8],d['upper'][i:i+8]);ls.append(l);hs.append(h)
        l,h=torch.cat(ls),torch.cat(hs)
        for a in rows:rows[a].append(record(d['prediction'],d['target'],d['lower']+a*(l-d['lower']),d['upper']+a*(h-d['upper'])))
    return {a:summary(v) for a,v in rows.items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=30);args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(47);split=json.loads((ROOT/'dev/backbone/split.json').read_text())
    train=load(split['train50']);val=load(split['eval25'])
    for name in split['train50']:
        for offset in (5,10,15):
            d=torch.load(ROOT/'dev/algorithm/cache_dense'/f'{Path(name).stem}_o{offset:02d}.pt',map_location='cuda');d['features'],_=features(d);d['features']=d['features'].half();del d['input'];train.append(d)
    base=summary([record(d['prediction'],d['target'],d['lower'],d['upper']) for d in val])
    keys=['features','target','lower','upper'];tr={k:torch.cat([d[k] for d in train]) for k in keys};del train
    n=len(tr['target']);model=Density(tr['features'].shape[1]).cuda().to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.03);scaler=torch.cuda.amp.GradScaler();beg=time.perf_counter();history=[];best=-float('inf')
    out=ROOT/'dev/algorithm/reports';print('density',n,flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();total=0;order=torch.randperm(n,device='cuda')
        for g in opt.param_groups:g['lr']=3e-4*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/args.epochs)))
        for ids in order.split(16):
            f,t,l,h=[tr[k][ids] for k in keys]
            factor=torch.empty(len(ids),1,1,1,1,device='cuda').uniform_(.8,1.2)
            f=f.clone();f[:,-6:-2]*=factor[:,0];f[:,-2:]+=factor[:,0].log();t,l,h=[v*factor for v in (t,l,h)]
            with torch.autocast('cuda',dtype=torch.float16):raw=model(f)
            obj=objective(raw,t,l,h);opt.zero_grad(set_to_none=True);scaler.scale(obj).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();total+=float(obj.detach())*len(ids)
        if epoch==1 or epoch%5==0:
            model.eval();scores=evaluate(model,val);a=max(scores,key=lambda a:scores[a]['proxy']);gain=scores[a]['proxy']-base['proxy']
            row=dict(epoch=epoch,loss=total/n,scores=scores,gain=gain,seconds=time.perf_counter()-beg);history.append(row);print({k:row[k] for k in ('epoch','loss','seconds')},flush=True)
            if scores[a]['proxy']>best:
                best=scores[a]['proxy'];torch.save(dict(state_dict=model.state_dict(),channels=tr['features'].shape[1],width=24,epoch=epoch,strength=a,scores=scores[a]),out/'density_best.pt')
            (out/'density.json').write_text(json.dumps(dict(baseline=base,history=history),indent=2))
if __name__=='__main__':main()
