"""Unified head distillation."""
import json,sys,time,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'dev/algorithm'),str(ROOT/'dev/shared'),str(ROOT/'submission')]
from train_wave import Wave,apply,evaluate
from train_refiner import features,load
from metrics import record,summary

def main():
    torch.set_num_threads(2);torch.manual_seed(53);split=json.loads((ROOT/'dev/backbone/split.json').read_text());data=[]
    for name in split['train50']:
        paths=[ROOT/'dev/shared/cache'/f'{Path(name).stem}.pt']+[ROOT/'dev/algorithm/cache_dense'/f'{Path(name).stem}_o{offset:02d}.pt' for offset in (5,10,15)]
        for path in paths:
            d=torch.load(path,map_location='cuda');d['features'],d['scale']=features(d);d['features']=d['features'].half();d['valid']=d.pop('target')!=0;del d['input']
            d.update(torch.load(ROOT/'dev/algorithm/cache_teacher'/path.name,map_location='cuda'));data.append(d)
    keys=['features','scale','prediction','lower','upper','valid','teacher_prediction','teacher_lower','teacher_upper']
    tr={}
    for key in keys:tr[key]=torch.cat([d.pop(key) for d in data])
    del data;val=load(split['eval25'])
    base=summary([record(d['prediction'],d['target'],d['lower'],d['upper']) for d in val])
    state=torch.load(ROOT/'dev/algorithm/reports/joint_32_41_dense_flip_best.pt',map_location='cuda')
    model=Wave(state['channels'],state['width'],'joint').cuda().to(memory_format=torch.channels_last);model.load_state_dict(state['state_dict'])
    with torch.no_grad():model.net.out.weight.mul_(state['strength']);model.net.out.bias.mul_(state['strength'])
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01);scaler=torch.cuda.amp.GradScaler();beg=time.perf_counter();history=[];best=-float('inf');n=len(tr['prediction']);out=ROOT/'dev/algorithm/reports'
    print('unified',n,flush=True)
    for epoch in range(1,51):
        order=torch.randperm(n,device='cuda');total=0;model.train()
        for g in opt.param_groups:g['lr']=1e-4*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/50)))
        for ids in order.split(16):
            f,s,p,l,h,valid,tq,tl,th=[tr[k][ids] for k in keys]
            for dim,component in [(-1,0),(-2,1)]:
                if bool(torch.rand((),device='cuda')<.5):
                    f,s,p,l,h,valid,tq,tl,th=[v.flip(dim) for v in (f,s,p,l,h,valid,tq,tl,th)]
                    sign=p.new_ones(2);sign[component]=-1;fs=f.new_ones(196);fs[:120]=sign.repeat(60)
                    for j in range(3):fs[160+8*j:162+8*j]=sign;fs[164+8*j:166+8*j]=sign
                    fs[184:186]=sign;fs[190:194]=sign.repeat(2);f=f*fs[None,:,None,None]
                    p,tq=[v*sign[None,None,:,None,None] for v in (p,tq)]
                    a,b=l*sign[None,None,:,None,None],h*sign[None,None,:,None,None];l,h=torch.minimum(a,b),torch.maximum(a,b)
                    a,b=tl*sign[None,None,:,None,None],th*sign[None,None,:,None,None];tl,th=torch.minimum(a,b),torch.maximum(a,b)
            with torch.autocast('cuda',dtype=torch.float16):raw=model(f)
            q,lo,hi=apply(raw,p,l,h,s,'joint',1.)
            point=((q-tq)/s[:,None]).square().mean()
            radius=((h-l)/2).clamp_min(1e-4)
            boundary=(F.smooth_l1_loss((lo-tl)/radius,torch.zeros_like(lo),reduction='none')+F.smooth_l1_loss((hi-th)/radius,torch.zeros_like(hi),reduction='none'))*valid
            obj=2*point+boundary.sum()/valid.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True);scaler.scale(obj).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();total+=float(obj.detach())*len(ids)
        if epoch==1 or epoch%5==0:
            model.eval();scores=evaluate(model,val,(.5,.75,1.));a=max(scores,key=lambda a:scores[a]['proxy']);gain=scores[a]['proxy']-base['proxy'];r=dict(epoch=epoch,loss=total/n,scores=scores,gain=gain,seconds=time.perf_counter()-beg);history.append(r);print({k:r[k] for k in ('epoch','loss','seconds')},flush=True)
            if scores[a]['proxy']>best:
                best=scores[a]['proxy'];torch.save(dict(state_dict=model.state_dict(),channels=state['channels'],width=state['width'],mode='joint',strength=a,epoch=epoch,scores=scores[a],objective='train50 ensemble distillation'),out/'unified_best.pt')
            (out/'unified.json').write_text(json.dumps(dict(baseline=base,history=history),indent=2))
if __name__=='__main__':main()
