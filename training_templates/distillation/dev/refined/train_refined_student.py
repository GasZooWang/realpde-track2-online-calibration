"""Refined endpoint distillation."""
import argparse,copy,json,math,sys,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'dev/aligned'),str(ROOT/'dev/algorithm'),str(ROOT/'dev/shared'),str(ROOT/'submission')]
from train_refiner import features,load
from train_wave import Wave,apply,evaluate
from train_metric_aligned import reflect,anchor_loss
from metrics import record,summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,default=ROOT/'submission_algorithm/algorithm.pt');ap.add_argument('--teacher-dir',type=Path,default=ROOT/'dev/refined/cache_refined');ap.add_argument('--mix',type=float,default=.5);ap.add_argument('--epochs',type=int,default=35);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--output',type=Path,default=ROOT/'dev/refined/reports/refined_student');args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(83);split=json.loads((ROOT/'dev/backbone/split.json').read_text())
    state=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    model=Wave(state['channels'],state['width'],'joint').cuda().to(memory_format=torch.channels_last);model.load_state_dict(state['state_dict'])
    with torch.no_grad():model.net.out.weight.mul_(state['strength']);model.net.out.bias.mul_(state['strength'])
    reference=copy.deepcopy(model).eval().requires_grad_(False);average=copy.deepcopy(reference)
    train=[]
    for name in split['train50']:
        paths=[ROOT/'dev/shared/cache'/f'{Path(name).stem}.pt']+[ROOT/'dev/algorithm/cache_dense'/f'{Path(name).stem}_o{o:02d}.pt' for o in (5,10,15)]
        for path in paths:
            d=torch.load(path,map_location='cuda');d['features'],d['scale']=features(d);d['features']=d['features'].half();d['valid']=d.pop('target')!=0;del d['input']
            ep=torch.load(args.teacher_dir/path.name,map_location='cuda');d['refined_lower']=ep['lower'];d['refined_upper']=ep['upper'];train.append(d)
    keys=['features','scale','prediction','valid','lower','upper','refined_lower','refined_upper'];tr={}
    for k in keys:tr[k]=torch.cat([d.pop(k) for d in train])
    del train;val=load(split['eval25']);baseline=summary([record(d['prediction'],d['target'],d['lower'],d['upper']) for d in val]);incumbent=evaluate(reference,val,(1.,))[1.]
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01);scaler=torch.cuda.amp.GradScaler();n=len(tr['prediction']);best=incumbent['proxy'];history=[];beg=time.perf_counter();args.output.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},windows=n)),flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();total=0;order=torch.randperm(n,device='cuda')
        for g in opt.param_groups:g['lr']=args.lr*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/args.epochs)))
        for ids in order.split(16):
            f,s,p,valid,l,h,rl,rh=[tr[k][ids] for k in keys]
            for dim,component in [(-1,0),(-2,1)]:
                if bool(torch.rand((),device='cuda')<.5):
                    f,s,p,_,l,h=reflect((f,s,p,torch.zeros_like(p),l,h),dim,component)
                    valid=valid.flip(dim);rl,rh=rl.flip(dim),rh.flip(dim);sign=p.new_ones(2);sign[component]=-1
                    a,b=rl*sign[None,None,:,None,None],rh*sign[None,None,:,None,None];rl,rh=torch.minimum(a,b),torch.maximum(a,b)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):rr=reference(f)
            with torch.no_grad():
                rq,tl,th=apply(rr,p,l,h,s,'joint');tl=tl+args.mix*(rl-tl);th=th+args.mix*(rh-th)
            with torch.autocast('cuda',dtype=torch.float16):raw=model(f)
            q,lo,hi=apply(raw,p,l,h,s,'joint');obj=anchor_loss(q,lo,hi,(rq,tl,th),s,l,h,valid)
            opt.zero_grad(set_to_none=True);scaler.scale(obj).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();total+=float(obj.detach())*len(ids)
            with torch.no_grad():
                for avg,param in zip(average.parameters(),model.parameters()):avg.lerp_(param,.005)
        if epoch==1 or epoch%5==0:
            model.eval();scores={'live':evaluate(model,val,(1.,))[1.],'ema':evaluate(average,val,(1.,))[1.]};which=max(scores,key=lambda k:scores[k]['proxy']);r=dict(epoch=epoch,loss=total/n,scores=scores,selection=which,gain_student=scores[which]['proxy']-baseline['proxy'],gain_incumbent=scores[which]['proxy']-incumbent['proxy'],seconds=time.perf_counter()-beg);history.append(r);print(json.dumps({k:r[k] for k in ('epoch','loss','selection','seconds')}),flush=True)
            if scores[which]['proxy']>best:
                best=scores[which]['proxy'];chosen=model if which=='live' else average;torch.save(dict(state_dict=chosen.state_dict(),channels=196,width=state['width'],mode='joint',strength=1.,epoch=epoch,scores=scores[which],objective='train50 refined-density interval distillation',mix=args.mix,source=str(args.checkpoint),selection=which),args.output.with_name(args.output.name+'_best.pt'))
            args.output.with_suffix('.json').write_text(json.dumps(dict(baseline=baseline,incumbent=incumbent,history=history),indent=2))
if __name__=='__main__':main()
