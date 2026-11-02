"""Ensemble teacher cache."""
import json,sys,time,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'dev/algorithm'),str(ROOT/'dev/shared'),str(ROOT/'submission')]
from train_wave import Wave,apply
from train_refiner import features
from train_density import Density,choose

@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    files=dict(joint='joint_32_41_dense_flip_best.pt',moments='moments_24_41_dense_flip_best.pt',interval='interval_24_41_dense_flip_hard_best.pt',density='density_best.pt')
    models={};states={};hashes={}
    for key,file in files.items():
        path=ROOT/'dev/algorithm/reports'/file;s=torch.load(path,map_location='cuda');states[key]=s;hashes[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        m=Density(s['channels'],s['width']) if key=='density' else Wave(s['channels'],s['width'],s['mode'])
        m.load_state_dict(s['state_dict']);models[key]=m.cuda().eval().requires_grad_(False)
    names=json.loads((ROOT/'dev/backbone/split.json').read_text())['train50'];paths=[]
    for name in names:
        paths.append(ROOT/'dev/shared/cache'/f'{Path(name).stem}.pt')
        paths.extend(ROOT/'dev/algorithm/cache_dense'/f'{Path(name).stem}_o{off:02d}.pt' for off in (5,10,15))
    out=ROOT/'dev/algorithm/cache_teacher';out.mkdir(exist_ok=True);manifest=dict(status='RUNNING',source_weights_sha256=hashes,files={});beg=time.perf_counter()
    for path in paths:
        d=torch.load(path,map_location='cuda');p,l,h=[d[k] for k in ('prediction','lower','upper')];f,scale=features(d);predictions={}
        for key,model in models.items():
            raw=[]
            for x in f.split(8):
                with torch.autocast('cuda',dtype=torch.float16):raw.append(model(x.half()))
            raw=torch.cat(raw);s=states[key]
            if key=='density':
                ll=[];hh=[]
                for i in range(0,len(p),8):
                    a,b=choose(raw[i:i+8],l[i:i+8],h[i:i+8]);ll.append(a);hh.append(b)
                a,b=torch.cat(ll),torch.cat(hh);predictions[key]=(p,l+s['strength']*(a-l),h+s['strength']*(b-h))
            else:predictions[key]=apply(raw,p,l,h,scale,s['mode'],s['strength'])
        jp=predictions['joint'][0];mp=predictions['moments'][0];mean=(jp+mp)/2;mu=mean.mean(1,keepdim=True)
        variance=(jp.var(1,correction=0,keepdim=True)+mp.var(1,correction=0,keepdim=True))/2
        q=mu+(mean-mu)*(variance/mean.var(1,correction=0,keepdim=True).clamp_min(1e-12)).sqrt()
        lo=(predictions['interval'][1]+predictions['density'][1])/2;hi=(predictions['interval'][2]+predictions['density'][2])/2
        torch.save(dict(teacher_prediction=q.cpu(),teacher_lower=lo.cpu(),teacher_upper=hi.cpu()),out/path.name)
        manifest['files'][path.stem]=len(p)
        if len(manifest['files'])%10==0:print('streams',len(manifest['files']),'seconds',round(time.perf_counter()-beg,1),flush=True)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    manifest['status']='COMPLETE';(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
