"""Energy features."""
import torch
from torch.nn import functional as F

def energy_features(x,history,pm,pv):
    n,_,h,w=x.shape;u=x.new_tensor([.1,.02])[None,:,None,None]
    inp=x[:,:40].reshape(n,20,2,h,w);im=inp.mean(1);iv=inp.var(1,correction=0)
    pk=pv.sum(1,keepdim=True);ik=(iv*u.square()).sum(1,keepdim=True)
    def pool(z,k):return F.avg_pool2d(z,k,stride=1,padding=k//2,count_include_pad=False)
    ref=.5*(pk+pool(ik,5))+1e-9
    pieces=[history,pm/u,pv.clamp_min(0).sqrt()/u,im,iv.clamp_min(0).sqrt(),
      inp[:,-5:].mean(1)-inp[:,:5].mean(1),((ik+1e-10)/(pk+1e-10)).log(),(pk/1e-4+1e-8).log(),x[:,162:166]]
    for k in (3,7,15):pieces.extend([pool(iv,k).clamp_min(0).sqrt(),((pool(pk,k)+1e-10)/(pk+1e-10)).log(),((pool(ik,k)+1e-10)/(pk+1e-10)).log()])
    pieces.append(im.mean((2,3),keepdim=True).expand(-1,-1,h,w))
    pieces.append((inp.reshape(n,4,5,2,h,w).mean(2)-im[:,None]).flatten(1,2))
    f=torch.cat(pieces,1).clamp(-12,12)
    assert f.shape[1]==78,f.shape
    return f,ref

def energy_logscale(delta,pm,pv,reference,strength=1.):
    pk=pv.sum(1,keepdim=True);newk=(pk+strength*delta*reference).clamp_min(1e-12)
    lv=(newk/pk.clamp_min(1e-12)).log().clamp(-3,3)
    return lv.expand_as(pm)
