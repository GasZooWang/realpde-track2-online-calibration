"""Neural correction wrapper."""
import torch
from neural_interval import Context, IntervalHead
from neural_moments import MomentContext, MomentHead


class NeuralTTTModel:
    def __init__(self, safe, interval_state, moment_state, rich_interval=False):
        self.safe=safe
        self.device=safe.device
        interval_class,context_class=IntervalHead,Context
        if rich_interval:
            from neural_interval_rich import IntervalHead as RichHead,Context as RichContext
            interval_class,context_class=RichHead,RichContext
        self.interval=interval_class().to(self.device).eval()
        self.interval.load_state_dict(interval_state, strict=True)
        self.moments=MomentHead().to(self.device).eval()
        self.moments.load_state_dict(moment_state, strict=True)
        for head in (self.interval,self.moments):
            for p in head.parameters():p.requires_grad_(False)
        self.interval_context=context_class(self.device)
        self.moment_context=MomentContext()

    def reset_ttt_state(self):
        self.safe.reset_ttt_state()
        self.interval_context.reset()
        self.moment_context.reset()

    @torch.inference_mode()
    def ttt_step(self,input_norm,prev_target_norm=None):
        prev_raw=None
        if prev_target_norm is not None:
            prev=torch.as_tensor(prev_target_norm,device=self.device,dtype=torch.float32)
            prev_raw=(prev*self.safe._target_std+self.safe._target_mean)[0,...,:2]
        self.interval_context.observe(prev_raw)
        self.moment_context.observe(prev_raw)
        pred,info=self.safe.ttt_step(input_norm,prev_target_norm)
        inp=torch.as_tensor(input_norm,device=self.device,dtype=torch.float32)
        inp_raw=(inp*self.safe._input_std+self.safe._input_mean)[0,...,:2]
        std=self.safe._target_std[0,...,:2]
        mean=self.safe._target_mean[0,...,:2]
        p=pred[0,...,:2]*std+mean
        lower=info['lower'][0,...,:2]*std+mean
        upper=info['upper'][0,...,:2]*std+mean
        center=.5*(lower+upper);half=.5*(upper-lower)
        ix=self.interval_context.features(inp_raw,p,center,half)
        mx=self.moment_context.features(inp_raw,p)
        shift,scale=self.interval(ix)
        mu,logvar=self.moments(mx)
        pm=p.mean(0)
        corrected=pm+mu*p.new_tensor([.1,.02])+(p-pm)*torch.exp(.5*logvar)
        result=pred.clone()
        result[0,...,:2]=(corrected-mean)/std
        lo=info['lower'].clone();hi=info['upper'].clone()
        lo[0,...,:2]=(center+half*shift-half*scale-mean)/std
        hi[0,...,:2]=(center+half*shift+half*scale-mean)/std
        result_info=dict(info)
        result_info['lower']=lo;result_info['upper']=hi
        return result,result_info
