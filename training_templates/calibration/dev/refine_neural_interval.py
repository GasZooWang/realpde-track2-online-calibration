"""Neural interval head training."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from neural_interval import Context, IntervalHead, FEATURE_NAMES, SIGMA
from refine_learned_interval import target_raw
from cached_data import load_data
from refine_temporal_filter import anchor_model
from refine_joint_grid import vector_errors

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'dev/cache/neural_interval'
REPORT = ROOT / 'dev/reports/neural_interval.json'
TIME_SCORE = 85.445047


def proxy(scores):
    weights = np.array([.5, .1, .1, .1, .2])
    return float((np.sum(weights*np.asarray(scores)**.6854))**(1/.6854))


def splits(names):
    groups = {}
    for name in names:
        groups.setdefault(int(Path(name).stem.rsplit('_', 1)[1]), []).append(name)
    hold = set()
    for angle in sorted(groups):
        hold.update(sorted(groups[angle])[::3])
    assert len(hold) == 10
    return sorted(set(names)-hold), sorted(hold)


def stream(data):
    base, model = anchor_model(data)
    context = Context()
    prev = None
    for i in range(len(data.targets)):
        if data.first[i]:
            model.reset_ttt_state()
            context.reset()
            prev = None
        context.observe(None if prev is None else
                        (prev*data.std_target+data.mean_target)[0, ..., :2])
        with torch.inference_mode():
            pred, info = model.ttt_step(data.inputs[i:i+1], prev)
            std, mean = data.std_target[0, ..., :2], data.mean_target[0, ..., :2]
            p = pred[0, ..., :2] * std + mean
            lower = info['lower'][0, ..., :2] * std + mean
            upper = info['upper'][0, ..., :2] * std + mean
            center = .5 * (lower+upper)
            half = .5 * (upper-lower)
            inp = (data.inputs[i:i+1]*data.std_input+data.mean_input)[0, ..., :2]
            x = context.features(inp, p, center, half)
        target = target_raw(data, i)[0, ..., :2]
        errors = vector_errors(p[None], target[None])[0].numpy().astype(np.float64)
        accuracy = float(np.dot([.5,.3,.2], .5/(.5+np.maximum(errors, 0))))
        yield i, x, p, lower, upper, center, half, target, errors, accuracy
        prev = data.targets[i:i+1]
    assert base.index == len(data.targets)


def prepare(args):
    WORK.mkdir(parents=True, exist_ok=True)
    data = load_data()
    train, held = splits(data.trajectory_ids)
    generator = torch.Generator().manual_seed(130906)
    buffers = {}
    for row in stream(data):
        i, x, _, lower, upper, center, half, target, errors, accuracy = row
        name = data.simulation_ids[i]
        buf = buffers.setdefault(name, [])
        index = torch.randperm(target.numel(), generator=generator)[:args.samples]
        residual = (target-center).flatten()[index]
        selected_half = half.flatten()[index]
        valid = (target != 0).flatten()[index]
        buf.append(dict(x=x.reshape(-1, len(FEATURE_NAMES))[index].clone(),
                        residual=residual, half=selected_half,
                        weight=valid.float()*accuracy))
        if i+1 == len(data.targets) or data.simulation_ids[i+1] != name:
            payload = {key:torch.cat([v[key] for v in buf]) for key in buf[0]}
            torch.save(payload, WORK / f'{Path(name).stem}.pt')
            del buffers[name]
            print(f'prepared {name} windows={i+1}/{len(data.targets)}', flush=True)
    meta = dict(features=list(FEATURE_NAMES), train=train, held_out=held,
                samples_per_window=args.samples, seed=130906,
                evidence='D30 development, train20/eval10',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (WORK/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')


def train(args):
    meta = json.loads((WORK/'manifest.json').read_text())
    payloads = [torch.load(WORK/f'{Path(name).stem}.pt', weights_only=True)
                for name in meta['train']]
    values = {k:torch.cat([p[k] for p in payloads]).to(args.device) for k in payloads[0]}
    del payloads
    torch.manual_seed(130906)
    model = IntervalHead().to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    n = len(values['x'])
    for epoch in range(args.epochs):
        order = torch.randperm(n)
        totals = []
        for batch in order.split(16384):
            x, residual, half, weight = [values[k][batch] for k in ('x','residual','half','weight')]
            shift, scale = model(x)
            temperature = max(.00025, .001 * (.75**epoch))
            r = residual-shift*half
            h = half*scale
            cover = torch.sigmoid((h-r)/temperature)*torch.sigmoid((h+r)/temperature)
            reward = cover*torch.exp(-2*h/SIGMA)
            loss = -(weight*reward).mean()+.0002*(shift.square()+(scale-1).square()).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
            totals.append(float(loss.detach().cpu()))
        print(f'epoch {epoch+1}/{args.epochs} loss={np.mean(totals):.7f}', flush=True)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    epochs=args.epochs, train=meta['train'], held_out=meta['held_out'],
                    feature_names=list(FEATURE_NAMES)), WORK/'head.pt')


def evaluate(args):
    payload = torch.load(WORK/'head.pt', weights_only=True)
    model = IntervalHead().eval()
    model.load_state_dict(payload['state_dict'])
    data = load_data()
    records = {name:dict(windows=0, errors=np.zeros(3), valid=0,
                          reward=np.zeros(3), covered=np.zeros(3)) for name in data.trajectory_ids}
    for row in stream(data):
        i, x, p, lower, upper, center, half, target, errors, accuracy = row
        rec = records[data.simulation_ids[i]]
        with torch.inference_mode():
            shift, scale = model(x)
        valid = target != 0
        lo_bank = [lower, center+.5*shift*half-half*scale.sqrt(), center+shift*half-half*scale]
        hi_bank = [upper, center+.5*shift*half+half*scale.sqrt(), center+shift*half+half*scale]
        for j, (lo, hi) in enumerate(zip(lo_bank, hi_bank)):
            inside = (target>=lo)&(target<=hi)&valid
            rec['reward'][j] += accuracy*float((inside*torch.exp(-(hi-lo)/SIGMA)).double().sum())
            rec['covered'][j] += int(inside.sum())
        rec['windows'] += 1
        rec['errors'] += errors
        rec['valid'] += int(valid.sum())
        if (i+1)%100 == 0:
            print(f'evaluated {i+1}/{len(data.targets)}', flush=True)
    def summary(names):
        rows = [records[name] for name in names]
        valid = sum(r['valid'] for r in rows)
        errors = sum(r['errors'] for r in rows)/sum(r['windows'] for r in rows)
        scores = 100/(1+.5*errors)
        sps = 100*sum(r['reward'] for r in rows)/valid
        cov = sum(r['covered'] for r in rows)/valid
        finals = np.array([proxy([*scores, TIME_SCORE, value]) for value in sps])
        return dict(trajectories=len(names), rel_l2=float(scores[0]), tke=float(scores[1]),
                    mvpe=float(scores[2]), sps=sps.tolist(), coverage=cov.tolist(),
                    final_proxy=finals.tolist(), delta_sps=(sps-sps[0]).tolist(),
                    delta_final=(finals-finals[0]).tolist())
    report = dict(evidence='D30 development, train20/eval10',
                  candidates=['exact_calibrated','neural_half_damping','neural_full_damping'],
                  train=summary(payload['train']), held_out=summary(payload['held_out']),
                  global_development=summary(data.trajectory_ids),
                  held_out_by_angle={str(a):summary([n for n in payload['held_out'] if n.endswith(f'_{a}.h5')])
                                     for a in (0,5,10,15,20)},
                  trajectories={n:summary([n]) for n in data.trajectory_ids})
    REPORT.write_text(json.dumps(report, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','train','evaluate','self-test'])
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == 'self-test':
        x = torch.randn(20, 32, 64, 2, len(FEATURE_NAMES))
        head = IntervalHead()
        shift, scale = head(x)
        assert torch.count_nonzero(shift)==0 and torch.equal(scale, torch.ones_like(scale))
        c = Context()
        raw = torch.rand(20,32,64,2)*.1
        f0 = c.features(raw, raw, raw, torch.ones_like(raw)*.01)
        c.observe(raw+.002)
        f1 = c.features(raw, raw, raw, torch.ones_like(raw)*.01)
        assert torch.isfinite(f1).all() and not torch.equal(f0,f1)
        c.reset()
        assert torch.equal(f0, c.features(raw,raw,raw,torch.ones_like(raw)*.01))
        import scoring
        torch.manual_seed(13)
        truth = torch.rand(1,20,32,64,2)*.1
        truth[:,:,0:3] = 0
        prediction = truth+torch.randn_like(truth)*.003
        lo, hi = prediction-.004, prediction+.006
        errors = vector_errors(prediction,truth)[0].numpy()
        accuracy = float(np.dot([.5,.3,.2], .5/(.5+np.maximum(errors,0))))
        valid = truth != 0
        inside = (truth>=lo)&(truth<=hi)&valid
        reward = accuracy*float((inside*torch.exp(-(hi-lo)/SIGMA)).double().sum())/int(valid.sum())
        official, coverage = scoring.aggregate_sps(prediction.numpy(),truth.numpy(),2,
                                                   lo.numpy(),hi.numpy())
        assert abs(reward-official)<2e-7, (reward, official)
        assert abs(coverage-float(inside.sum())/int(valid.sum()))<1e-12
        print('SELF_TEST_OK: no-op; matured context; reset; official SPS formula/zero-mask parity')
    else:
        globals()[args.command](args)


if __name__ == '__main__':
    main()
