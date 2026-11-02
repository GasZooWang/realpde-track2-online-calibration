"""Accuracy screening."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from cached_data import load_data
from refine_temporal_filter import anchor_model
from refine_neural_interval import proxy, splits
from refine_joint_grid import vector_errors
from neural_wrapper import NeuralTTTModel
from neural_interval import SIGMA

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'dev/cache/screen_cache'
REPORT = ROOT / 'dev/reports/center_screen.json'
TIME = 83.170982
SOURCE = ROOT / 'ridge_neural'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Scores:
    def __init__(self, data, configs):
        self.data, self.configs = data, configs
        n, t = len(configs), len(data.trajectory_ids)
        self.errors = np.zeros((n, t, 3))
        self.reward = np.zeros((n, t))
        self.windows = np.zeros(t)
        self.valid = np.zeros(t)

    def add(self, i, bank, target, kernel):
        tr = self.data.trajectory_index[i]
        err = vector_errors(torch.stack(bank), target[None]).numpy().astype(np.float64)
        accuracy = (.5 / (.5 + np.maximum(err, 0)) * np.array([.5, .3, .2])).sum(-1)
        self.errors[:, tr] += err
        self.reward[:, tr] += accuracy * np.asarray(kernel)
        self.windows[tr] += 1
        self.valid[tr] += int((target != 0).sum())

    def summary(self, names):
        ix = [self.data.trajectory_ids.index(n) for n in names]
        sc = 100 / (1 + .5 * self.errors[:, ix].sum(1) / self.windows[ix].sum())
        sps = 100 * self.reward[:, ix].sum(1) / self.valid[ix].sum()
        final = [proxy([*s, TIME, q]) for s, q in zip(sc, sps)]
        m = np.c_[sc, sps, final]
        return dict(names=list(names), metrics=m.tolist(), delta=(m-m[0]).tolist())

    def report(self):
        train, held = splits(self.data.trajectory_ids)
        return dict(evidence='D30 development, train20/eval10',
                    configs=self.configs, columns=['Rel-L2','TKE','MVPE','SPS','fixed_Time_proxy'],
                    fixed_time=TIME, train=self.summary(train), eval10=self.summary(held),
                    all_d30=self.summary(self.data.trajectory_ids),
                    by_angle={str(a): self.summary([n for n in held if n.endswith(f'_{a}.h5')]) for a in (0,5,10,15,20)},
                    trajectories={n: self.summary([n]) for n in self.data.trajectory_ids})


def main():
    torch.set_num_threads(2)
    WORK.mkdir(exist_ok=True, parents=True)
    for name in ('neural_wrapper.py', 'neural_interval.py',
                 'neural_interval_rich.py', 'neural_moments.py'):
        assert sha(ROOT/'dev'/name) == sha(SOURCE/name), name
    data = load_data()
    base, safe = anchor_model(data)
    payload = torch.load(SOURCE/'neural_heads.pt', weights_only=True)
    model = NeuralTTTModel(safe, payload['interval_state_dict'], payload['moment_state_dict'], rich_interval=True)
    configs = ['neural'] + [f'{kind}_{a}' for kind in ('center','mean','phase') for a in (.25,.5,1.)]
    score = Scores(data, configs)
    previous = None
    buf = []
    begun = time.perf_counter()
    with torch.inference_mode():
        for i in range(len(data.targets)):
            if data.first[i]:
                model.reset_ttt_state()
                previous = None
            out, info = model.ttt_step(data.inputs[i:i+1], previous)
            assert out.shape == (1,20,32,64,3) and torch.isfinite(out).all()
            assert torch.isfinite(info['lower']).all() and torch.isfinite(info['upper']).all()
            assert (info['lower'] <= info['upper']).all()
            assert info['adapt_loss'] is None or isinstance(info['adapt_loss'], float)
            std, mean = data.std_target[0,...,:2], data.mean_target[0,...,:2]
            p = out[0,...,:2] * std + mean
            lo = info['lower'][0,...,:2] * std + mean
            hi = info['upper'][0,...,:2] * std + mean
            center, half = .5*(lo+hi), .5*(hi-lo)
            pm, pv = p.mean(0), p.var(0, correction=0)
            bank = [p]
            for kind in ('center','mean','phase'):
                for a in (.25,.5,1.):
                    q = p + a*(center-p)
                    if kind == 'mean':
                        q = p + a*(center-p).mean(0)
                    elif kind == 'phase':
                        qm, qv = q.mean(0), q.var(0, correction=0)
                        q = pm + (q-qm)*((pv+1e-10)/(qv+1e-10)).sqrt().clamp(.5,2.)
                    bank.append(q)
            target = (data.targets[i:i+1]*data.std_target+data.mean_target)[0,...,:2]
            valid = target != 0
            kernel = float((((target>=lo)&(target<=hi)&valid)*torch.exp(-(hi-lo)/SIGMA)).double().sum())
            score.add(i, bank, target, kernel)
            buf.append(dict(prediction=p.clone(), lower=lo.clone(), upper=hi.clone(), index=i, kernel=kernel))
            previous = data.targets[i:i+1]
            if i+1 == len(data.targets) or data.first[i+1]:
                name = data.simulation_ids[i]
                pack = {key: torch.stack([row[key] for row in buf]) for key in ('prediction','lower','upper')}
                pack.update(indices=[r['index'] for r in buf], kernel=[r['kernel'] for r in buf], trajectory=name,
                            format_version=1, physical_units=True)
                dest = WORK / f'{Path(name).stem}.pt'
                assert not dest.exists(), dest
                tmp = dest.with_suffix('.partial')
                torch.save(pack, tmp)
                tmp.replace(dest)
                buf=[]
                print(f'cached {name}: {i+1}/1241 windows, elapsed={time.perf_counter()-begun:.1f}s', flush=True)
    assert base.index == len(data.targets)
    report=score.report()
    report['elapsed_seconds']=time.perf_counter()-begun
    report['source_heads_sha256']=sha(SOURCE/'neural_heads.pt')
    report['source_zip_sha256']=sha(ROOT/'ridge_neural_heads.zip')
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    (WORK/'manifest.json').write_text(json.dumps(dict(format_version=1,windows=1241,trajectories=list(data.trajectory_ids),
        source_heads_sha256=report['source_heads_sha256'],source_zip_sha256=report['source_zip_sha256'],
        evidence=report['evidence'],source_script_sha256=sha(Path(__file__))),indent=2)+'\n')
    print(json.dumps({'configs':configs},indent=2),flush=True)


if __name__ == '__main__':
    main()
