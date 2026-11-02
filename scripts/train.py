"""Train the model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def preflight(data, output):
    recipe = json.loads((ROOT/'training/recipe.json').read_text())
    if not (data/'mean_std_real.pt').is_file():
        raise ValueError('Missing official mean_std_real.pt')
    if (data/'train_real/7575_0.h5').exists():
        raise ValueError('Exclude 7575_0.h5 from the training-data mount')
    names = sorted(p.name for p in (data/'train_real').glob('*.h5'))
    if names != sorted(recipe['all81']):
        raise ValueError('train_real does not match the 81 filenames in training/recipe.json')
    if output == data or data in output.parents:
        raise ValueError('Place output_dir outside the released data directory')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Use a fresh empty output directory')
    asset = json.loads((ROOT/'official_assets/assets.json').read_text())
    checkpoint = ROOT/asset['destination']
    if not checkpoint.is_file():
        raise ValueError('Run python scripts/download_assets.py before training')
    if checkpoint.stat().st_size != asset['bytes'] or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != asset['sha256']:
        raise ValueError('Official checkpoint checksum mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--check-only', action='store_true', help='Check file names, initialization checksum, and output placement only')
    args = parser.parse_args()
    data, output = args.data_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    try:
        preflight(data, output)
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')
    if args.check_only:
        print('Preflight passed.')
        return
    import solution
    solution.train(str(data), str(output), args.device)


if __name__ == '__main__':
    main()
