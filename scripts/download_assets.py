"""Download the official CNO checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def verify(path, asset):
    if path.stat().st_size != asset['bytes'] or digest(path) != asset['sha256']:
        raise ValueError('Official checkpoint length or SHA256 does not match assets.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Override the destination file')
    parser.add_argument('--check-only', action='store_true', help='Verify an existing file without network access')
    args = parser.parse_args()
    asset = json.loads((ROOT/'official_assets/assets.json').read_text())
    target = (args.output or ROOT/asset['destination']).expanduser().resolve()
    try:
        if target.exists():
            verify(target, asset)
            print(f'Verified {target} (SHA256 {asset["sha256"]})')
            return
        if args.check_only:
            raise FileNotFoundError(f'Checkpoint not found: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=target.name+'.', suffix='.part', dir=target.parent)
        temporary = Path(temporary_name)
        try:
            request = urllib.request.Request(asset['url'], headers={'User-Agent': 'AI4Silence-research-assets/1.0'})
            with os.fdopen(descriptor, 'wb') as destination, urllib.request.urlopen(request, timeout=60) as response:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    destination.write(block)
            verify(temporary, asset)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Downloaded and verified {target} (SHA256 {asset["sha256"]})')
    except (OSError, ValueError, urllib.error.URLError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
