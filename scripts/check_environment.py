"""Check the runtime environment."""
import argparse
import json
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-cuda', action='store_true', help='Skip the CUDA availability check')
    args = parser.parse_args()
    try:
        import torch
        import numpy
        import h5py
        import triton
    except ImportError as error:
        parser.exit(1, f'{error}; use the Dockerfile or install requirements.txt\n')
    checks = {
        'linux_x86_64': platform.system() == 'Linux' and platform.machine() == 'x86_64',
        'python_3_10': sys.version_info[:2] == (3, 10),
        'torch_2_2_2': torch.__version__.split('+')[0] == '2.2.2',
        'torch_cuda_12_1': torch.version.cuda == '12.1',
        'numpy_1_26_4': numpy.__version__ == '1.26.4',
        'h5py_3_16_0': h5py.__version__ == '3.16.0',
        'triton_2_2_0': triton.__version__ == '2.2.0',
    }
    if not args.skip_cuda:
        checks['cuda_available'] = torch.cuda.is_available()
        if checks['cuda_available']:
            checks['compute_capability_at_least_80'] = torch.cuda.get_device_capability() >= (8, 0)
    print(json.dumps({'checks': checks, 'cuda_check_skipped': args.skip_cuda}, indent=2))
    if not all(checks.values()):
        parser.exit(1, 'Runtime mismatch: use the reference Docker environment.\n')


if __name__ == '__main__':
    main()
