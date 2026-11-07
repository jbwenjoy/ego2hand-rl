"""Default metadata only; --run requires a separately authorized full152 result."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from full151_gradient.analysis105_v1 import metadata_preflight, run_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        result = run_analysis(args.config)
    else:
        p = metadata_preflight(args.config)
        result = {k: p[k] for k in ('status', 'config_pin', 'input_array_sha256', 'Torch_imported', 'Warp_imported')}
        result.update(WM_run=False, model_loads=0, Env_created=False, physics_steps=0)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
