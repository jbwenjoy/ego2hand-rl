"""Small eager projection/gradient check, CPU default; --cuda is opt-in.

This probes the pure absolute-target projection under an outer BF16 autocast.
Native policy control_target currently runs outside autocast. No actor, WM,
H8 rollout, physical controller, or deployment qualification is inferred.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
STARTED = time.perf_counter()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(device, output):
    import numpy as np
    import torch
    from target_inset import target_inset
    rows, arrays = [], {}
    for dtype in (torch.float32, torch.bfloat16):
        if time.perf_counter() - STARTED > 25:
            raise TimeoutError('25s soft limit; external child watchdog must enforce30s')
        u = torch.tensor([-1., -.99, -.5, 0., .5, .99, 1.], device=device, dtype=dtype, requires_grad=True)
        lower, upper = torch.full_like(u, -1.), torch.full_like(u, 1.)
        before = u.detach().clone()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            projected = target_inset(u, lower, upper, .020)
        projected.sum().backward()
        x = u.detach().float().cpu().numpy()
        lo, hi = np.full_like(x, -1.) + np.float32(.020), np.full_like(x, 1.) - np.float32(.020)
        expected = np.clip(x, lo, hi)
        expected_gradient = ((x > lo) & (x < hi)).astype(np.float32)
        y, gradient = projected.detach().cpu().numpy(), u.grad.float().cpu().numpy()
        checks = {'FP32_output': projected.dtype == torch.float32,
                  'independent_numpy_projection_exact': np.array_equal(y, expected),
                  'interior_one_clipped_zero_gradient_exact': np.array_equal(gradient, expected_gradient),
                  'input_not_mutated': torch.equal(u.detach(), before),
                  'finite': bool(np.isfinite(y).all() and np.isfinite(gradient).all()),
                  'contains_interior_and_strictly_clipped': bool((expected_gradient == 0).any() and (expected_gradient == 1).any()),
                  'no_nondifferentiable_boundary_case': bool(np.all((x != lo) & (x != hi)))}
        name = str(dtype).split('.')[-1]
        rows.append({'input_dtype': name, 'outer_autocast_dtype': 'bfloat16', 'checks': checks, 'passed': all(checks.values())})
        arrays.update({name + '_input_fp32_view': x, name + '_output': y, name + '_gradient': gradient,
                       name + '_expected': expected, name + '_expected_gradient': expected_gradient})
    path = output / (device + '_cases.npz')
    np.savez_compressed(path, **arrays)
    return {'passed': all(r['passed'] for r in rows), 'device': device, 'cases': rows, 'cases_sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'runtime.json').write_text(json.dumps({'pid': os.getpid(), 'parent_pid': os.getppid(), 'started_utc': datetime.now(timezone.utc).isoformat()}) + '\n', encoding='utf-8')
    sys.path.insert(0, str(ROOT / 'src'))
    report = {'status': 'running', 'script_sha256': sha(__file__), 'pure_sha256': sha(ROOT / 'src/target_inset.py'),
              'GPU_requested': args.cuda, 'training_updates': 0, 'scope': __doc__}
    try:
        import torch
        report['cpu'] = check('cpu', args.output)
        if args.cuda:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable; no CPU substitution')
            report['cuda'] = check('cuda', args.output)
        report['cuda_initialized'] = torch.cuda.is_initialized()
        report['status'] = 'passed' if all(report[d]['passed'] for d in ['cpu'] + (['cuda'] if args.cuda else [])) else 'failed'
        if report['status'] != 'passed':
            raise AssertionError('Projection/autocast/gradient contract failed')
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['wall_seconds_including_imports'] = time.perf_counter() - STARTED
        (args.output / 'checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: report[key] for key in ['status', 'GPU_requested', 'wall_seconds_including_imports']}))


if __name__ == '__main__':
    main()
