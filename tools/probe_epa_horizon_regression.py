"""Exercise installed official EPA regression with explicit scratch sizes.

The test helper already accepts horizon_size. This does not patch the installed
engine, change a native checkpoint, or demonstrate a fix for our pen incident.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import numpy as np
import warp as wp
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.pen_matched_recovery import write_metadata as write


def main(a):
    started = time.perf_counter()
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh isolated helper probe required')
    out.mkdir()
    from mujoco_warp._src import collision_gjk_test as official
    from mujoco_warp._src import collision_gjk, types, collision_core
    installed = [file_pin(Path(m.__file__)) for m in [official, collision_gjk, types, collision_core]]
    receipt = freeze_sources([Path(__file__)], out / 'source_snapshot', ROOT,
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'source_receipt.json', receipt)
    original = official._geom_dist
    rows = []; first_call = None

    def recorded(*args, **kwargs):
        nonlocal first_call
        if first_call is None:
            first_call = (args, kwargs.copy())
        flag = kwargs.setdefault('overflow_out', wp.zeros(1, dtype=int, device='cuda:0'))
        result = original(*args, **kwargs)
        rows.append(dict(horizon=kwargs.get('horizon_size', types.MJ_MAX_EPAHORIZON),
            raw_mask=int(flag.numpy()[0]), distance=float(result[0]), contacts=int(result[1]),
            # Only the first ncon entries are initialized by the helper.
            positions=[np.asarray(v).tolist() for v in result[2:2 + min(2, int(result[1]))]]))
        return result

    wp.init()
    with wp.ScopedDevice('cuda:0'):
        official._geom_dist = recorded
        try:
            official.GJKTest('test_box_box_horizon').test_box_box_horizon()
            args, kwargs = first_call
            recorded(*args, **dict(kwargs, horizon_size=48, warn_overflow=False,
                                  overflow_out=wp.zeros(1, dtype=int, device='cuda:0')))
        finally:
            official._geom_dist = original
    bits = lambda r: np.asarray([r['distance']] + sum(r['positions'], []), dtype=np.float32).tobytes()
    checks = dict(official_test_assertions_pass=True,
        installed_default_24=types.MJ_MAX_EPAHORIZON == rows[0]['horizon'] == 24,
        default_no_overflow=rows[0]['raw_mask'] == 0,
        undersized_5_reports_EPA_bit=rows[1]['horizon'] == 5 and bool(rows[1]['raw_mask'] & int(types.OverflowType.EPA_HORIZON)),
        expanded_helper_48_no_overflow=rows[2]['horizon'] == 48 and rows[2]['raw_mask'] == 0,
        default_expanded_same_contact_count=rows[0]['contacts'] == rows[2]['contacts'] == 1,
        default_expanded_output_FP32_bytes_equal=bits(rows[0]) == bits(rows[2]),
        installed_source_bytes_unchanged=all(file_pin(verify_file_pin(p)) == p for p in installed))
    verify_source_snapshot(receipt, live=True)
    report = dict(schema='installed_official_EPA_horizon_helper_probe:1', installed_sources=installed,
        source_receipt=file_pin(out / 'source_receipt.json'), checks=checks, passed=sum(checks.values()), total=len(checks),
        rows=rows, actual_native_integrations=0, optimizer_steps=0, isolated_collision_queries=3,
        official_fixture_initializations=1, runner_wall_s=time.perf_counter() - started,
        scope='Actual GPU installed official box-box regression with its existing caller-provided horizon scratch size. 5 is an intentional positive overflow control;24 vs48 is a nonoverflow equivalence control. Not the pen failure fixture, a production engine variant, an admitted checkpoint migration, or a demonstrated repair.')
    write(out / 'report.json', report)
    print(json.dumps(dict(report=file_pin(out / 'report.json'), checks=[report['passed'], report['total']], rows=rows)), flush=True)
    if not all(checks.values()): raise ValueError('EPA helper regression qualification failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); main(p.parse_args())
