"""Independent native FP64 distance queries on pinned GPU geometry inputs."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import mujoco
import numpy as np
import torch
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.runtime.collision_geometry_artifact import load_collision_geometry
from supermimic_research.learning.pen_matched_recovery import write_metadata as write


def main(a):
    started = time.perf_counter(); report_path = ROOT / 'runs/pen_epa_pair_capacity_v2/report.json'
    report = json.loads(report_path.read_text())
    verify_source_snapshot(json.loads(verify_file_pin(report['source_receipt']).read_text()), live=True)
    pairs = json.loads(verify_file_pin(report['failing_pairs']).read_text())
    if len(pairs) != 1 or pairs[0]['geoms'] != [1, 55]: raise ValueError('Explicit isolated pair1,55 required')
    results = json.loads(verify_file_pin(report['capacity_results']).read_text())
    selected = [r for r in results if r['capacity'] == 48]
    if len(selected) != 12: raise ValueError('Exactly12 registered sensitivity/control scenes required')
    host_path = verify_file_pin(report['inputs']['host_model'])
    parent = torch.load(verify_file_pin(report['inputs']['parent']), map_location='cpu', weights_only=False)
    model_arrays = parent['native']['native']['model']['arrays']
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'): raise ValueError('Fresh native distance review required')
    out.mkdir()
    receipt = freeze_sources([Path(__file__), ROOT / 'src/supermimic_research/runtime/collision_geometry_artifact.py'],
        out / 'source_snapshot', ROOT, subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'source_receipt.json', receipt)
    m = mujoco.MjModel.from_binary_path(str(host_path)); d = mujoco.MjData(m)
    # Native Model fields may be shared (leading1) or world-batched. Match
    # the exact modulo addressing used by production kernels.
    at_world = lambda key: model_arrays[key][31 % model_arrays[key].shape[0]]
    if not np.array_equal(m.geom_dataid, at_world('geom_dataid')) or not np.array_equal(m.mesh_vert, model_arrays['mesh_vert']):
        raise ValueError('Matching mesh data/vertices required for distance reference')
    if not np.array_equal(m.geom_size.astype(np.float32), at_world('geom_size')):
        raise ValueError('Host/native physical size mismatch')
    # Express the SAME captured FP32 physical shape sizes in the FP64 oracle.
    m.geom_size[:] = at_world('geom_size').astype(np.float64)
    m.opt.ccd_tolerance = float(model_arrays['opt.ccd_tolerance'][0])
    rows = []; attempted = completed = 0
    try:
        for entry in selected:
            scene = load_collision_geometry(verify_file_pin(entry['artifact']))
            d.geom_xpos[:] = scene['geom_xpos'][31].astype(np.float64)
            d.geom_xmat[:] = scene['geom_xmat'][31].reshape(m.ngeom, 9).astype(np.float64)
            values = []
            for iterations in [50, 200]:
                m.opt.ccd_iterations = iterations; fromto = np.full(6, np.nan, dtype=np.float64)
                attempted += 1
                distance = mujoco.mj_geomDistance(m, d, 1, 55, 1., fromto)
                completed += 1
                values.append(dict(iterations=iterations, signed_distance_m=float(distance), fromto=fromto.tolist(),
                    finite=bool(np.isfinite(distance) and np.isfinite(fromto).all())))
            rows.append(dict(scene=entry['scene'], source=entry['artifact'], native_FP64_queries=values,
                GPU={str(r['capacity']): dict(raw_mask=r['raw_mask'][31], nacon=r['nacon'], finite=r['contacts_finite'])
                     for r in results if r['scene'] == entry['scene']}))
            write(out / 'rows.json', rows)
        verify_source_snapshot(receipt, live=True)
        result = dict(schema='derived_pen_EPA_native_signed_distance_review:1', source_report=file_pin(report_path),
            source_receipt=file_pin(out / 'source_receipt.json'), parent=report['inputs']['parent'], host_model=file_pin(host_path),
            rows=rows, native_version=mujoco.__version__, actual_native_integrations=0, optimizer_steps=0,
            attempted_native_distance_queries=attempted, actual_native_distance_queries=completed,
            native_geometry_forward_builds=0, wall_s=time.perf_counter() - started,
            scope='Independent FP64 mj_geomDistance on the SAME saved FP32 geometry transforms/shape sizes, mesh identities/vertices checked. Native algorithm and precision differ from Warp;this is a diagnostic reference, not exact original incident recovery, force ground truth, a production fix or state admission. Iterations50/200 explicit.')
        write(out / 'report.json', result)
        print(json.dumps(dict(report=file_pin(out / 'report.json'), queries=completed,
            rows=[dict(scene=r['scene'], distance=[v['signed_distance_m'] for v in r['native_FP64_queries']], GPU=r['GPU']) for r in rows])), flush=True)
    except Exception as error:
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error), attempted_native_distance_queries=attempted,
            actual_native_distance_queries=completed, actual_native_integrations=0, optimizer_steps=0))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); main(p.parse_args())
