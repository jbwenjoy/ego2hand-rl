"""Query physical pair translation sensitivity with unchanged installed CCD."""
import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import numpy as np
import torch
import mujoco
import warp as wp
from mujoco_warp._src import collision_driver, collision_core
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.runtime.collision_geometry_artifact import load_collision_geometry
from supermimic_research.learning.pen_matched_recovery import write_metadata as write
from probe_retained_pen_collision import build_fixture


def main(a):
    started = time.perf_counter(); cp = a.config.resolve(); cfg = tomllib.loads(cp.read_text())
    if (cfg['study'], cfg['placements'], cfg['max_pair_queries'], cfg['max_native_distance_queries']) != (
            'DERIVED_PAIR_TRANSLATION_NUMERICS', ['WORLD', 'GEOM1_ORIGIN', 'MIDPOINT'], 36, 36): raise ValueError('Fixed translation diagnostic required')
    paths = {k: (cp.parent / cfg[k]).resolve() for k in ['pair_report', 'distance_report', 'protocol', 'registration']}
    pair = json.loads(paths['pair_report'].read_text())
    verify_source_snapshot(json.loads(verify_file_pin(pair['source_receipt']).read_text()), live=True)
    failed = json.loads(verify_file_pin(pair['failing_pairs']).read_text())
    if len(failed) != 1 or failed[0]['geoms'] != [1, 55]: raise ValueError('Explicit isolated pair required')
    results = json.loads(verify_file_pin(pair['capacity_results']).read_text())
    selected = [r for r in results if r['capacity'] == 48]
    if len(selected) != 12: raise ValueError('Fixed12 scenes required')
    inputs = {k: file_pin(paths[k]) for k in ['pair_report', 'distance_report', 'protocol']}
    inputs.update(parent=pair['inputs']['parent'], host_model=pair['inputs']['host_model'])
    files = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__), cp, paths['protocol'], ROOT / 'tools/probe_retained_pen_collision.py']
    def freeze(out):
        receipt = freeze_sources(files, out / 'source_snapshot', ROOT, subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        write(out / 'source_receipt.json', receipt); return receipt
    if a.register:
        out = paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT / 'runs'): raise ValueError('Fresh registration required')
        out.mkdir(); freeze(out)
        write(paths['registration'], dict(schema='derived_pair_translation_registration:1', config=file_pin(cp), inputs=inputs,
            source_receipt=file_pin(out / 'source_receipt.json'), max_pair_queries=36, max_native_distance_queries=36))
        print(json.dumps(dict(registration=file_pin(paths['registration']))), flush=True); return
    reg = json.loads(paths['registration'].read_text())
    if reg['config'] != file_pin(cp) or reg['inputs'] != inputs: raise ValueError('Registration changed')
    verify_source_snapshot(json.loads(verify_file_pin(reg['source_receipt']).read_text()), live=True)
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'): raise ValueError('Fresh diagnostic output required')
    out.mkdir(); receipt = freeze(out)
    attempted = completed = cpu_attempted = cpu_completed = 0; rows = []
    try:
        wp.init()
        with wp.ScopedDevice('cuda:0'):
            parent = torch.load(verify_file_pin(inputs['parent']), map_location='cpu', weights_only=False)
            host = mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['host_model'])))
            model, data, construction = build_fixture(host, parent['native']['native']); write(out / 'construction.json', construction)
            reference = mujoco.MjData(host)
            ms = parent['native']['native']['model']['arrays']
            host.geom_size[:] = ms['geom_size'][31 % ms['geom_size'].shape[0]].astype(np.float64)
            host.opt.ccd_iterations = 50; host.opt.ccd_tolerance = float(ms['opt.ccd_tolerance'][0])
            ctx = collision_core.create_collision_context(data.naconmax)
            pairs = np.zeros((data.naconmax, 2), dtype=np.int32); pairs[0] = [1, 55]
            pairids = np.full((data.naconmax, 2), -1, dtype=np.int32)
            ctx.collision_pair.assign(pairs); ctx.collision_pairid.assign(pairids); ctx.collision_worldid.fill_(31)
            for entry in selected:
                original = load_collision_geometry(verify_file_pin(entry['artifact']))
                p = original['geom_xpos'][31].astype(np.float64); relative = p[55] - p[1]
                for placement in cfg['placements']:
                    shift = np.zeros(3) if placement == 'WORLD' else p[1] if placement == 'GEOM1_ORIGIN' else .5 * (p[1] + p[55])
                    positions = original['geom_xpos'].copy(); positions[31] = (p - shift).astype(np.float32)
                    data.geom_xpos.assign(positions); data.geom_xmat.assign(original['geom_xmat'])
                    data.nacon.zero_(); data.ncollision.fill_(1); data.overflow.zero_()
                    attempted += 1; collision_driver._narrowphase(model, data, ctx); wp.synchronize(); completed += 1
                    count = int(data.nacon.numpy()[0]); mask = data.overflow.numpy().tolist()
                    distances = data.contact.dist.numpy()[:count].copy()
                    reference.geom_xpos[:] = positions[31].astype(np.float64)
                    reference.geom_xmat[:] = original['geom_xmat'][31].reshape(host.ngeom, 9).astype(np.float64)
                    fromto = np.full(6, np.nan); cpu_attempted += 1
                    distance = mujoco.mj_geomDistance(host, reference, 1, 55, 1., fromto); cpu_completed += 1
                    artifact = out / f"{entry['scene']}_{placement}.npz"
                    np.savez_compressed(artifact, geom_xpos=positions, geom_xmat=original['geom_xmat'],
                        GPU_distances=distances, native_fromto=fromto, raw_mask=np.asarray(mask, dtype=np.int32))
                    row = dict(scene=entry['scene'], placement=placement, shift=shift.tolist(), raw_mask=mask,
                        nacon=count, GPU_distances_m=distances.tolist(), native_signed_distance_m=float(distance),
                        finite=bool(np.isfinite(distance) and np.isfinite(fromto).all() and np.isfinite(distances).all()),
                        relative_position_max_rounding_m=float(np.max(np.abs(positions[31,55].astype(np.float64) - positions[31,1].astype(np.float64) - relative))), artifact=file_pin(artifact))
                    rows.append(row); write(out / 'rows.json', rows)
                print(json.dumps(dict(scene=entry['scene'], mask=[r['raw_mask'][31] for r in rows[-3:]], nacon=[r['nacon'] for r in rows[-3:]])), flush=True)
            verify_source_snapshot(receipt, live=True)
            report = dict(schema='derived_pair_translation_probe:1', config=file_pin(cp), registration=file_pin(paths['registration']),
                inputs=inputs, source_receipt=file_pin(out / 'source_receipt.json'), construction=file_pin(out / 'construction.json'), rows=file_pin(out / 'rows.json'),
                attempted_GPU_pair_queries=attempted, actual_GPU_pair_queries=completed,
                attempted_native_distance_queries=cpu_attempted, actual_native_distance_queries=cpu_completed,
                actual_native_integrations=0, optimizer_steps=0, FK_builds=0, wall_s=time.perf_counter() - started,
                scope='Rigid translations of explicitly derived scenes with measured FP32 rounding;unchanged installed horizon24 CCD and separate native FP64 distance reference. No production algorithm/state change, exact original reconstruction or successful repair claim.')
            write(out / 'report.json', report); print(json.dumps(dict(report=file_pin(out / 'report.json'), GPU=completed, CPU=cpu_completed)), flush=True)
    except Exception as error:
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error), attempted_GPU_pair_queries=attempted,
            actual_GPU_pair_queries=completed, attempted_native_distance_queries=cpu_attempted, actual_native_distance_queries=cpu_completed,
            actual_native_integrations=0, optimizer_steps=0, source_receipt=file_pin(out / 'source_receipt.json')))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--output', type=Path)
    p.add_argument('--register', action='store_true'); a = p.parse_args()
    if not a.register and a.output is None: p.error('--output required')
    main(a)
