"""Isolate all captured broadphase candidates, then test explicit capacity artifacts."""
import argparse
import importlib.util
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
from mujoco_warp._src import collision_driver as driver, collision_core, collision_convex, smooth, types
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.runtime.isolated_epa_capacity import write_capacity_artifact
from supermimic_research.runtime.collision_geometry_artifact import load_collision_geometry
from supermimic_research.learning.pen_matched_recovery import write_metadata as write
from probe_retained_pen_collision import build_fixture


def main(a):
    started = time.perf_counter(); cp = a.config.resolve(); cfg = tomllib.loads(cp.read_text())
    if (cfg['study'], cfg['world'], cfg['max_candidates'], cfg['max_failing_pairs'], cfg['capacities'], cfg['ulp_radius']) != (
            'ISOLATED_DERIVED_PAIR_EPA_CAPACITY', 31, 275, 4, [24, 48, 96], 1): raise ValueError('Fixed pair/capacity scope required')
    paths = {k: (cp.parent / cfg[k]).resolve() for k in ['scene_report', 'protocol', 'registration']}
    scene = json.loads(paths['scene_report'].read_text()); sr = {r['label']: r for r in scene['rows']}
    source = sr['ORIGINAL_ROW63_DERIVED']
    if source['raw_mask'][31] != 256 or source['affected_worlds'] != [31]: raise ValueError('Observed derived incident required')
    verify_source_snapshot(json.loads(verify_file_pin(scene['source_receipt']).read_text()), live=True)
    inputs = dict(scene_report=file_pin(paths['scene_report']), protocol=file_pin(paths['protocol']),
        parent=scene['inputs']['parent'], host_model=scene['inputs']['host_model'], scene=source['artifact'],
        installed_convex=file_pin(Path(collision_convex.__file__)), scene_builder=file_pin(ROOT / 'tools/probe_retained_pen_collision.py'))
    if 'previous_failure' in cfg:
        failed_path = (cp.parent / cfg['previous_failure']).resolve()
        failed_start = json.loads(failed_path.read_text())
        if failed_start['counts'] != dict(full_attempted=1, full_completed=1, pair_attempted=12, pair_completed=12,
                capacity_attempted=1, capacity_completed=1, FK_attempted=0, FK_completed=0):
            raise ValueError('Previous failed persistence work differs')
        inputs['previous_failure'] = file_pin(failed_path)
    for pin in inputs.values(): verify_file_pin(pin)
    files = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__), cp, paths['protocol'], ROOT / 'tools/probe_retained_pen_collision.py']
    def freeze(out):
        receipt = freeze_sources(files, out / 'source_snapshot', ROOT,
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        write(out / 'source_receipt.json', receipt); return receipt
    budget = dict(max_full_narrowphase_queries=1, max_candidate_pair_queries=275,
        max_capacity_pair_queries=144, max_FK_builds=32, native_integrations=0, optimizer_steps=0)
    if a.register:
        out = paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT / 'runs'): raise ValueError('Fresh registration required')
        out.mkdir(); freeze(out)
        write(paths['registration'], dict(schema='derived_EPA_pair_capacity_registration:1', config=file_pin(cp),
            inputs=inputs, budget=budget, source_receipt=file_pin(out / 'source_receipt.json')))
        print(json.dumps(dict(registration=file_pin(paths['registration']), budget=budget)), flush=True); return
    reg = json.loads(paths['registration'].read_text())
    if reg['inputs'] != inputs or reg['budget'] != budget or reg['config'] != file_pin(cp): raise ValueError('Registration changed')
    verify_source_snapshot(json.loads(verify_file_pin(reg['source_receipt']).read_text()), live=True)
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'): raise ValueError('Fresh output required')
    out.mkdir(); receipt = freeze(out)
    counts = dict(full_attempted=0, full_completed=0, pair_attempted=0, pair_completed=0,
        capacity_attempted=0, capacity_completed=0, FK_attempted=0, FK_completed=0)
    phase = 'construction'; results = []; isolated = []
    try:
        wp.init()
        with wp.ScopedDevice('cuda:0'):
            parent = torch.load(verify_file_pin(inputs['parent']), map_location='cpu', weights_only=False)
            host = mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['host_model'])))
            model, data, construction = build_fixture(host, parent['native']['native'])
            write(out / 'construction.json', construction)
            frame = load_collision_geometry(verify_file_pin(inputs['scene']))
            data.geom_xpos.assign(frame['geom_xpos']); data.geom_xmat.assign(frame['geom_xmat'])
            ctx = collision_core.create_collision_context(data.naconmax)
            data.ncollision.zero_(); data.nacon.zero_(); data.overflow.zero_()
            phase = 'full_broadphase_and_narrowphase'; counts['full_attempted'] += 1
            broadphase = driver.nxn_broadphase if model.opt.broadphase == types.BroadphaseType.NXN else driver.sap_broadphase
            broadphase(model, data, ctx); wp.synchronize()
            n = int(data.ncollision.numpy()[0])
            if n > data.naconmax: raise ValueError('Broadphase context truncated')
            pairs, pairids, worlds = [v.numpy()[:n].copy() for v in [ctx.collision_pair, ctx.collision_pairid, ctx.collision_worldid]]
            np.savez_compressed(out / 'broadphase_context.npz', pairs=pairs, pairids=pairids, worlds=worlds)
            driver._narrowphase(model, data, ctx); wp.synchronize(); counts['full_completed'] += 1
            raw = data.overflow.numpy().tolist()
            write(out / 'full_replay.json', dict(raw_mask=raw, nacon=int(data.nacon.numpy()[0]), ncollision=n))
            if raw != source['raw_mask']: raise ValueError('Saved geometry full collision does not reproduce derived flag')
            indices = np.flatnonzero(worlds == 31)
            if not 0 < len(indices) <= 275: raise ValueError('Candidate budget differs')

            def query(index, capacity=24):
                data.nacon.zero_(); data.ncollision.fill_(1); data.overflow.zero_()
                # Only prefix0 is consumed; all remaining context entries are explicitly unused.
                ctx.collision_pair.assign(np.concatenate([pairs[index:index + 1], np.zeros((data.naconmax - 1, 2), dtype=np.int32)]))
                ctx.collision_pairid.assign(np.concatenate([pairids[index:index + 1], np.zeros((data.naconmax - 1, 2), dtype=np.int32)]))
                ctx.collision_worldid.assign(np.full(data.naconmax, 31, dtype=np.int32))
                if capacity == 24: driver._narrowphase(model, data, ctx)
                else:
                    convex_pairs = [k for k, v in driver.MJ_COLLISION_TABLE.items() if v == driver.CollisionType.CONVEX]
                    primitive_pairs = [k for k, v in driver.MJ_COLLISION_TABLE.items() if v == driver.CollisionType.PRIMITIVE]
                    variants[capacity].convex_narrowphase(model, data, ctx, convex_pairs)
                    driver.primitive_narrowphase(model, data, ctx, primitive_pairs)
                wp.synchronize()
                mask = data.overflow.numpy().tolist(); count = int(data.nacon.numpy()[0])
                if count > data.naconmax: raise ValueError('Contact prefix truncated')
                contact = {k: getattr(data.contact, k).numpy()[:count].copy() for k in ['worldid', 'geom', 'dist', 'pos', 'frame', 'dim']}
                finite = all(np.isfinite(v).all() for v in contact.values())
                signatures = sorted(b''.join(contact[k][j].tobytes() for k in sorted(contact)) for j in range(count))
                import hashlib
                return dict(raw_mask=mask, nacon=count, contacts_finite=bool(finite),
                    contact_bytes_sha256=hashlib.sha256(b''.join(signatures)).hexdigest()), contact

            variants = {}
            for index in indices.tolist():
                phase = f'isolate_candidate_{index}'; counts['pair_attempted'] += 1
                row, contacts = query(index); counts['pair_completed'] += 1
                row.update(index=index, geoms=pairs[index].tolist(), pairid=pairids[index].tolist(),
                    names=[mujoco.mj_id2name(host, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in pairs[index]],
                    types=host.geom_type[pairs[index]].tolist())
                isolated.append(row); write(out / 'isolated_candidates.json', isolated)
            failed = [r for r in isolated if any(r['raw_mask'])]
            write(out / 'failing_pairs.json', failed)
            if not failed or len(failed) > 4: raise ValueError('No unique bounded set of failing pairs')
            if model.has_sdf_geom or model.opt.disableflags & types.DisableBit.NATIVECCD:
                raise ValueError('Explicit native convex/primitive dispatch required')
            variant_pins = []
            for capacity in [48, 96]:
                target = out / 'isolated_engine_sources' / f'capacity{capacity}' / 'collision_convex.py'
                write_capacity_artifact(Path(collision_convex.__file__), target, capacity)
                spec = importlib.util.spec_from_file_location(f'isolated_pen_convex_capacity{capacity}', target)
                module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
                spec.loader.exec_module(module); variants[capacity] = module
                variant_pins.append(dict(capacity=capacity, source=file_pin(target), only_change='EPA horizon allocation literal'))
            object_body = host.body('object/cube').id
            qa = int(host.jnt_qposadr[host.body_jntadr[object_body]])
            for failed_pair in failed:
                index = failed_pair['index']; scenes = [('INCIDENT_BASE', frame, None)]
                for component in range(4):
                    for direction in [-1, 1]:
                        q = frame['qpos'].copy(); adr = qa + 3 + component
                        q[31, adr] = np.nextafter(q[31, adr], np.float32(np.inf * direction))
                        scenes.append((f'QUAT{component}_ULP{direction:+d}', None, q))
                for label in ['PARENT_RAW', 'ORIGINAL_ROW61_DERIVED', 'ORIGINAL_ROW62_DERIVED']:
                    scenes.append((label, load_collision_geometry(verify_file_pin(sr[label]['artifact'])), None))
                for label, geometry, q in scenes:
                    phase = f'capacity_pair{index}_{label}'
                    if q is not None:
                        data.qpos.assign(q); counts['FK_attempted'] += 1
                        smooth.kinematics(model, data); wp.synchronize(); counts['FK_completed'] += 1
                        geometry = {k: getattr(data, k).numpy().copy() for k in ['geom_xpos', 'geom_xmat']}
                        geometry['qpos'] = q
                    else:
                        data.geom_xpos.assign(geometry['geom_xpos']); data.geom_xmat.assign(geometry['geom_xmat'])
                    delta = {k: float(np.max(np.abs(geometry[k][31] - frame[k][31]))) for k in ['geom_xpos', 'geom_xmat']}
                    for capacity in [24, 48, 96]:
                        counts['capacity_attempted'] += 1; row, contacts = query(index, capacity); counts['capacity_completed'] += 1
                        artifact = out / f'pair{index}_{label}_capacity{capacity}.npz'
                        np.savez_compressed(artifact, **geometry, **{'contact_' + k: v for k, v in contacts.items()}, overflow=np.asarray(row['raw_mask'], dtype=np.int32))
                        row.update(index=index, scene=label, capacity=capacity, geometry_delta=delta, artifact=file_pin(artifact))
                        results.append(row); write(out / 'capacity_results.json', results)
                    print(json.dumps(dict(pair=index, scene=label, raw_world31=[r['raw_mask'][31] for r in results[-3:]])), flush=True)
            verify_source_snapshot(receipt, live=True)
            verify_file_pin(inputs['installed_convex'])
            report = dict(schema='derived_EPA_pair_capacity_probe:1', config=file_pin(cp), registration=file_pin(paths['registration']),
                inputs=inputs, source_receipt=file_pin(out / 'source_receipt.json'), construction=file_pin(out / 'construction.json'),
                broadphase=file_pin(out / 'broadphase_context.npz'), full_replay=file_pin(out / 'full_replay.json'),
                isolated_candidates=file_pin(out / 'isolated_candidates.json'), failing_pairs=file_pin(out / 'failing_pairs.json'),
                capacity_results=file_pin(out / 'capacity_results.json'), variant_sources=variant_pins,
                actual_native_integrations=0, optimizer_steps=0, counts=counts,
                total_runner_wall_s=time.perf_counter() - started,
                scope='All-candidate isolation and isolated capacity/ULP controls on a DERIVED retained scene. Not exact original incident quaternion recovery, original SUPPORT206 attribution, production backend/state admission, policy success, ego or speedup.')
            write(out / 'report.json', report)
            print(json.dumps(dict(report=file_pin(out / 'report.json'), failing_pairs=failed, counts=counts)), flush=True)
    except Exception as error:
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error), phase=phase, counts=counts,
            source_receipt=file_pin(out / 'source_receipt.json'), actual_native_integrations=0, optimizer_steps=0,
            total_runner_wall_s=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path); p.add_argument('--register', action='store_true'); a = p.parse_args()
    if not a.register and a.output is None: p.error('--output required')
    main(a)
