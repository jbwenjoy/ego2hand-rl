"""Pinned derived scenes using unchanged production collision kernels."""
import argparse
import dataclasses
import enum
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import numpy as np
import torch
import mujoco
import mujoco_warp as mjwarp
import warp as wp
from scipy.spatial.transform import Rotation
from mujoco_warp._src import smooth, types, collision_driver
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.runtime.retained_collision_scene import audit_boundary, derive_qpos, direct_joint_contract
from supermimic_research.runtime.gpu_model_snapshot import model_leaves
from supermimic_research.learning.pen_matched_recovery import write_metadata as write


def freeze(cp, paths, out):
    files = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__), cp, paths['protocol']]
    receipt = freeze_sources(files, out / 'source_snapshot', ROOT,
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'source_receipt.json', receipt)
    return receipt


def set_numerical_scalar(model, key, value):
    if not key.startswith(('opt.', 'block_dim.')):
        raise ValueError('Unexpected fresh-fixture structural scalar difference: ' + key)
    obj = model; pieces = key.split('.')
    for part in pieces[:-1]: obj = getattr(obj, part)
    old = getattr(obj, pieces[-1])
    setattr(obj, pieces[-1], type(old)(value) if isinstance(old, enum.Enum) else value)


def build_fixture(host, captured):
    saved_model, saved_data = captured['model'], captured['data']
    batches = {f.name: int(saved_model['arrays'][f.name].shape[0]) for f in dataclasses.fields(types.Model)
               if getattr(f.type, 'shape', ()) and f.type.shape[0] == '*'}
    model = mjwarp.put_model(host, batch_sizes=batches)
    arrays, scalars = model_leaves(model)
    if arrays.keys() != saved_model['arrays'].keys() or scalars.keys() != saved_model['scalars'].keys():
        raise ValueError('Fresh fixture Model layout differs')
    changes = []
    for key, old in scalars.items():
        value = saved_model['scalars'][key]
        if old != value:
            changes.append(dict(key=key, before=old, captured=value))
            set_numerical_scalar(model, key, value)
    opts = saved_data['scalars']
    kwargs = {k: opts[k] for k in ['nworld', 'naconmax', 'naccdmax', 'njmax', 'njmax_nnz']}
    if saved_data['arrays']['cM'].size: kwargs['nvmax'] = opts['nvmax']
    data = mjwarp.make_data(host, **kwargs)
    audits = {}
    for name, obj, saved in [('model', model, saved_model), ('data', data, saved_data)]:
        arrays, scalars = model_leaves(obj)
        if scalars != saved['scalars'] or arrays.keys() != saved['arrays'].keys():
            raise ValueError('Fresh fixture scalar/array keys differ: ' + name)
        for key, target in arrays.items():
            template = target.numpy(); value = saved['arrays'][key]
            if template.shape != value.shape or template.dtype != value.dtype:
                raise ValueError('Fresh fixture array shape/type differs: ' + name + '.' + key)
        for key, target in arrays.items():
            if target.size: target.assign(saved['arrays'][key])
        audits[name] = dict(arrays=len(arrays), scalars=len(scalars),
            all_array_bytes_equal=all(target.numpy().tobytes() == saved['arrays'][key].tobytes() for key, target in arrays.items()))
        if not audits[name]['all_array_bytes_equal']: raise ValueError('Fixture input byte readback failed')
    return model, data, dict(readback=audits, explicit_fresh_numerical_scalar_reconstruction=changes,
        scope='New isolated collision fixture, not a training native-state restore or source rebinding.')


def main(a):
    started = time.perf_counter(); cp = a.config.resolve(); cfg = tomllib.loads(cp.read_text())
    if (cfg['study'], cfg['world'], cfg['rows'], cfg['num_envs'], cfg['max_geometry_queries'], cfg['geometry_reconstruction_tolerance']) != (
            'DERIVED_RETAINED_COLLISION_SCENES', 31, [61, 62, 63], 32, 5, 1e-6):
        raise ValueError('Fixed registered scene budget required')
    paths = {k: (cp.parent / cfg[k]).resolve() for k in ['parent', 'failure', 'model_evidence', 'prior_ledger', 'protocol', 'registration']}
    parent = torch.load(paths['parent'], map_location='cpu', weights_only=False)
    if (parent['iteration'], parent['optimizer_steps'], parent['metadata']['arm']) != (201, 6432, 'RESIDUAL'):
        raise ValueError('Valid recovery201 parent required')
    verify_source_snapshot(parent['native']['sources'], live=True)
    failure = json.loads(paths['failure'].read_text())
    if failure['phase'] != 'RESIDUAL_consistency_252': raise ValueError('Original retained252 failure required')
    memory_path = verify_file_pin(failure['evidence']['failed_memory'])
    packet = torch.load(memory_path, map_location='cpu', weights_only=False)
    if not packet['rollout_complete'] or packet['captured_contact_rows'] != 64: raise ValueError('Complete retained memory required')
    memory = packet['memory']; direct_joint_contract(parent)
    boundaries = [audit_boundary(parent, memory, row, 31) for row in cfg['rows']]
    if not all(r['admitted_for_derived_query'] for r in boundaries): raise ValueError('Retained boundary contract failed')
    model_evidence = json.loads(paths['model_evidence'].read_text())
    host_path = verify_file_pin(model_evidence['host_model'])
    inputs = {k: file_pin(paths[k]) for k in ['parent', 'failure', 'model_evidence', 'prior_ledger', 'protocol']}
    inputs.update(memory=file_pin(memory_path), host_model=file_pin(host_path))
    budget = dict(max_batched_collision_queries=5, max_world_collision_queries=160,
                  max_batched_FK_builds=5, native_integrations=0, optimizer_steps=0)
    if a.register:
        out = paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT / 'runs'): raise ValueError('Fresh registration required')
        out.mkdir(); freeze(cp, paths, out)
        write(paths['registration'], dict(schema='retained_collision_scene_registration:1', config=file_pin(cp), inputs=inputs,
            budget=budget, temporal_boundaries=boundaries, source_receipt=file_pin(out / 'source_receipt.json')))
        print(json.dumps(dict(registration=file_pin(paths['registration']), boundaries=boundaries)), flush=True); return
    reg = json.loads(paths['registration'].read_text())
    if reg['config'] != file_pin(cp) or reg['inputs'] != inputs or reg['budget'] != budget: raise ValueError('Registered inputs changed')
    verify_source_snapshot(json.loads(verify_file_pin(reg['source_receipt']).read_text()), live=True)
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'): raise ValueError('Fresh scene output required')
    out.mkdir(); receipt = freeze(cp, paths, out)
    engine = [file_pin(Path(v.__file__)) for name, v in sorted(sys.modules.items())
              if name.startswith(('mujoco_warp.', 'warp.')) and getattr(v, '__file__', '').endswith('.py')]
    fk_attempts = fk_done = collision_attempts = collision_done = 0
    phase = 'fresh_fixture_construction'; rows = []
    try:
        wp.init()
        with wp.ScopedDevice('cuda:0'):
            host = mujoco.MjModel.from_binary_path(str(host_path))
            body = host.body('object/cube').id; qa = int(host.jnt_qposadr[host.body_jntadr[body]])
            captured = parent['native']['native']; model, data, construction = build_fixture(host, captured)
            write(out / 'fixture_construction.json', construction)
            baseline = captured['data']['arrays']['qpos']
            expected = {k: captured['data']['arrays'][k] for k in ['geom_xpos', 'geom_xmat']}
            cases = [('PARENT_RAW', baseline.copy())]
            derived_parent = baseline.copy()
            for world in range(32):
                matrix = captured['data']['arrays']['xmat'][world, body]
                xyzw = Rotation.from_matrix(matrix.astype(np.float64)).as_quat()
                derived_parent[world, qa:qa + 3] = captured['data']['arrays']['xipos'][world, body]
                derived_parent[world, qa + 3:qa + 7] = np.roll(xyzw, 1).astype(np.float32)
            cases.append(('PARENT_DERIVED', derived_parent))
            for row in cfg['rows']: cases.append((f'ORIGINAL_ROW{row}_DERIVED', derive_qpos(parent, memory, row, 31, qa)[0]))
            for label, qpos in cases:
                phase = label + '_FK'; data.qpos.assign(qpos)
                fk_attempts += 1; smooth.kinematics(model, data); wp.synchronize(); fk_done += 1
                geometry = {k: getattr(data, k).numpy().copy() for k in expected}
                errors = {k: float(np.max(np.abs(geometry[k] - expected[k]))) for k in expected}
                if label.startswith('PARENT'):
                    write(out / (label + '_geometry.json'), dict(max_errors=errors,
                        all_bytes_equal={k: geometry[k].tobytes() == expected[k].tobytes() for k in expected}))
                    if max(errors.values()) > cfg['geometry_reconstruction_tolerance']:
                        raise ValueError('Known parent geometry roundtrip exceeds registered diagnostic tolerance')
                phase = label + '_collision'; data.overflow.zero_()
                collision_attempts += 1; collision_driver.collision(model, data); wp.synchronize(); collision_done += 1
                mask = data.overflow.numpy().copy(); nacon = int(data.nacon.numpy()[0])
                valid = min(nacon, data.naconmax)
                contacts = {k: getattr(data.contact, k).numpy()[:valid].copy()
                            for k in ['worldid', 'geom', 'dist', 'pos', 'frame', 'dim']}
                path = out / (label + '.npz')
                np.savez_compressed(path, qpos=qpos, **geometry, overflow=mask, nacon=np.asarray(nacon), **{'contact_' + k: v for k, v in contacts.items()})
                row = dict(label=label, raw_mask=mask.tolist(), affected_worlds=np.flatnonzero(mask).tolist(), nacon=nacon,
                    ncollision=int(data.ncollision.numpy()[0]), artifact=file_pin(path), geometry_error_from_parent=errors,
                    scene_origin='EXACT_PARENT_INPUT' if label == 'PARENT_RAW' else 'DERIVED_NOT_CAPTURED_INCIDENT',
                    object_R_from_FK_world31=data.xmat.numpy()[31, body].tolist())
                rows.append(row); write(out / (label + '_result.json'), row)
                print(json.dumps(dict(label=label, affected_worlds=row['affected_worlds'], raw_world31=int(mask[31]), nacon=nacon)), flush=True)
            verify_source_snapshot(receipt, live=True)
            for pin in engine: verify_file_pin(pin)
            result = dict(schema='derived_retained_collision_scene_probe:1', config=file_pin(cp), registration=file_pin(paths['registration']),
                inputs=inputs, source_receipt=file_pin(out / 'source_receipt.json'), engine_sources=engine,
                temporal_boundaries=boundaries, fixture_construction=file_pin(out / 'fixture_construction.json'), rows=rows,
                actual_native_integrations=0, optimizer_steps=0, attempted_FK_builds=fk_attempts, actual_FK_builds=fk_done,
                attempted_batched_collision_queries=collision_attempts, actual_batched_collision_queries=collision_done,
                actual_world_collision_queries=collision_done * 32, total_runner_wall_s=time.perf_counter() - started,
                scope='Unchanged production collision pipeline on exact-parent and explicitly DERIVED retained scenes. No original incident raw geom capture, integration, policy optimization, production capacity variant or checkpoint migration. Numerical sensitivity and pair isolation remain required even if a derived scene overflows.')
            write(out / 'report.json', result)
            print(json.dumps(dict(report=file_pin(out / 'report.json'), FK=fk_done, collision_queries=collision_done)), flush=True)
    except Exception as error:
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error), phase=phase,
            actual_native_integrations=0, optimizer_steps=0, attempted_FK_builds=fk_attempts, actual_FK_builds=fk_done,
            attempted_batched_collision_queries=collision_attempts, actual_batched_collision_queries=collision_done,
            source_receipt=file_pin(out / 'source_receipt.json'), total_runner_wall_s=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path); p.add_argument('--register', action='store_true')
    a = p.parse_args()
    if not a.register and a.output is None: p.error('--output required')
    main(a)
