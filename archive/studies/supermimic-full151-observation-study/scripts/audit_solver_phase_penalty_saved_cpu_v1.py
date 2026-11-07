"""Independent NumPy/stdlib-only saved dense same-solver-phase review.

Source formed before actual outputs are read. Reuses frozen independent P/SDF
math, not the production converter or proxy. Raw numeric arrays are reviewed
only after Root announces terminal. Source-backed counters/order/RNG assertions
are reported separately from independently reconstructed saved-array relations.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATH = ROOT/'src/full151_gradient/observability_saved_math_v1.py'
MATH_SHA = '08c9d1d5d4f7c9d6866cdb51292a904ae7eb167cbb2bf5c093c70648622ff7a1'
PREREG_SHA = '1ec18db963bc2931920bedce3f2bf43e345544798ee4ac1e0f856ac119cea13c'
CONSUMER_SHA = '00e1aea5163c7f6d07ea75755e076d808f171d0a766acbf6a683db8a3c00b22c'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def pin(p):
    return {'path': str(Path(p).resolve(strict=True)), 'sha256': digest(p)}


def checked(p):
    path = Path(p['path']).resolve(strict=True)
    if digest(path) != p['sha256']:
        raise ValueError('Pin mismatch: '+str(path))
    return path


def same(a, b):
    return a['sha256'] == b['sha256'] and Path(a['path']).samefile(b['path'])


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def loadnp(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def f32(values, shape):
    a = np.asarray(values)
    b = a.astype('f4')
    if a.shape != shape or a.dtype.kind not in 'fiu' or not np.isfinite(a).all() or not np.array_equal(a.astype('f8'), b.astype('f8')):
        raise ValueError('Saved original FP32 array representation/shape invalid')
    return b


def equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def quat_matrix(q):
    """Independent FP32 wxyz normalization and native column-vector formula."""
    q = np.asarray(q, dtype='f4')
    if q.shape[-1] != 4 or not np.isfinite(q).all() or np.any(abs(np.linalg.norm(q.astype('f8'), axis=-1)-1) > 1e-5):
        raise ValueError('Malformed saved object quaternion')
    q = q/np.sqrt(np.sum(q*q, axis=-1, keepdims=True))
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3, 3))


def reconstruct58(q, v, pp, R, handq, handv, oq, ov):
    # This NumPy identity check does not call native SimReader.state or FK.
    obj = quat_matrix(q[:, oq+3:oq+7])
    inv = R.swapaxes(-1, -2)
    mv = lambda A, b: np.sum(A*b[..., None, :], axis=-1)
    mm = lambda A, B: np.sum(A[..., :, :, None]*B[..., None, :, :], axis=-2)
    return np.concatenate((q[:, handq], v[:, handv], mv(inv, q[:, oq:oq+3]-pp),
        mm(inv, obj).reshape(-1, 9), mv(inv, v[:, ov:ov+3]),
        mv(inv, mv(obj, v[:, ov+3:ov+6]))), axis=-1)


def contact_record(row, hand, objects, covered, minima=None):
    contacts = row['contacts']; n = row['nacon']; raw = row['raw_wp_contact']
    if type(n) is not int or not 0 <= n <= 192 or type(row['nefc']) is not int or not 0 <= row['nefc'] <= 4096:
        raise ValueError('Invalid saved perstep capacities')
    if len(contacts) != n or set(raw) != {'geom', 'type', 'worldid', 'efc_address'}:
        raise ValueError('Missing/malformed contact inventory is not empty')
    if n == 0:
        if contacts or any(type(v) is not list or v for v in raw.values()):
            raise ValueError('Malformed knownempty raw contact')
    else:
        g, w, typ = (np.asarray(raw[k]) for k in ('geom', 'worldid', 'type'))
        if g.shape != (n, 2) or w.shape != (n,) or typ.shape != (n,) or any(v.dtype.kind not in 'iu' for v in (g, w, typ)) or (w != 0).any() or (g < 0).any() or (typ < 0).any():
            raise ValueError('Invalid raw contact geom/world/type')
        if not np.array_equal(g, np.asarray([c['geom'] for c in contacts])):
            raise ValueError('Saved WP versus CPU contact order mismatch')
    pairs = []
    for c in contacts:
        a, b = c['geom']; d = c['dist']
        if type(a) is not int or type(b) is not int or type(d) not in (int, float) or not np.isfinite(d):
            raise ValueError('Unknown contact geometry/distance')
        if (a in hand and b in objects) or (b in hand and a in objects):
            pairs.append({'geom': [a, b], 'hand_geom': a if a in hand else b, 'depth_m': max(0., -float(d))})
    maximum = max((p['depth_m'] for p in pairs), default=0.)
    tied = [p for p in pairs if p['depth_m'] == maximum]
    violating = [p for p in pairs if p['depth_m'] > .003]
    def coverage(items):
        ids = sorted({p['hand_geom'] for p in items})
        return {'hand_geom_ids': ids, 'geom_set_gap_ids': [g for g in ids if g not in covered],
            # Producer field intentionally uses nominal F64 -.003 comparison;
            # source-FP32 boundary equality is independently retained elsewhere.
            'covered_proxy_undersensitivity_ids': [] if minima is None else [g for g in ids if g in covered and minima[g] >= -.003],
            'overlapping_not_exclusive': True}
    return {'native_generated_HO_depth_m': maximum, 'native_generated_HO_count': len(pairs),
        'native_HO_depth_gt3mm': maximum > .003, 'all_exact_tied_deepest_pairs': tied,
        'all_violating_pairs_SECONDARY': violating,
        'primary_deepest_tie_coverage': coverage(tied if maximum > .003 else []),
        'secondary_all_violating_coverage': coverage(violating)}


def summary(rows):
    return {'rows': len(rows), 'native_HO_depth_gt3mm': sum(r['native_HO_depth_gt3mm'] for r in rows),
        'proxy_P_positive': sum(r['proxy_P_positive'] for r in rows),
        'both_positive': sum(r['native_HO_depth_gt3mm'] and r['proxy_P_positive'] for r in rows),
        'native_positive_proxy_zero': sum(r['primary_miss'] for r in rows),
        'proxy_alert_without_nativeHO3mm': sum(r['proxy_alert_without_observed_native_HO3mm'] for r in rows),
        'both_nonpositive': sum(not r['native_HO_depth_gt3mm'] and not r['proxy_P_positive'] for r in rows),
        'primary_deepest_tie_geom_gap_rows': sum(r['primary_miss'] and bool(r['primary_deepest_tie_coverage']['geom_set_gap_ids']) for r in rows),
        'primary_deepest_tie_covered_undersensitivity_rows': sum(r['primary_miss'] and bool(r['primary_deepest_tie_coverage']['covered_proxy_undersensitivity_ids']) for r in rows),
        'overlapping_categories_not_independent_N': True}


def audit(config_path, run, out):
    if out.exists():
        raise ValueError('Fresh audit output required')
    out.mkdir(parents=True)
    source = pin(__file__)
    (out/'executed_checker_source.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'equations_frozen_before_numeric_read.json', {'checker': source, 'math': pin(MATH),
        'UTC': datetime.now(timezone.utc).isoformat(), 'numeric_output_read': False,
        'scope': __doc__, 'FP32_SDF_arithmetic': 'Original P/maxrelu/.003/.001 and BP-.1; exact bits expected',
        'conversion': 'q/qd exact bits; remaining18 and normalized solverR <=8eps32componentwise',
        'thresholds': 'native nominal .003; source F32 scalar equality separate; no posthoc thresholds'})
    if digest(MATH) != MATH_SHA:
        raise ValueError('Independent frozen math changed')
    spec = importlib.util.spec_from_file_location('independent_observability_math', MATH)
    math = importlib.util.module_from_spec(spec); spec.loader.exec_module(math)
    checks, details = {}, {}
    def gate(k, v):
        checks[k] = bool(v)
    cfg, report = read(config_path), read(run/'report.json')
    proto = read(checked(cfg['protocol'])); inputs = proto['inputs']
    gate('exact_root_config_savedcopy_schema_protocol',
         cfg['schema'] == 'full151105_solver_phase_penalty_CPU_run:1'
         and cfg['CPU_run_authorized'] is True and cfg['cpu_threads'] == 1
         and cfg == read(run/'config.json') and cfg['protocol']['sha256'] == PREREG_SHA
         and same(report['config'], pin(config_path)) and same(report['protocol'], cfg['protocol'])
         and Path(cfg['output']).samefile(run) and proto['revision'] == 2)
    gate('consumer_pin_source_inventory_and_protocol_copy',
         cfg['source_pins'] == report['source_pins'] == read(run/'source_inventory.json')
         and read(run/'protocol.json') == proto
         and any(p['sha256'] == CONSUMER_SHA and Path(p['path']).name == 'penalty_observability105_v1.py' for p in cfg['source_pins'].values()))
    allpins = [*cfg['source_pins'].values(), *inputs.values(), cfg['protocol'], pin(config_path)]
    for p in allpins:
        checked(p)
    gate('all_source_and_preregistered_input_pins_current', True)
    with zipfile.ZipFile(run/'execution_sources.zip') as z:
        zipok = len(z.namelist()) == len(cfg['source_pins'])
        for i, p in enumerate(cfg['source_pins'].values()):
            name = '%04d_%s' % (i, Path(p['path']).name)
            zipok &= hashlib.sha256(z.read(name)).hexdigest() == p['sha256']
    gate('archived_source_bytes_exact', zipok)
    outputpins = read(run/'output_pins.json')
    for p in outputpins.values():
        checked(p)
    gate('all_original_output_pins_exact', True)
    physical = read(checked(inputs['physical_saved_independent1762']))
    gate('original_physical_saved1762_bound_not_repeated', physical['schema'] == 'full151105_full_serial_saved_independent:1'
         and physical['passed'] == physical['total'] and all(v is True for v in physical['checks'].values())
         and physical['completed_branches'] == 152 and physical['episode_step_returned'] == 1808
         and physical['physical_gradient_qualified'] is False and same(physical['config'], inputs['physical_config']))
    a = loadnp(run/'converted_state58.npz'); sample = loadnp(run/'proxy_sample_buffers.npz')
    score = loadnp(run/'all1808_proxy_scores.npz'); mapping = read(run/'geometry_ID_mapping.json')
    samples = jsonl(checked(inputs['physical_samples'])); rewards = jsonl(checked(inputs['reward_snapshots']))
    rows = read(run/'rows_all1808.json'); native_before = read(run/'native_depth_before_proxy.json')
    rewardmap = {(r['path'], r['k'], r['branch_id']): r for r in rewards}
    groups = {}
    coverage = {'donor': 0, 'baseline': 0, 'direction': 0}
    expected_clock = np.zeros(601, dtype='f4')
    for i in range(600):
        expected_clock[i+1] = np.float32(expected_clock[i]+np.float32(.005))
    timeline = len(samples) == len(rows) == len(native_before) == 1808 and len(rewards) == len(rewardmap) == 452
    for i, s in enumerate(samples):
        k, sub = s['phase'], s['substep']; key = (s['path'], k, s['branch_id'])
        groups.setdefault(key, []).append(i); coverage[s['mode']] += 1
        timeline &= (s['path'] in ('A0', 'A2') and type(k) is int and 0 <= k < 150
            and type(sub) is int and 1 <= sub <= 4 and s['source_frame'] == 200+k
            and s['canonical_pre_tick'] == 4*k+sub-1 and s['canonical_post_tick'] == 4*k+sub
            and s['pre_time'] == float(expected_clock[4*k+sub-1]) and s['post_time'] == float(expected_clock[4*k+sub])
            and s['external_force_max'] == 0
            and s['phase_contract'] == 'solver quantities preintegration; q/qvel/time postintegration; before scene.update and Manager forward')
    timeline &= len(groups) == 452 and set(groups) == set(rewardmap) and coverage == {'donor': 1200, 'baseline': 96, 'direction': 512}
    for key, ids in groups.items():
        timeline &= ids == list(range(ids[0], ids[0]+4)) and [samples[i]['substep'] for i in ids] == [1, 2, 3, 4]
        timeline &= all(np.array_equal(np.asarray(samples[x][p], dtype='f4'), np.asarray(samples[y][q], dtype='f4'))
            for x, y in zip(ids, ids[1:]) for p, q in [('post_qpos', 'pre_qpos'), ('post_qvel', 'pre_qvel')])
        timeline &= all(np.array_equal(np.asarray(samples[i]['ctrl'], dtype='f4'), np.asarray(samples[ids[0]]['ctrl'], dtype='f4')) for i in ids)
    gate('all1808_raw_groups_452_clock_prepost_heldu_noassist', timeline)
    q = f32([s['pre_qpos'] for s in samples], (1808, 27)); v = f32([s['pre_qvel'] for s in samples], (1808, 26))
    last = np.asarray([ids[-1] for ids in groups.values()], dtype='i8')
    postq = f32([samples[i]['post_qpos'] for i in last], (452, 27)); postv = f32([samples[i]['post_qvel'] for i in last], (452, 26))
    gate('all_saved_raw_prepost_arrays_exact_and2260_count', equal(q, a['pre_qpos']) and equal(v, a['pre_qvel'])
         and equal(postq, a['post20_qpos']) and equal(postv, a['post20_qvel']) and equal(last, a['group_last_indices'])
         and a['pre_x'].shape == (1808, 58) and a['post20_x'].shape == (452, 58))
    pp = np.empty((1808, 3), dtype='f4'); R = np.empty((1808, 3, 3), dtype='f4'); gpp = []; gR = []; refs = []
    palm_constant = True; palms = {}
    for key, ids in groups.items():
        rw = rewardmap[key]
        p = f32(rw['state20_reader_palm_pos'], (1, 3))[0]; r = f32(rw['state20_reader_palm_rot'], (1, 3, 3))[0]
        pp[ids] = p; R[ids] = r; gpp.append(p); gR.append(r); refs.append(rw['physics'])
        if key[0] in palms:
            palm_constant &= equal(palms[key[0]][0], p) and equal(palms[key[0]][1], r)
        else:
            palms[key[0]] = (p.copy(), r.copy())
    native = loadnp(checked(inputs['native_data'])); wrist = native['wrist_pose_w'].astype('f4')
    palm_ok = wrist.shape == (151, 7) and np.array_equal(wrist, np.broadcast_to(wrist[:1], wrist.shape))
    for name, (p, r) in palms.items():
        palm_ok &= bool(math.arithmetic_bound(p, wrist[0, :3])[0].all() and math.arithmetic_bound(r, quat_matrix(wrist[0, 3:]))[0].all())
        for filename in ('initial_readback.json', 'runtime_reset_readback.json'):
            rd = read(checked(inputs[name+'__'+filename]))
            palm_ok &= equal(np.asarray(rd['mocap_pos'], dtype='f4').reshape(3), wrist[0, :3]) and equal(np.asarray(rd['mocap_quat'], dtype='f4').reshape(4), wrist[0, 3:])
    gate('all452_recorded_reader_frames_constant_native151_start_reset_bound', palm_ok and palm_constant and equal(pp, a['palm_pos']) and equal(R, a['palm_rot'])
         and bool(math.arithmetic_bound(R.swapaxes(-1, -2).astype('f8')@R.astype('f8'), np.broadcast_to(np.eye(3), R.shape))[0].all()) and bool((np.linalg.det(R.astype('f8')) > 0).all()))
    handq, handv = a['hand_qpos'], a['hand_dof']; oq, ov = int(a['object_qpos_start']), int(a['object_dof_start'])
    gate('typed_unique_disjoint_native_state_mapping', handq.shape == handv.shape == (20,) and handq.dtype.kind in 'iu' and handv.dtype.kind in 'iu'
         and len(set(handq.tolist())) == len(set(handv.tolist())) == 20 and set(handq).isdisjoint(range(oq, oq+7))
         and set(handv).isdisjoint(range(ov, ov+6)) and 0 <= oq <= 20 and 0 <= ov <= 20
         and ((0 <= handq) & (handq < 27)).all() and ((0 <= handv) & (handv < 26)).all())
    allq, allv = np.concatenate((q, postq)), np.concatenate((v, postv))
    allpp, allR = np.concatenate((pp, np.asarray(gpp, dtype='f4'))), np.concatenate((R, np.asarray(gR, dtype='f4')))
    independent = reconstruct58(allq, allv, allpp, allR, handq, handv, oq, ov)
    actual = np.concatenate((a['pre_x'], a['post20_x']))
    codec = math.state58_comparison(independent, actual); details['all2260_NumPy_formula_vs_native_saved'] = codec
    gate('all2260_saved_native58_formula_hand_world_body_angular_relations', codec['q_qd_exact_bits'] and codec['other18_within_arithmetic'])
    reference = f32(refs, (452, 1, 58)).reshape(452, 58)
    endpoint = math.state58_comparison(a['post20_x'], reference); details['all452_endpoint'] = endpoint
    gate('all452_endpoint_original_reward_physics58_q_qd_exact_other8eps', equal(reference, a['post20_reference']) and endpoint['q_qd_exact_bits'] and endpoint['other18_within_arithmetic'])
    anchor = loadnp(checked(inputs['physical_arrays']))['snapshot_x']
    selected, targets = [], []
    for key, ids in groups.items():
        if key[2] is not None or key[1] in (0, 49, 99, 149):
            sid = ('A0', 'A2').index(key[0])*4+(0, 49, 99, 149).index(key[1])
            selected.append(a['pre_x'][ids[0]]); targets.append(anchor[sid])
    anchors = math.state58_comparison(np.asarray(selected), np.asarray(targets)); details['all160_anchor'] = anchors
    gate('all160_firstanchors_not_extra_conversions', len(selected) == 160 and anchors['q_qd_exact_bits'] and anchors['other18_within_arithmetic'])
    solver = f32([s['object_solver_pose'] for s in samples], (1808, 7))
    posok, poserr = math.arithmetic_bound(solver[:, :3], q[:, oq:oq+3])
    rotok, roterr = math.arithmetic_bound(quat_matrix(solver[:, 3:]), quat_matrix(q[:, oq+3:oq+7]))
    details['solver_pre_pose_maxerrors'] = {'position_m': float(poserr.max()), 'R_components': float(roterr.max())}
    gate('all1808_solverpose_matches_PRE_not_post_normalized_signsafe', equal(solver, a['object_solver_pose']) and bool(posok.all() and rotok.all()))
    prereq = read(run/'converter_qualification.json')
    gate('converter_gate_saved_values_and_source_order_before_proxy', prereq == report['converter_gates'] and prereq['all_converter_gates_passed'] is True
         and prereq['proxy_calls_before_gates'] == 0
         and prereq['all452_post20_endpoint_maxerror'] == endpoint['other18_max_abs']
         and prereq['all160_leading_anchor_maxerror'] == anchors['other18_max_abs']
         and prereq['all1808_solver_pre_position_maxerror'] == float(poserr.max())
         and prereq['all1808_solver_pre_R_maxerror'] <= 8*np.finfo('f4').eps)
    for path in ('A0', 'A2'):
        expected = loadnp(checked(inputs[path+'__expected_context_fields.npz']))
        ctx = loadnp(checked(inputs['actual_context'])); got = loadnp(checked(inputs[path+'__actual_context_readbacks.npz']))
        fields = set(expected)
        ctxok = len(fields) == 8 and set(got) == {stage+'__'+f for stage in ('after_copy', 'after_set_const_0') for f in fields}
        for f in fields:
            want = ctx[f][0]; ctxok &= np.array_equal(expected[f], want)
            for stage in ('after_copy', 'after_set_const_0'):
                z = got[stage+'__'+f]
                ctxok &= z.ndim == want.ndim+1 and z.shape[1:] == want.shape and np.array_equal(z, np.broadcast_to(want.astype(z.dtype), z.shape))
        gate(path+'_actual8_double_raw_readbacks_exact', ctxok)
    sid = sample['sample_geom_ids']; local = sample['sample_body_ids']; pts = sample['sample_points_body']
    gtable = {v['id']: v for v in mapping['actual_proxy_hand_geometry_records']}
    bodies = {v['local_FK_index']: v for v in mapping['FK_local_body_table']}
    unique = np.unique(sid); hand = {v['id'] for v in mapping['native_hand_geoms']}; obj = {v['id'] for v in mapping['native_object_geoms']}
    idok = (sid.dtype.kind in 'iu' and local.dtype.kind in 'iu' and sid.shape == local.shape == (1344,)
        and pts.shape == (1344, 3) and pts.dtype == np.float32 and np.isfinite(pts).all()
        and len(unique) == len(gtable) == 21 and set(unique) == set(gtable)
        and all((sid == g).sum() == 64 for g in unique)
        and all(re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)', v['name']) for v in mapping['native_hand_geoms'])
        and hand.isdisjoint(obj) and len(obj) == 3
        and mapping['sample_body_ids_semantics'] == 'local_FK_index, not nativebody/mocapID'
        and mapping['sample_geom_ids_semantics'] == 'nativeMJB geomID')
    for i in range(1344):
        idok &= int(local[i]) in bodies and bodies[int(local[i])]['native_body_name'] == gtable[int(sid[i])]['body']
    gate('sample_native_geom_vs_local_FK_body_ID_spaces_and64_each', idok)
    factory = read(run/'factory_receipt.json'); oldgeometry = read(checked(inputs['previous_bonus_factory_geometry_manifest']))['P_ONLY']['geometry']
    gate('factory_geometry_manifest_sampling_hash_same_frozen_bonus', factory['geometry'] == oldgeometry
         and factory['geometry']['sampling_points_sha256'] == hashlib.sha256(np.ascontiguousarray(pts).tobytes()).hexdigest()
         and mapping['actual_proxy_hand_geometry_records'] == factory['geometry']['hand_collision_geoms']
         and mapping['actual_proxy_object_geometry_records'] == factory['geometry']['object_geoms'])
    recomputed = math.proxy_from_saved_sdf(score['sampled_sdf_m'], sid)
    gate('all1808x1344_SDF_P_BP_min_pergeom_exact_F32_arithmetic', score['sampled_sdf_m'].shape == (1808, 1344)
         and all(equal(recomputed[k], score[v]) for k, v in [('P', 'P'), ('BP', 'BP'), ('min_sdf_m', 'minimum_sdf_m'), ('geom_ids', 'geom_ids'), ('pergeom_min_sdf_m', 'per_geom_min_sdf_m')]))
    chunkok = True
    for batch, start in enumerate(range(0, 1808, 128)):
        stop = min(start+128, 1808); chunk = loadnp(run/('chunk_%02d.npz' % batch))
        chunkok &= np.array_equal(chunk['row_indices'], np.arange(start, stop)) and equal(chunk['geom_ids'], score['geom_ids'])
        chunkok &= all(equal(chunk[k], score[k][start:stop]) for k in ('sampled_sdf_m', 'P', 'BP', 'minimum_sdf_m', 'per_geom_min_sdf_m'))
    gate('all15_saved_chunks_14x128_plus16_exact_aggregate', chunkok)
    rebuilt = []
    beforeok = finalok = categoriesok = True
    for i, s in enumerate(samples):
        minima = {int(g): float(recomputed['pergeom_min_sdf_m'][i, j]) for j, g in enumerate(unique)}
        r = contact_record(s, hand, obj, set(unique), minima)
        categoriesok &= r['native_generated_HO_depth_m'] == s['measured_hand_object']['penetration_by_category_m']['hand_object']
        before = {**r, 'primary_deepest_tie_coverage': None, 'secondary_all_violating_coverage': None, 'proxy_coverage_status': 'NOT_EVALUATED_BEFORE_FACTORY'}
        beforeok &= native_before[i] == before
        minimum, P, BP = score['minimum_sdf_m'][i], score['P'][i], score['BP'][i]
        observed = bool(P > 0)
        r.update(row_id=i, path=s['path'], mode=s['mode'], k=s['phase'], branch_id=s['branch_id'], substep=s['substep'],
            canonical_pre_tick=s['canonical_pre_tick'], source_frame=s['source_frame'], P=float(P), BP=float(BP), minimum_sampled_sdf_m=float(minimum),
            proxy_sample_depth_m=max(0., -float(minimum)), proxy_P_positive=observed,
            proxy_min_sdf_below_minus3mm=bool(minimum < -.003), proxy_min_sdf_exactly_minus_FP32_allowance=bool(minimum == np.float32(-.003)),
            native_depth_minus_nominal_3mm_m=r['native_generated_HO_depth_m']-.003,
            primary_miss=bool(r['native_HO_depth_gt3mm'] and not observed),
            proxy_alert_without_observed_native_HO3mm=bool(observed and not r['native_HO_depth_gt3mm']))
        finalok &= rows[i] == r
        rebuilt.append(r)
    gate('native_HO_exactmask_world_order_no_force_filter_category_agrees', categoriesok)
    gate('all1808_native_depth_before_factory_unknown_coverage_not_false', beforeok)
    gate('all1808_final_rows_exact_primaryties_secondaryallpairs_boundary_flags', finalok)
    gate('score_arrays_native_depth_phase_and_P_mask_allrows', np.array_equal(score['native_HO_depth_m'], [r['native_generated_HO_depth_m'] for r in rebuilt])
         and np.array_equal(score['phases'], [s['phase'] for s in samples]) and equal(score['proxy_P_positive'], recomputed['P_positive']))
    stats = {'all1808': summary(rebuilt)}
    for path in ('A0', 'A2'):
        for mode in ('donor', 'baseline', 'direction'):
            stats[path+'/'+mode] = summary([r for r in rebuilt if r['path'] == path and r['mode'] == mode])
    gate('allrows_and_six_path_mode_summaries_recomputed_no_selection', stats == read(run/'statistics.json') == report['statistics'])
    with (run/'rows_all1808.csv').open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f); csvok = True; count = 0
        for expected, actual in zip(rebuilt, reader, strict=True):
            count += 1
            for key, value in actual.items():
                want = expected[key]
                csvok &= value == ('' if want is None else str(want)) if isinstance(want, (str, bool)) or want is None else float(value) == want
    gate('all1808_CSV_rows_exact_original_fields', count == 1808 and csvok)
    wantedcounts = {'passive_mapping_MJB': 1, 'native_state_conversion': 1, 'objective_factory': 1, 'objective_forward': 15}
    gate('counts_complete_source_instrumented_not_new_calls', report['converted_rows'] == 2260 and report['scored_rows'] == 1808
         and report['counts'] == {k: {'attempted': n, 'returned': n} for k, n in wantedcounts.items()})
    gate('original_saved_readonly_invariants_source_assertions_bound', bool(report['invariants']) and all(v is True for v in report['invariants'].values()))
    gate('scientific_qualification_not_upgraded', report['schema'] == 'full151105_same_solver_phase_penalty_CPU_analysis:1'
         and report['status'] == 'COMPLETED_SAME_SOLVER_PHASE_PROXY_OBSERVABILITY_NOT_PHYSICS_CERTIFICATE'
         and report['physical_gradient_qualified'] is report['wholehiddenstate_qualified'] is False
         and report['WM_Actor_weight_loads'] == report['new_AD'] == report['Env_physics_nativeFK_collision_GPU'] == 0)
    for p in allpins:
        checked(p)
    for p in outputpins.values():
        checked(p)
    gate('all_source_input_output_pins_unchanged_after_saved_audit', pin(__file__) == source and digest(MATH) == MATH_SHA)
    gate('no_backend_model_or_geometry_imported', not bool({'torch', 'warp', 'mujoco', 'scipy'} & set(sys.modules)))
    result = {'schema': 'full151105_solver_phase_penalty_saved_independent:1',
        'status': 'PASS_SAVED_ONLY_REVIEW' if all(checks.values()) else 'FAILED_SAVED_ONLY_REVIEW',
        'engineering_qualified': all(checks.values()), 'passed': sum(checks.values()), 'total': len(checks), 'checks': checks,
        'checker': source, 'math': pin(MATH), 'config': pin(config_path), 'producer_report': pin(run/'report.json'),
        'protocol': cfg['protocol'], 'run_dir': str(run.resolve()), 'source_pin_count': len(cfg['source_pins']),
        'counts': report['counts'], 'independent_numeric': details, 'statistics': stats,
        'native_HO_depth_max_m': max(r['native_generated_HO_depth_m'] for r in rebuilt),
        'proxy_P_max': float(score['P'].max()), 'source_FP32_allowance_boundary_rows': sum(r['proxy_min_sdf_exactly_minus_FP32_allowance'] for r in rebuilt),
        'physical_gradient_qualified': False, 'whole_hidden_state_qualified': False,
        'limitations': ['Saved native58 relations reconstructed in NumPy, not bitexact feature replay or new native converter execution.',
            '452 actual Reader palm frames and fixed-root source topology are bound; not1808 direct mocap readbacks.',
            'Passive MJB named-index/topology/plugin/integrator validations and full native geom/type range proof remain source-bound/inherited1762; no MJB opened by this audit.',
            'Before-factory order/counters and weights/buffers/RNG invariants are source-instrumented assertions; no raw hidden-state/RNG replay.',
            'sample_body_ids are FK-local indices, sample_geom_ids native IDs; neither is mocapid.',
            'Exact deepest ties and all violating pairs are overlapping evidence, not exclusive cause assignments.',
            'Proxy alerts without generated HO3mm are disagreements, not proof of global geometric false positives.',
            'Generated-empty depth0 is observed candidate absence, not exhaustive positive clearance.',
            'No independent N, method benefit, physical/HO gradient, force/contact or wholehidden-state qualification.'],
        'new_native_converter_calls': 0, 'model_loads': 0, 'geometry_queries': 0, 'physics_steps': 0, 'GPU_used': False}
    dump(out/'report.json', result)
    print(json.dumps({'report': pin(out/'report.json'), 'passed': result['passed'], 'total': result['total'], 'statistics': stats}, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        raise SystemExit(audit(args.config, args.run_dir, args.output))
    except Exception as exc:
        if args.output.exists() and not (args.output/'failure.json').exists():
            dump(args.output/'failure.json', {'checker': pin(__file__), 'error': repr(exc), 'saved_only': True,
                'no_model_or_geometry_execution': True, 'new_version_required_for_harness_fix': True})
        raise
