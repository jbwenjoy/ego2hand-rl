"""One fixed frame0 contact-intent hypothesis -> geometric initialization.

No source robot-q labels, gap oracle, force solve, physical step or GPU. Fixed
material points and normals encode a prospective hypothesis; actual native
filtered contacts independently determine whether that hypothesis was attained.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, 'D:/workspace/supermimic-observation-retarget-study/scripts')
from native_filtered_contact_geometry import NativeFilteredContactGeometry
from retarget_palmroot_cpu import BONES
from material_anchors import MaterialAnchorGeometry


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


@contextmanager
def geometry_only():
    names = ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse', 'mj_geomDistance', 'mj_contactForce')
    saved = {name: getattr(mujoco, name) for name in names}
    def forbidden(*a, **kw): raise AssertionError('This is geometry only, no physics/force/gap query')
    try:
        for name in names: setattr(mujoco, name, forbidden)
        yield
    finally:
        for name, method in saved.items(): setattr(mujoco, name, method)


def json_anchor(anchor):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(anchor).items()}


def contact_records(g):
    records = []
    for c in g.data.contact:
        a, b = map(int, c.geom)
        if a not in g.object_ids and b not in g.object_ids: continue
        hand = b if a in g.object_ids else a
        if hand not in g.hand_ids: continue
        records.append({'geom': [a, b], 'names': [g.model.geom(a).name, g.model.geom(b).name],
                        'hand_geom': hand, 'distance_m': float(c.dist), 'position_w': c.pos.tolist(),
                        'frame_rows_geom0_to_geom1_w': c.frame.reshape(3, 3).tolist(),
                        'effective_condim': int(c.dim), 'friction5': c.friction.tolist(),
                        'phase': 'native_geometry_query_not_solved_force'})
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    recipe = read(args.recipe)
    assert recipe['schema'] == 'frame0_material_contact_hypothesis:2'
    for name, pin in recipe['pins'].items():
        if sha(pin['path']) != pin['sha256']: raise ValueError('Source/input drift: ' + name)
    assert recipe['weights'] == {'bone': 1., 'native_overlap': 1., 'anchor_position': 1.,
                                'anchor_normal': .25, 'q_baseline': .03, 'q_nominal': .001}
    assert recipe['anchor_position_scale_m'] == .001 and recipe['anchor_inset_m'] == .0003
    assert recipe['max_nfev'] == 120 and recipe['CPU_soft_wall_seconds'] == 60
    assert recipe['fingers'] in ([2, 3], [1, 2, 3])
    assert recipe['rankings_variant'] == 'source_rigid_mapped'
    assert recipe['material_geometry_rule'] == 'ordinal_landmark_to_link4_or_tip_only'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'recipe_snapshot.json', recipe)
    (out / 'optimizer_source.py').write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    def budget():
        if time.monotonic() - started > 60: raise TimeoutError('Fixed60s CPU recipe; preserve failure, no retry')
    pins = recipe['pins']
    with np.load(pins['baseline']['path'], allow_pickle=False) as z:
        qbase = z['joint_pos_rad'][0].copy()
        pose = z['mapped_object_pose_w'][0].copy()
        target_points = z['morphology_normalized_target_points_w'][0].copy()
    vector = target_points[BONES[:, 1]] - target_points[BONES[:, 0]]
    desired = vector / np.linalg.norm(vector, axis=1)[:, None]
    hypotheses = read(pins['rankings']['path'])['variants'][recipe['rankings_variant']]['nearest_per_finger_distal2']
    hypotheses = [next(h for h in hypotheses if h['finger_chain'] == f) for f in recipe['fingers']]
    assert all(h['confidence'] is None and h['is_known_contact_label'] is False and h['component'] == 'object/cube' and h['feature'] == 'cylinder_lateral_external' for h in hypotheses)
    identity = np.eye(20)
    with geometry_only():
        g = NativeFilteredContactGeometry(pins['model']['path'], pins['context']['path'])
        g.set_object_pose(pose)
        robot, material = g.robot, MaterialAnchorGeometry(g)
        with np.load(pins['context']['path'], allow_pickle=False) as z:
            # Contact material is needed only for the saved geometric wrench
            # screen. Never patch nominal host inertia prior to Warp upload.
            g.model.geom_friction[:] = z['geom_friction'][0]
        initial_native = g.evaluate(qbase)
        initial_contacts = contact_records(g)
        anchors, goals, normals = [], [], []
        cylinder = g.model.geom('object/cube').id
        for h in hypotheses:
            point, normal = np.array(h['object_surface_closest_w']), np.array(h['outward_normal_w'])
            # Independently verify the selected analytic surface target using
            # the actual compiled geom pose, not the observation reporter.
            local = g.data.geom_xmat[cylinder].reshape(3, 3).T @ (point - g.data.geom_xpos[cylinder])
            assert abs(np.linalg.norm(local[:2]) - g.model.geom_size[cylinder, 0]) < 1e-10
            assert abs(local[2]) < g.model.geom_size[cylinder, 1] - .01
            local_n = g.data.geom_xmat[cylinder].reshape(3, 3).T @ normal
            assert np.max(np.abs(local_n - np.r_[local[:2] / np.linalg.norm(local[:2]), 0.])) < 1e-8
            # Keep the existing ordinal landmark correspondence. Choosing a
            # proximal collider because it is near the failed baseline can
            # remove the distal joint DOFs required by the observed intent.
            within_chain = (h['landmark_index'] - 1) % 4
            suffix = {2: 'link4', 3: 'tip_sensor'}[within_chain]
            geom_name = f"robot/right_finger{h['finger_chain']}_{suffix}_col"
            geom_id = g.model.geom(geom_name).id
            assert geom_id in g.hand_ids
            restricted = SimpleNamespace(robot=g.robot, model=g.model, data=g.data,
                                         hand_ids=np.array([geom_id], dtype=int))
            anchor = MaterialAnchorGeometry(restricted).select(qbase, h['finger_chain'], point, normal)
            anchor.point_geom_local.setflags(write=False)
            anchor.support_normal_geom_local.setflags(write=False)
            anchors.append(anchor)
            goals.append(point - .0003 * normal)
            normals.append(-normal)
        goals, normals = np.array(goals), np.array(normals)
        bindings = {'hypothesis_id': recipe['hypothesis_id'], 'observation_hypotheses': hypotheses,
                    'frozen_material_anchors': [json_anchor(a) for a in anchors],
                    'target_material_points_w': goals.tolist(), 'target_hand_normals_w': normals.tolist(),
                    'selection_and_arrays_frozen_before_optimizer': True,
                    'known_contacts_or_forces': False, 'physical_reference_qualified': False}
        write(out / 'frozen_anchor_bindings.json', bindings)
        def residual(q):
            budget()
            bone, _ = robot.bone_features(q)
            collision, _ = g.penetration_residual(q)
            features = [material.oriented_features(q, anchor) for anchor in anchors]
            return np.r_[(bone - desired).ravel(), collision,
                         ((np.array([x[0] for x in features]) - goals) / .001).ravel(),
                         (.25 * (np.array([x[2] for x in features]) - normals)).ravel(),
                         .03 * (q - qbase), .001 * (q - robot.nominal)]
        def jacobian(q):
            budget()
            features = [material.oriented_features(q, anchor) for anchor in anchors]
            return np.vstack((robot.bone_features(q)[1], g.penetration_residual(q)[1],
                              np.vstack([x[1] / .001 for x in features]),
                              np.vstack([.25 * x[3] for x in features]), .03 * identity, .001 * identity))
        result = least_squares(residual, qbase.copy(), jac=jacobian, bounds=(robot.lower, robot.upper),
                               max_nfev=120, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        q = result.x.copy()
        native = g.evaluate(q)
        points = robot.forward(q)[0].copy()
        features = [material.oriented_features(q, anchor) for anchor in anchors]
        records = contact_records(g)
        bone = robot.bone_features(q)[0]
        angles = np.rad2deg(np.arccos(np.clip(np.sum(bone * desired, axis=1), -1, 1)))
        summary = {}
        for kind in ('hand_object', 'hand_self'):
            mask = native.generated_mask & (g.pair_kind == kind)
            summary[kind] = {'generated_pairs': int(mask.sum()),
                             'max_generated_depth_m': float(np.maximum(-native.signed_distance_m[mask], 0).max()) if mask.any() else None}
        attained = {str(f): any(f'right_finger{f}_' in g.model.geom(c['hand_geom']).name for c in records) for f in recipe['fingers']}
        report = {'status': 'COMPLETED_GEOMETRIC_HYPOTHESIS_ONLY', 'hypothesis_id': recipe['hypothesis_id'],
                  'optimizer_success': bool(result.success), 'optimizer_status': int(result.status),
                  'nfev': int(result.nfev), 'cost': float(result.cost),
                  'material_point_goal_error_m': np.linalg.norm(np.array([x[0] for x in features]) - goals, axis=1).tolist(),
                  'material_normal_goal_error_deg': np.rad2deg(np.arccos(np.clip(np.sum(np.array([x[2] for x in features]) * normals, axis=1), -1, 1))).tolist(),
                  'native_contacts_attained_by_hypothesized_finger': attained, 'native_summary': summary,
                  'bone_angle_mean_deg': float(angles.mean()), 'bone_angle_max_deg': float(angles.max()),
                  'target_tip_error_mean_m': float(np.linalg.norm(points[[4,8,12,16,20]] - target_points[[4,8,12,16,20]], axis=1).mean()),
                  'hard_joint_excess_max_rad': float(np.maximum(np.maximum(robot.lower - q, q - robot.upper), 0).max()),
                  'q_baseline_change_abs_max_rad': float(np.abs(q - qbase).max()),
                  'native_contact_tie_slots': int(np.count_nonzero(native.generated_mask & (native.tie_count > 1))),
                  'recipe_sha256': sha(args.recipe), 'script_sha256': sha(__file__),
                  'bindings_sha256': sha(out / 'frozen_anchor_bindings.json'),
                  'geometry_candidate_accepted_for_dynamics': False, 'PD_target': None,
                  'velocities_or_forces_inferred': False, 'physical_reference_qualified': False,
                  'wall_seconds_excluding_imports': time.monotonic() - started,
                  'physics_steps': 0, 'GPU_used': False,
                  'scope': 'Single frame0, explicit contact-intent hypothesis. Native counts are geometric contacts, not load-bearing/force/robot-equilibrium or trajectory qualification.'}
        np.savez_compressed(out / 'initialization_candidate.npz', joint_pos_rad=q[None], mapped_object_pose_w=pose[None],
                            fk_hand_points_w=points[None], morphology_normalized_target_points_w=target_points[None],
                            bone_angle_error_deg=angles[None], baseline_joint_pos_rad=qbase[None],
                            selected_native_contact_distance_m=native.signed_distance_m[None],
                            generated_contact_mask=native.generated_mask[None], geometry_pair_indices=g.pairs,
                            geometry_pair_kind=g.pair_kind, material_goal_points_w=goals,
                            material_goal_normals_w=normals, joint_lower_rad=robot.lower, joint_upper_rad=robot.upper)
        write(out / 'baseline_native_contacts.json', initial_contacts)
        write(out / 'native_contacts.json', records)
        report['output_npz_sha256'] = sha(out / 'initialization_candidate.npz')
        write(out / 'report.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
