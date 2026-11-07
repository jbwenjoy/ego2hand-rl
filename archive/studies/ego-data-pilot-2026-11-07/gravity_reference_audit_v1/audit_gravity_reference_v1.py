"""Saved-pose/source audit only. No model loading, FK, geometry query or integration."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
P = ROOT.parent
CONSTANTS = Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/tasks/pen_spin/config/wuji_hand2/hand2_constants.py')


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def pose(record):
    q = np.asarray(record['quaternion_wxyz'], np.float64)
    p = np.asarray(record['translation_xyz'], np.float64)
    if q.shape != (4,) or p.shape != (3,) or not np.isfinite(np.r_[q, p]).all():
        raise ValueError('Invalid saved pose')
    # scipy normalizes the already-unit raw quaternion; the raw norm is retained.
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(), p, float(np.linalg.norm(q))


def angle_deg(a, b):
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def rotation_angle_deg(R):
    return float(np.degrees(Rotation.from_matrix(R).magnitude()))


def gravity_paths(value, prefix=''):
    out = []
    if isinstance(value, dict):
        for k, v in value.items():
            key = f'{prefix}.{k}' if prefix else k
            if 'grav' in k.lower():
                out.append(key)
            out.extend(gravity_paths(v, key))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out.extend(gravity_paths(v, f'{prefix}[{i}]'))
    return out


def line_descriptor(points, com, g, mass):
    origin = points[0]
    axis = points[1] - origin
    length = float(np.linalg.norm(axis))
    if not length > 0:
        raise ValueError('Degenerate saved contact line')
    axis /= length
    v = com - origin
    vp = v - axis * (axis @ v)
    gp = g - axis * (axis @ g)
    return {'contact_separation_m': length, 'axis_world': axis.tolist(),
            'COM_perpendicular_line_m': vp.tolist(),
            'gravity_perpendicular_line_m_s2': gp.tolist(),
            'gravity_perpendicular_magnitude_m_s2': float(np.linalg.norm(gp)),
            'projected_COM_gravity_angle_deg': angle_deg(vp, gp),
            'gravity_torque_about_line_Nm_signed': float(axis @ np.cross(v, mass * g))}


def main():
    if (ROOT / 'report.json').exists():
        raise FileExistsError('Do not overwrite an executed saved audit')
    clip_path = P / 'hot3d/train_aria/clip-001856.tar'
    with tarfile.open(clip_path) as tar:
        members = {kind: json.load(tar.extractfile(f'000042.{kind}.json'))
                   for kind in ('hands', 'objects', 'cameras', 'info')}
    hands = members['hands']
    objects = members['objects']['24']
    if len(objects) != 1 or objects[0]['object_name'] != 'cellphone':
        raise ValueError('Ambiguous phone instance')
    human_pose = hands['left']['umetrack_pose']['T_world_from_wrist']
    object_pose = objects[0]['T_world_from_object']
    Rh, ph, qh_norm = pose(human_pose)
    Ro, po, qo_norm = pose(object_pose)
    placement = read(P / 'hand2_stage1/completed_v2/palm_keypoints_and_placement.json')
    Tp = np.asarray(placement['T_world_from_palm'], np.float64)
    Rp, pp = Tp[:3, :3], Tp[:3, 3]
    constant = None
    for node in ast.parse(CONSTANTS.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RIGHT_FIXED_WRIST_POSE_W' for t in node.targets):
            constant = np.asarray(ast.literal_eval(node.value), np.float64)
    if constant is None or not np.array_equal(constant, np.asarray(placement['fixed_source_wrist_pos_quat_wxyz'])):
        raise ValueError('Robot placement does not match the original fixed-wrist constant')
    with np.load(P / 'retarget_geometry_v2/seed2_selected_state.npz', allow_pickle=False) as saved:
        Rr = saved['phone_R_world'].copy()
        pr = saved['phone_p_world'].copy()
    probe = read(P / 'gravity_alignment_probe_v1/report.json')
    g_robot = np.asarray(probe['gravity_world_m_s2'], np.float64)
    if not np.array_equal(g_robot, np.array([0., 0., -9.81])):
        raise ValueError('Unexpected robot gravity')
    mass = float(probe['mass_kg_assumed'])
    points = np.asarray(probe['material_contact_points_world_m'], np.float64)
    original = next(c for c in probe['cases'] if c['name'] == 'original')
    aligned = next(c for c in probe['cases'] if c['name'] == 'gravity_line_aligned')
    com = np.asarray(original['COM_world_m'], np.float64)
    g_source_assumed = g_robot.copy()
    # These two maps are descriptors, not an applied scene/world registration.
    A_wrist = Rp @ Rh.T
    A_phone = Rr @ Ro.T
    ego_t = Rh.T @ (po - ph)
    robot_t = Rp.T @ (pr - pp)
    ego_R = Rh.T @ Ro
    robot_R = Rp.T @ Rr
    local_points = (points - pr) @ Rr
    local_com = (com - pr) @ Rr
    prep = read(P / 'retarget_geometry_v2/fixed_anchors_and_targets.json')
    proxy_points = np.asarray([row['target_projected_phone_convex_object_m'] for row in prep['preparation']], np.float64)
    derived = {
        'robot_gravity_in_palm_m_s2': (Rp.T @ g_robot).tolist(),
        'CONDITIONAL_source_gravity_in_left_wrist_m_s2': (Rh.T @ g_source_assumed).tolist(),
        'CONDITIONAL_source_gravity_in_phone_m_s2': (Ro.T @ g_source_assumed).tolist(),
        'robot_gravity_in_fitted_phone_m_s2': (Rr.T @ g_robot).tolist(),
        'CONDITIONAL_wrist_axis_map': {
            'R_robot_world_from_source_world': A_wrist.tolist(), 'determinant': float(np.linalg.det(A_wrist)),
            'mapped_source_gravity_m_s2': (A_wrist @ g_source_assumed).tolist(),
            'angle_to_robot_gravity_deg': angle_deg(A_wrist @ g_source_assumed, g_robot),
            'scope': 'Hypothetical equality of human wrist and robot palm coordinate axes; no anatomical axis calibration, not the v2 retarget mapping.'},
        'CONDITIONAL_object_material_map': {
            'R_robot_world_from_source_world': A_phone.tolist(), 'determinant': float(np.linalg.det(A_phone)),
            'mapped_source_gravity_m_s2': (A_phone @ g_source_assumed).tolist(),
            'angle_to_robot_gravity_deg': angle_deg(A_phone @ g_source_assumed, g_robot),
            'mapped_human_wrist_vs_fixed_robot_palm_angle_deg': rotation_angle_deg(Rp.T @ A_phone @ Rh),
            'scope': 'Proper rigid map aligning the two phone poses; preserves material coordinates, not gravity or anatomical hand shape.'},
        'relative_pose_unregistered_wrist_axes': {
            'T_left_wrist_from_phone': np.block([[ego_R, ego_t[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]).tolist(),
            'T_robot_palm_from_fitted_phone': np.block([[robot_R, robot_t[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]).tolist(),
            'translation_difference_norm_m': float(np.linalg.norm(robot_t - ego_t)),
            'rotation_difference_deg': rotation_angle_deg(robot_R @ ego_R.T),
            'scope': 'Different human-left/robot-right coordinate bases and shapes; descriptive, not a calibrated retarget error.'},
        'actual_robot_contact_line': line_descriptor(points, com, g_robot, mass),
        'CONDITIONAL_same_robot_material_points_in_ego_phone_pose': line_descriptor(local_points @ Ro.T + po, local_com @ Ro.T + po, g_source_assumed, mass),
        'CONDITIONAL_projected_human_proxy_points_in_ego_phone_pose': line_descriptor(proxy_points @ Ro.T + po, local_com @ Ro.T + po, g_source_assumed, mass),
        'projected_human_proxy_points_in_robot_phone_pose': line_descriptor(proxy_points @ Rr.T + pr, com, g_robot, mass),
        'robot_closest_material_to_projected_human_proxy_point_distances_m': np.linalg.norm(local_points - proxy_points, axis=1).tolist(),
        'saved_analytic_rotation_about_fixed_contact_line_deg': aligned['rotation_about_world_line_deg'],
        'saved_line_alignment_geometry': {k: aligned[k] for k in ('pad_gap_window_pass', 'all_nonpad_nonpenetration_pass', 'deepest_penetration_m', 'thumb_index_anchor_error_m', 'robot_q20_and_all_hand_geom_poses_unchanged')},
        'roundoff_crosschecks_not_new_scientific_gates': {
            'saved_vs_recomputed_original_line_torque_abs_difference_Nm': abs(float(original['gravity_torque_about_line_Nm_signed']) - line_descriptor(points, com, g_robot, mass)['gravity_torque_about_line_Nm_signed']),
            'projected_angle_vs_saved_shortest_rotation_abs_difference_deg': abs(derived_angle := line_descriptor(points, com, g_robot, mass)['projected_COM_gravity_angle_deg'] - abs(aligned['rotation_about_world_line_deg']))}
    }
    inputs = {
        'raw_clip': clip_path, 'fixed_wrist_constants': CONSTANTS,
        'palm_placement': P / 'hand2_stage1/completed_v2/palm_keypoints_and_placement.json',
        'reference_targets': P / 'retarget_geometry_v1/frame42_targets.json',
        'reference_target_source': P / 'retarget_geometry_v1/prepare_frame42_targets_v1.py',
        'retarget_protocol': P / 'retarget_geometry_v2/protocol_v2.json',
        'retarget_source': P / 'retarget_geometry_v2/solve_contact_first_v2.py',
        'retarget_selected_pose': P / 'retarget_geometry_v2/seed2_selected_state.npz',
        'retarget_anchor_targets': P / 'retarget_geometry_v2/fixed_anchors_and_targets.json',
        'saved_gravity_probe': P / 'gravity_alignment_probe_v1/report.json',
        'saved_gravity_probe_source': P / 'gravity_alignment_probe_v1/run_gravity_alignment_probe_v1.py',
        'two_point_support_obstruction': P / 'hand2_holding_v1/support_preparation_v1/support_explanation_v1.json',
        'official_hand_parser': P / 'sources/independent_parser_source_v1/hand_tracking_toolkit__hand_tracking_toolkit__dataset.py',
        'official_object_visualizer': P / 'sources/independent_parser_source_v1/hot3d__hot3d__clips__vis_clips.py',
        'source_plan_v0_22': Path('D:/workspace/supermimic-chord-study/docs/research-plan-v0.22.md')}
    write(ROOT / 'frame42_raw_pose_metadata.json', {
        'clip': '001856', 'frame': 42, 'info': members['info'],
        'left_wrist': human_pose, 'phone24': object_pose,
        'phone_visibility_modeled': objects[0]['visibilities_modeled'],
        'left_visibility_modeled': hands['left']['visibilities_modeled'],
        'gravity_named_fields_in_four_raw_JSON_members': {k: gravity_paths(v) for k, v in members.items()}})
    report = {
        'schema': 'saved_gravity_reference_coordinate_audit:1',
        'status': 'COMPLETED_SAVED_GEOMETRY_CONDITIONAL_SOURCE_GRAVITY_NOT_PHYSICAL_FEASIBILITY_CERTIFICATE',
        'source': pin(__file__), 'inputs': {k: pin(v) for k, v in inputs.items()},
        'scope': {'new_NPZ_reads': 1, 'raw_clip_JSON_members': 4, 'model_loads': 0, 'FK': 0, 'geometry_queries': 0, 'integration': 0, 'training': 0, 'GPU': 0, 'wrist_or_gravity_changes': 0},
        'pose_arithmetic': 'FP64 saved-pose rigid transforms; scipy normalizes raw wxyz quaternions. No model/framework import.',
        'quaternion_norms': {'raw_human_left': qh_norm, 'raw_phone24': qo_norm},
        'raw_pose_rotation_determinants': {'human_left': float(np.linalg.det(Rh)), 'ego_phone24': float(np.linalg.det(Ro)), 'robot_palm': float(np.linalg.det(Rp)), 'fitted_phone': float(np.linalg.det(Rr))},
        'robot_fixed_wrist_pos_quat_wxyz': constant.tolist(),
        'source_gravity_status': 'NOT_EXPORTED_IN_THE_FOUR_RAW_FRAME_JSONS; GT_WORLD_TO_MPS_GRAVITY_FRAME_REGISTRATION_NOT_VERIFIED',
        'source_gravity_used_for_CONDITIONAL_descriptors_m_s2': g_source_assumed.tolist(),
        'robot_gravity_world_m_s2': g_robot.tolist(),
        'derived': derived,
        'source_contract': {
            'transform_direction': 'Official parser/visualizer apply T_world_from_wrist/object as local-to-world, not the reverse.',
            'left_to_right': 'Same anatomical digit, no reflection/permutation or 22-angle transfer. Shape/chirality mismatch remains; a proper SE3 is not a left-to-right shape equivalence.',
            'v2_reference': 'Object-centric human near-surface/tip material points. Phone SE3 and robot q20 optimized relative to the constant palm. No preservation term for source wrist-to-phone SE3 or source gravity.',
            'current_v3_pose': 'No scene or solver is changed by this audit. Current v3 must retain the original RIGHT_FIXED_WRIST_POSE_W.'},
        'interpretation': [
            'The 98.34deg result is the COM lever/gravity angle after projection perpendicular to a particular fixed two-contact line. It is not the 99.86deg conditional object-gravity discrepancy or 86.39deg hypothetical wrist-axis discrepancy.',
            'At the saved robot pose and mass/COM assumption, two forces located at the two fixed contact points cannot generate torque along their connecting line. The nonzero gravity axial torque is a necessary-condition obstruction for that fixed two-point model, not all grasp geometries.',
            'The conditional object-material comparison supports a coordinate/gravity relation mismatch hypothesis, but raw clip gravity registration is unresolved. It does not prove a transform bug, prove human static equilibrium, or assign the full 98deg to the fixed wrist.',
            'Rotating only the phone by the saved 98.34deg aligns the line condition but introduces about8.49mm collision and changes reference alignment. This saved failure is retained; neither zero line torque nor low geometric residual certifies support.',
            'Source projected points and robot closest contacts are distinct proxies. Human contact patch/force, mass/COM and dynamics are unmeasured; a source two-point counterfactual is not a human-force reconstruction.'],
        'future_constant_installation_variant': {
            'status': 'SOURCE_ONLY_SUGGESTION_NOT_AUTHORIZATION_OR_EXECUTED_VARIANT',
            'distinction': 'Fixed wrist during rollout is a temporal constraint; RIGHT_FIXED_WRIST_POSE_W fixes one specific installation frame in the existing experiment.',
            'possible_variant': 'After independently verifying source world gravity and a proper human-wrist/robot-palm basis calibration, register a new constant wrist installation before rollout and keep it constant throughout. No moving wrist, reflection or gravity editing.',
            'mathematical_contract': 'Choose A in SO(3) with A*g_source=g_robot (matched magnitude/convention), and declare C=R_humanWrist_from_robotPalm. At a fixed predeclared reference time, R_install=A*R_sourceWrist*C. Gravity alone leaves yaw undetermined; define it from a physical registration/reference axis before fitting.',
            'required': ['Actual clip GT gravity/IMU or a documented registered trajectory, not MPS documentation alone', 'Declared proper basis/installation calibration and reference time; no mirror', 'Fresh scene/protocol/pins and all collision/contact/actuator/support/dynamics checks', 'Shared installation for all matched method/control arms, with calibration cost disclosed'],
            'limitations': 'This changes the experimental scene and reachable contact/gravity lever arms. It cannot be called the original-pose result or a same-scene/same-budget method gain. A single constant mount cannot reproduce a time-varying source wrist; human-left/robot-right morphology still differs. Current original-pose joint solving continues unchanged.'},
        'primary_documentation': [
            {'url': 'https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_trajectory', 'claim': 'MPS closed-loop world is gravity aligned with [0,0,-9.81]; the documentation explicitly allows other ground-truth sources to use different values. This alone does not bind this clip GT world.'},
            {'url': 'https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md', 'claim': 'Clip archive and per-frame pose metadata format. Direction here follows the pinned official parser/visualizer rather than inconsistent prose in the README.'}],
        'outputs': {'raw_metadata': pin(ROOT / 'frame42_raw_pose_metadata.json')}
    }
    write(ROOT / 'report.json', report)
    d = report['derived']
    text = f'''Saved gravity/reference audit, clip001856 frame42\n\nThe original fixed robot wrist remains unchanged. This audit reads saved poses only: 0 model/FK/query/integration/GPU. Proper local-to-world transforms and same-anatomical-digit/no-reflection mapping are consistent in the source. V2 fits object-centric contact/tip references; it does not preserve human wrist-to-phone SE3 or source gravity.\n\nConditional on source world gravity [0,0,-9.81] (not independently verified for this clip GT), the object-frame gravity discrepancy is {d['CONDITIONAL_object_material_map']['angle_to_robot_gravity_deg']:.6f}deg. A hypothetical direct wrist-axis map gives {d['CONDITIONAL_wrist_axis_map']['angle_to_robot_gravity_deg']:.6f}deg, but human wrist and robot palm axes have no anatomical calibration. The raw clip's four frame JSONs have no gravity field. Official MPS documentation describes Z-down gravity but explicitly permits other GT conventions; it cannot certify this clip's actual gravity registration.\n\nThe saved {d['saved_analytic_rotation_about_fixed_contact_line_deg']:.6f}deg is instead the projected COM/gravity angle around a fixed contact line. At this robot pose its axial gravity moment is {d['actual_robot_contact_line']['gravity_torque_about_line_Nm_signed']:.9f}Nm. Two forces applied on that line cannot cancel this moment. Phone-only alignment fails saved collision constraints (deepest {1000*d['saved_line_alignment_geometry']['deepest_penetration_m']:.6f}mm); no physical grasp success follows. The same robot material points moved to the ego phone pose would have only {d['CONDITIONAL_same_robot_material_points_in_ego_phone_pose']['gravity_torque_about_line_Nm_signed']:.9f}Nm axial moment, conditionally; those are not measured human contact forces or proof of source equilibrium.\n\nA future one-time gravity-calibrated constant wrist installation could satisfy a rollout-fixed-wrist requirement, while changing the scene from RIGHT_FIXED_WRIST_POSE_W. It needs verified source gravity plus a proper wrist/palm basis and a predeclared yaw/reference time; then fresh scene qualification and matched public controls. It is neither moving-wrist control nor evidence of an original-pose method gain. No such change is performed or authorized here; original-pose v3 solving continues.\n\nPrimary references: https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_trajectory ; https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md\nExact source/input hashes, all vectors, transforms and limitations: report.json.\n'''
    (ROOT / 'description.txt').write_text(text, encoding='utf-8')
    print(json.dumps({'report': pin(ROOT / 'report.json'), 'description': pin(ROOT / 'description.txt'),
                      'angles_deg': {'line': d['saved_analytic_rotation_about_fixed_contact_line_deg'], 'conditional_phone': d['CONDITIONAL_object_material_map']['angle_to_robot_gravity_deg'], 'conditional_wrist': d['CONDITIONAL_wrist_axis_map']['angle_to_robot_gravity_deg']}, 'scope': report['scope']}))


if __name__ == '__main__':
    main()
