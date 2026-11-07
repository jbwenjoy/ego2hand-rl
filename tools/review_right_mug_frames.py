"""Saved-array/MJB review independent of the FK and scene builders."""
from pathlib import Path
import argparse
import json
import sys
import tarfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin, ArtifactRegistry


def main(args):
    import mujoco
    stage = args.reference.resolve(); out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Fresh unified runs output')
    report = json.loads((stage/'report.json').read_text()); binding = json.loads((stage/'binding.json').read_text())
    for pin in report['sources']+list(report['outputs'].values())+[report['clip']]:
        verify_file_pin(pin)
    z = np.load(stage/'reference61.npz'); frames = z['frame_ids']; stamps = []
    raw = json.loads((stage/'raw_profile_and_poses.json').read_text())
    raw_records_match = True
    with tarfile.open(report['clip']['path']) as tar:
        for i, frame in enumerate(frames):
            info = json.load(tar.extractfile(f'{frame:06d}.info.json'))
            hand = json.load(tar.extractfile(f'{frame:06d}.hands.json'))['right']['umetrack_pose']
            ob = next(x for x in json.load(tar.extractfile(f'{frame:06d}.objects.json'))['9']
                      if str(x['object_uid']) == '223371871635142')
            raw_records_match &= (hand == raw['frames'][i]['umetrack_pose'])
            raw_records_match &= (ob['T_world_from_object'] == raw['frames'][i]['object']['T_world_from_object'])
            raw_records_match &= np.array_equal(hand['joint_angles'], z['joint_angles22_source_F64'][i])
            stamps.append(info['ref_timestamp_ns'])
    W = z['T_source_world_from_human_wrist_F64']; O = z['T_source_world_from_object_BODY_F64']
    H = z['T_human_wrist_from_rest_palm']; R = z['T_robot_world_from_robot_palm']
    relative = np.linalg.inv(W @ H) @ O
    expected = R[None] @ relative
    map_error = float(np.abs(expected-z['T_robot_world_from_object_BODY']).max())
    vertices = z['hand_vertices_object_BODY_F64']
    reconstructed = np.einsum('nij,nvj->nvi', O[:, :3, :3], vertices)+O[:, None, :3, 3]
    vertex_roundtrip_error = float(np.abs(reconstructed-z['hand_vertices_source_world_FP32']).max())
    model = mujoco.MjModel.from_binary_path(str(stage/'hand2_mug.mjb'))
    phone_path = ArtifactRegistry(ROOT/'configs/artifacts.toml').resolve('phone_model')
    phone = mujoco.MjModel.from_binary_path(str(phone_path))
    robot_same = True
    for name in ('actuator_gainprm', 'actuator_biasprm', 'actuator_gear', 'actuator_ctrlrange',
                 'actuator_ctrllimited', 'actuator_forcerange', 'actuator_forcelimited'):
        robot_same &= np.array_equal(getattr(model, name), getattr(phone, name))
    for i in range(20):
        old = phone.joint(model.joint(i).name).id
        for name in ('jnt_range', 'jnt_limited', 'jnt_pos', 'jnt_axis', 'jnt_actfrcrange', 'jnt_actfrclimited'):
            robot_same &= np.array_equal(getattr(model, name)[i], getattr(phone, name)[old])
        for name in ('dof_armature', 'dof_damping', 'dof_frictionloss'):
            robot_same &= np.array_equal(getattr(model, name)[model.jnt_dofadr[i]], getattr(phone, name)[phone.jnt_dofadr[old]])
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if name in binding['hand_collision_names']:
            old = phone.geom(name).id
            for field in ('geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_friction'):
                robot_same &= np.array_equal(getattr(model, field)[gid], getattr(phone, field)[old])
            if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                new_mid, old_mid = int(model.geom_dataid[gid]), int(phone.geom_dataid[old])
                for array_name, adr, num in [('mesh_vert', 'mesh_vertadr', 'mesh_vertnum'),
                                              ('mesh_face', 'mesh_faceadr', 'mesh_facenum')]:
                    na, nn = int(getattr(model, adr)[new_mid]), int(getattr(model, num)[new_mid])
                    oa, on = int(getattr(phone, adr)[old_mid]), int(getattr(phone, num)[old_mid])
                    robot_same &= np.array_equal(getattr(model, array_name)[na:na+nn], getattr(phone, array_name)[oa:oa+on])
    for bid in range(model.nbody):
        name = model.body(bid).name
        if name.startswith('right_'):
            old = phone.body(name).id
            for field in ('body_pos', 'body_quat', 'body_mass', 'body_ipos', 'body_iquat', 'body_inertia'):
                robot_same &= np.array_equal(getattr(model, field)[bid], getattr(phone, field)[old])
    oid = model.body('hot3d_mug9').id
    parts = set(binding['mug_part_geom_ids']); others = binding['hand_collision_names']+['mug_task_table']
    observed = set(); parameters_exact = True
    for i in range(model.npair):
        a, b = int(model.pair_geom1[i]), int(model.pair_geom2[i])
        if a not in parts and b not in parts:
            continue
        part, other = (a, b) if a in parts else (b, a)
        key = (part, model.geom(other).name)
        if key in observed:
            parameters_exact = False
        observed.add(key)
        mu = .6 if model.geom(other).name == 'mug_task_table' else .7
        parameters_exact &= (model.pair_dim[i] == 3 and model.pair_margin[i] == model.pair_gap[i] == 0
            and np.array_equal(model.pair_friction[i], [mu, mu, .005, .0001, .0001])
            and np.array_equal(model.pair_solref[i], [.002, 1])
            and np.array_equal(model.pair_solimp[i], [.9, .95, .001, .5, 2]))
    pair_coverage = observed == {(part, name) for part in parts for name in others}
    # No mj_forward/integration: saved free joint and BODY-local COM algebra.
    qadr = binding['mug_free_qpos']; qpos = z['initial_qpos']; q = qpos[qadr+3:qadr+7]
    root_error = float(np.abs(qpos[qadr:qadr+3]-expected[0, :3, 3]).max())
    com = z['object_COM_in_BODY_m']
    com_error = float(np.abs(np.einsum('nij,j->ni', expected[:, :3, :3], com)
        +expected[:, :3, 3]-z['object_COM_robot_world_m']).max())
    pid = model.body('right_palm_link').id
    checks = dict(all_source_pins_and_saved_outputs_verified=True,
        all61_raw_pose_records_and_full22_angles_equal=raw_records_match,
        timestamps_equal_actual_tar_and_not_stretched=np.array_equal(stamps, z['timestamps_ns']) and
            np.array_equal(z['elapsed_s'], (np.array(stamps)-stamps[0])*1e-9),
        fixed_palm_mapping_reconstructed=map_error < 1e-12,
        source_world_object_frame_geometry_roundtrip=vertex_roundtrip_error < 1e-12,
        robot_gains_limits_collision_shape_match_frozen_phone=robot_same,
        exactly27q26v20u_no_aux=(model.nq, model.nv, model.nu, model.neq, model.nmocap) == (27, 26, 20, 0, 0),
        no_wrist_freejoint_or_body_gravity_compensation=model.body_jntnum[pid] == 0 and np.all(model.body_gravcomp == 0),
        all128_components_same_rigid_BODY=len(parts) == 128 and all(model.geom_bodyid[p] == oid for p in parts),
        explicit2816_pair_coverage=pair_coverage and len(observed) == 2816,
        all_pair_parameters_exact=parameters_exact,
        visual_excluded=model.geom_contype[model.geom('mug9_visual').id] == model.geom_conaffinity[model.geom('mug9_visual').id] == 0,
        BODY_COM_distinguished=root_error < 1e-12 and com_error < 1e-12 and np.array_equal(com, model.body_ipos[oid]),
        free_quaternion_unit=abs(np.linalg.norm(q)-1) < 1e-12,
        source_gravity_and_grasp_claim_remain_unknown=binding['gravity']['source_world'] is None and not binding['physical_grasp_qualified'])
    checks = {k:bool(v) for k, v in checks.items()}
    out.mkdir(parents=True)
    result = dict(schema='right_mug_saved_reference_review:1', checks=checks, passed=sum(checks.values()), total=len(checks),
        reference_report=file_pin(stage/'report.json'), source=file_pin(__file__), frozen_robot_comparison=file_pin(phone_path),
        numerical=dict(map_error=map_error, vertex_roundtrip_m=vertex_roundtrip_error, COM_error_m=com_error, root_error_m=root_error),
        native_model_loads=2, native_forward_calls=0, integration_steps=0, neural_updates=0,
        scope='Independent saved JSON/array/MJB checks, not independent human reconstruction, source-gravity calibration, continuous collision, robot IK, holding, policy benefit or ego generalization.')
    (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(dict(report=file_pin(out/'report.json'), checks=checks, passed=result['passed'], total=result['total'])))
    if not all(checks.values()):
        raise RuntimeError('Saved review failed; retain report')


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
