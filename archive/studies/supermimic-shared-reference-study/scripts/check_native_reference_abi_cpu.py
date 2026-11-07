"""Real native-reader ABI check for an observation-derived reference only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from native_reference_abi import export_clip, sha


def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    started = time.monotonic()
    out = ROOT / 'outputs/native_ABI_cpu_v1'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'checker_source.py').write_bytes(Path(__file__).read_bytes())
    source = Path('D:/workspace/supermimic-observation-retarget-study/outputs/native_contact_refined_geometry_v1/geometric_retarget.npz')
    observations = Path('D:/workspace/supermimic-chord-study/outputs/observation_only_fixture_v1/observations.npz')
    model = Path('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb')
    constants = Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/tasks/pen_spin/config/wuji_hand2/hand2_constants.py')
    forbidden_calls = {}
    def forbid(name):
        def call(*a, **kw):
            forbidden_calls[name] = forbidden_calls.get(name, 0) + 1
            raise AssertionError('No physics/force/gap API: '+name)
        return call
    for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_contactForce','mj_geomDistance'):
        setattr(mujoco, name, forbid(name))
    with np.load(source, allow_pickle=False) as z:
        q, pose, t, macro = (z[k].copy() for k in ('joint_pos_rad','mapped_object_pose_w','time_s','fk_hand_points_w'))
    group = out / 'clips/observation_derived_geometry'
    path, sidecar, info = export_clip(group, 'fixture007', t, q, pose, model_path=model, constants_path=constants,
        provenance={'is_real_ego':False,'source_robot_labels_used':False,
                    'observation_source':{'path':str(observations),'sha256':sha(observations)},
                    'geometric_derivative':{'path':str(source),'sha256':sha(source)},
                    'lineage':'upstream_01_single_axis_original_parent_unknown_ALL10',
                    'confidence':None,'physical_reference_qualified':False})
    write(group / 'index.json', {'schema':'wuji_pen_spin_clips:1','group_id':group.name,
        'group_name':'Observation-derived geometry ABI check, not physical reference',
        'frame_dt':.020,'clip_count':1,'clips':[{'clip_id':'fixture007','file':path.name,'sha256':sha(path)}]})
    recipe = out / 'recipe.json'
    write(recipe, {'schema':'wuji_pen_spin_recipe:1','clips':[group.name+'/fixture007']})
    # Import and execute the REAL reader, rather than copying its methods.
    import torch
    from dex_supermimic.tasks.pen_spin.tooling.recipe import resolve_recipe
    from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingReader
    selection = resolve_recipe(recipe, clips_root=out / 'clips', verify_clip_hashes=True)
    loaded = SingleHandTrackingReader(selection.clip_sets[0], device='cpu').read()
    with np.load(path, allow_pickle=False) as z: derived = {k:z[k].copy() for k in z.files}
    wrist = derived['wrist_pose_w']
    Rw = Rotation.from_quat(wrist[:,[4,5,6,3]]).as_matrix()
    Ro = Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix()
    expected_p = np.einsum('tji,tj->ti', Rw, pose[:,:3]-wrist[:,:3])
    expected_R = np.einsum('tji,tjk->tik', Rw, Ro)
    native_obj = loaded['object_pose'].numpy()
    native_R = Rotation.from_quat(native_obj[:,[4,5,6,3]]).as_matrix()
    expected_links = np.einsum('tji,tkj->tki', Rw, derived['link_pos_w']-wrist[:,None,:3])
    q32 = q.astype(np.float32)
    v = np.zeros_like(q32);v[:-1]=(q32[1:]-q32[:-1])/np.float32(.020);v[-1]=v[-2]
    errors = {'joint_pos_rad':float(np.max(np.abs(loaded['joint_pos'].numpy()-q32))),
              'joint_goal_finite_difference_velocity_rad_s':float(np.max(np.abs(loaded['joint_vel'].numpy()-v))),
              'object_local_position_m':float(np.max(np.abs(native_obj[:,:3]-expected_p))),
              'object_local_rotation_matrix':float(np.max(np.abs(native_R-expected_R))),
              'native_BODY_links_local_m':float(np.max(np.abs(loaded['link_pos'].numpy()-expected_links)))}
    macro_difference = float(np.linalg.norm(derived['link_pos_w']-macro,axis=-1).max())
    checks = {'real_native_resolve_and_reader_executed':True,
              'trajectory_length151':loaded['trajectory_lengths'].tolist()==[151],
              'derived_q_exact_FP32':errors['joint_pos_rad']==0.,
              'native_goal_velocity_rule':errors['joint_goal_finite_difference_velocity_rad_s']<1e-5,
              'object_transform_parity':errors['object_local_position_m']<1e-6 and errors['object_local_rotation_matrix']<1e-6,
              'BODY_link_transform_parity':errors['native_BODY_links_local_m']<1e-6,
              'macro_FK_not_silently_renamed':macro_difference>1e-4,
              'no_forbidden_calls':not forbidden_calls,
              'CUDA_not_initialized':not torch.cuda.is_initialized(),
              'all_native_tensors_CPU':all(not x.is_cuda for x in loaded.values() if torch.is_tensor(x))}
    assert all(checks.values()), checks
    report={'status':'PASS_NATIVE_READER_ABI_CPU_ONLY','checks':checks,'errors':errors,
            'native_vs_macro_landmark_max_distance_m':macro_difference,
            'native_reference':{'path':str(path),'sha256':sha(path),'sidecar_sha256':sha(sidecar)},
            'export_contract':info,'checker_sha256':sha(__file__),
            'native_reader_source_sha256':sha(Path(sys.modules[SingleHandTrackingReader.__module__].__file__)),
            'forbidden_calls':forbidden_calls,'GPU_used':False,'physics_steps':0,
            'wall_seconds':time.monotonic()-started,
            'physical_reference_qualified':False,'training_authorized':False,
            'scope':'One existing observation-derived geometric fixture, native ABI/coordinate parity only. Goal finite-difference qvel is not measured robot velocity. No new IK, multi-trajectory training or physical success.'}
    write(out / 'report.json', report)
    print(json.dumps({k:report[k] for k in ('status','checks','errors','native_vs_macro_landmark_max_distance_m','wall_seconds')},indent=2))


if __name__=='__main__':main()
