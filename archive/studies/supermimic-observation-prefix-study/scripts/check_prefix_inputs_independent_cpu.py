"""Independent read-only prefix-input checks; no solve, collision query or physics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=Path('D:/workspace/supermimic-observation-dynamic-wrench-study/proposals/short_dynamic_tracking_v1/protocol.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 p=read(PROTOCOL);assert sha(PROTOCOL)=='94ea7cbe03f3e2aaa7bb4f7c8860bae8e599cd4e8e823cbe47220f73ffe101a3'
 pins=p['inputs'];checks=[]
 def check(name,ok,detail=None):
  checks.append({'name':name,'pass':bool(ok),'detail':detail})
 for name,pin in pins.items():check('input SHA '+name,sha(pin['path'])==pin['sha256'])
 kinpath=pins['kinematics']['path'];spec=importlib.util.spec_from_file_location('prefix_independent_kin',kinpath);kin=importlib.util.module_from_spec(spec);spec.loader.exec_module(kin)
 # Do not request cached wrist, confidence arrays or original robot-q from fixture/old151.
 with np.load(pins['observation_fixture']['path'],allow_pickle=False) as z:
  t=z['time_s'].copy();hand=z['hand_landmarks_source_w'][:16].astype(np.float64);obj_source=z['object_pose_source_w'][:16].astype(np.float64)
 with np.load(pins['frozen_full151_geometry']['path'],allow_pickle=False) as z:
  mapped=z['mapped_object_pose_w'][:16].copy();rot=z['source_to_fixed_wrist_R'][:16].copy();trans=z['source_to_fixed_wrist_translation_m'][:16].copy()
 with np.load(pins['canonical_H123_q0']['path'],allow_pickle=False) as z:
  q0=z['joint_pos_rad'][0].copy();lower=z['joint_lower_rad'].copy();upper=z['joint_upper_rad'].copy();parentpose=z['mapped_object_pose_w'][0].copy()
 with np.load(pins['rawFD_design_arrays']['path'],allow_pickle=False) as z:
  fd={k:z[k][:16].copy() for k in ('object_pose_mapped_w','raw_v_world_m_s','raw_a_world_m_s2','raw_omega_world_rad_s','raw_alpha_world_rad_s2')}
 desired=np.einsum('tij,tkj->tki',rot,kin.directions(hand))
 check('time16 canonical20ms',np.allclose(t[:16],np.arange(16)*.02,atol=1e-15,rtol=0))
 check('bone direction15x3 dimensionless unit',desired.shape==(16,15,3) and np.isfinite(desired).all() and np.max(abs(np.linalg.norm(desired,axis=-1)-1))<1e-12)
 mappedp=np.einsum('tij,tj->ti',rot,obj_source[:,:3])+trans
 srcR=Rotation.from_quat(obj_source[:,[4,5,6,3]]).as_matrix();targetR=Rotation.from_quat(mapped[:,[4,5,6,3]]).as_matrix()
 check('R(t),t(t) fixed mapped object reconstruction',np.max(abs(mappedp-mapped[:,:3]))<1e-12 and np.max(abs(rot@srcR-targetR))<1e-12)
 check('q0 source64 hardbound without adjustment',q0.shape==(20,) and np.isfinite(q0).all() and np.all(q0>=lower) and np.all(q0<=upper),
  {'min_margin_rad':float(np.minimum(q0-lower,upper-q0).min()),'bound_indices':np.flatnonzero(np.minimum(q0-lower,upper-q0)<=1e-10).tolist(),
   'FP32_cast_hard_excess_rad':float(np.maximum(np.maximum(lower-q0.astype(np.float32),q0.astype(np.float32)-upper),0).max())})
 check('canonical pose0 matches fixed object reference',np.max(abs(parentpose-mapped[0]))<1e-12)
 check('rawFD reference knots preserved',np.array_equal(mapped,fd['object_pose_mapped_w']))
 model=mujoco.MjModel.from_binary_path(pins['model']['path'])
 names=['right_palm_link']+[f'right_finger{f}_link{j}' for f in range(1,6) for j in range(1,5)]
 bodyids=[model.body('robot/'+name).id for name in names]
 check('native21 BODY distinct from macroFK',len(set(bodyids))==21)
 check('source hardbounds match MJB',np.array_equal(lower,np.array([model.joint(f'robot/right_finger{f}_joint{j}').range[0] for f in range(1,6) for j in range(1,5)])) and
  np.array_equal(upper,np.array([model.joint(f'robot/right_finger{f}_joint{j}').range[1] for f in range(1,6) for j in range(1,5)])))
 check('COM offset zero only for current object',np.array_equal(model.body('object/cube').ipos,np.zeros(3)))
 binding=read(pins['frozen_material_anchors']['path']);g0=np.array(binding['target_material_points_w']);hn0=np.array(binding['target_hand_normals_w'])
 R0=targetR[0];outward0=-hn0;surface0=g0+.0003*outward0
 surface_local=(surface0-mapped[0,:3])@R0;outward_local=outward0@R0
 goals=np.einsum('tij,kj->tki',targetR,surface_local-.0003*outward_local)+mapped[:,None,:3]
 normals=-np.einsum('tij,kj->tki',targetR,outward_local)
 check('material object BODY-local transport roundtrip frame0',np.max(abs(goals[0]-g0))<1e-15 and np.max(abs(normals[0]-hn0))<1e-15)
 check('transported normals remain unit',np.max(abs(np.linalg.norm(normals,axis=-1)-1))<1e-12)
 check('parent material identities frozen',len(binding['frozen_material_anchors'])==3 and [a['finger'] for a in binding['frozen_material_anchors']]==[1,2,3] and not binding['known_contacts_or_forces'])
 check('distinct legacy window from shared registry',p['source_identity']['primary_pose_source_frame_range_half_open']==[200,216] and p['registry_contract']['actual_prefix_entry'] is None)
 init=np.repeat(q0[None],16,axis=0);check('300variable repeated q0 design',init[1:].size==300 and np.array_equal(init[0],q0))
 # Arrays are independent reference calculations, not a new producer candidate.
 out=ROOT/'outputs/independent_prefix_input_review_v3';out.mkdir(parents=True,exist_ok=False)
 report={'status':'PASS_INPUT_PRECHECKS_PRODUCER_PENDING' if all(r['pass'] for r in checks) else 'FAIL_INPUT_PRECHECKS',
  'checks':checks,'source':{'path':str(Path(__file__)),'sha256':sha(__file__)},'protocol_sha256':sha(PROTOCOL),
  'old539d_q1_to_q15_loaded':False,'old539d_q0_loaded':False,'fixture_wrist_loaded':False,'new_candidate_generated':False,
  'physics_steps':0,'optimization_calls':0,'collision_or_LP_queries':0,'GPU_used':False,
  'bone_residual_contract':'weight1*(current_unit_direction-desired_unit_direction), no mm scale/no angle/no sqrtweight',
  'temporal_scale_receipt':'draft raw adjacent and second q differences times.03; no implicit dt or dt² denominator',
  'producer_sparse_Jacobian_and_curve_validation':'PENDING_READY_SOURCE'}
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps(report,indent=2));assert all(r['pass'] for r in checks)
if __name__=='__main__':main()
