"""Post-hoc complete error arrays; FK/Jac only, no optimization/collision/physics."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,sys,os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,'D:/workspace/supermimic-observation-retarget-study/scripts')
from material_anchors import MaterialAnchor,MaterialAnchorGeometry
from retarget_palmroot_cpu import RobotKinematics,BONES
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
forbidden={n:0 for n in ('mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_geomDistance','mj_contactForce','mj_collision')}
saved={n:getattr(mujoco,n) for n in forbidden}
def deny(n):
 def fail(*a,**kw):forbidden[n]+=1;raise AssertionError(n)
 return fail
rows=[]
try:
 for n in forbidden:setattr(mujoco,n,deny(n))
 for short in ['baseline','midpoint']:
  out=ROOT/f'outputs/frame0_h123_seeded_{short}_v3';recipe=read(out/'recipe_snapshot.json');report=read(out/'report.json');binding=read(out/'frozen_anchor_bindings.json')
  for pin in recipe['pins'].values():assert sha(pin['path'])==pin['sha256']
  assert binding==read(recipe['pins']['parent_bindings']['path'])
  with np.load(out/'initialization_candidate.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
  q=data['joint_pos_rad'][0];pose=data['mapped_object_pose_w'][0];base=data['baseline_joint_pos_rad'][0]
  robot=RobotKinematics(recipe['pins']['model']['path']);m,d=robot.model,robot.data
  with np.load(recipe['pins']['context']['path']) as z:
   for k in ['geom_size','geom_pos']:getattr(m,k)[:]=z[k][0]
  fj=m.joint('object/cube_freejoint');adr=int(fj.qposadr[0]);d.qpos[adr:adr+7]=pose;robot.last_q=None;robot.forward(q)
  helper=MaterialAnchorGeometry(SimpleNamespace(robot=robot,model=m,data=d));points=[];normals=[]
  for b in binding['frozen_material_anchors']:
   a=MaterialAnchor(**{**b,'point_geom_local':np.array(b['point_geom_local']),'support_normal_geom_local':np.array(b['support_normal_geom_local'])});a.point_geom_local.setflags(write=False);a.support_normal_geom_local.setflags(write=False)
   p,j,n,jn=helper.oriented_features(q,a);points.append(p);normals.append(n)
  points,normals=np.array(points),np.array(normals);goals=np.array(binding['target_material_points_w']);ntarget=np.array(binding['target_hand_normals_w']);point_errors=points-goals;normal_errors=normals-ntarget
  bone=robot.bone_features(q)[0];targets=data['morphology_normalized_target_points_w'][0];v=targets[BONES[:,1]]-targets[BONES[:,0]];desired=v/np.linalg.norm(v,axis=1)[:,None];bone_angle=np.rad2deg(np.arccos(np.clip(np.sum(bone*desired,axis=1),-1,1)))
  fkp=robot.forward(q)[0];tiperr=fkp[[4,8,12,16,20]]-targets[[4,8,12,16,20]]
  checks={'bindings_exact_parent':True,'output_npz_SHA':sha(out/'initialization_candidate.npz')==report['output_npz_sha256'],'point_norms_match_report':np.allclose(np.linalg.norm(point_errors,axis=1),report['material_point_goal_error_m'],rtol=0,atol=1e-12),'normal_angles_match_report':np.allclose(np.rad2deg(np.arccos(np.clip(np.sum(normals*ntarget,axis=1),-1,1))),report['material_normal_goal_error_deg'],rtol=0,atol=1e-8),'all_bone_angles_match_saved':np.allclose(bone_angle,data['bone_angle_error_deg'][0],rtol=0,atol=1e-8),'FK_saved_exact':np.array_equal(fkp,data['fk_hand_points_w'][0]),'initial_unaffected8_exact':np.array_equal(data['optimizer_initial_joint_pos_rad'][0,12:],base[12:]),'all_final_q_legal':bool(np.all(q>=robot.lower) and np.all(q<=robot.upper))}
  assert all(checks.values()),checks
  np.savez_compressed(out/'complete_error_arrays.npz',material_points_w=points,material_normals_w=normals,material_point_error_vector_m=point_errors,material_normal_error_vector=normal_errors,bone_direction_error=bone-desired,bone_angle_error_deg=bone_angle,tip_error_vector_m=tiperr,generated_pair_mask=data['generated_contact_mask'][0],generated_pair_signed_distance_m=data['selected_native_contact_distance_m'][0],geometry_pair_indices=data['geometry_pair_indices'],geometry_pair_kind=data['geometry_pair_kind'])
  rows.append({'condition':short,'checks':{k:bool(v) for k,v in checks.items()},'report_sha256':sha(out/'report.json'),'complete_error_npz_sha256':sha(out/'complete_error_arrays.npz'),'source_sha256':sha(__file__)})
finally:
 for n,v in saved.items():setattr(mujoco,n,v)
summary={'status':'COMPLETE_BOTH_FIXED_GEOMETRIC_CONDITIONS','conditions':rows,'forbidden_calls':forbidden,'no_Torch_Warp_CUDA':not any(n in sys.modules for n in ['torch','warp','mujoco_warp']),'physical_reference_qualified':False,'physics_steps':0,'scope':'No new optimization. Independent FK/error-array reconstruction of both retained outputs; unknown native pair gaps remain unknown; generated contacts are not solved load-bearing contacts.'}
assert not any(forbidden.values()) and summary['no_Torch_Warp_CUDA']
write(ROOT/'outputs/h123_warmstart_preparation_v1/completion_review.json',summary)
print(json.dumps(summary,indent=2))
