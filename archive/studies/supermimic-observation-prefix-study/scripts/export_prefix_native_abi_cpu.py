"""Explicit post-candidate CPU ABI export; default validates sources only.

Only --run reads a supplied, SHA-bound real candidate. No solve or simulation.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--contract',type=Path,required=True);ap.add_argument('--run',action='store_true')
 ap.add_argument('--candidate',type=Path);ap.add_argument('--candidate-sha256');ap.add_argument('--output',type=Path);args=ap.parse_args()
 contract=read(args.contract)
 if contract['schema']!='legacy_prefix16_native_ABI_CPU_contract:1' or contract['physical_reference_qualified'] is not False:raise ValueError('Contract')
 for name,p in contract['pins'].items():
  if sha(p['path'])!=p['sha256']:raise ValueError('Source drift: '+name)
 if not args.run:
  print(json.dumps({'status':'SOURCE_PREFLIGHT_ONLY_CANDIDATE_NOT_READ','contract_sha256':sha(args.contract),'candidate_loaded':False,'physics_steps':0,'GPU_used':False}));return
 if not args.candidate or not args.candidate_sha256 or not args.output:raise ValueError('Explicit real candidate/SHA/new output required')
 if sha(args.candidate)!=args.candidate_sha256:raise ValueError('Candidate SHA mismatch')
 if args.output.exists():raise FileExistsError('Never overwrite or silently retry ABI outcome')
 import numpy as np
 import mujoco
 from scipy.spatial.transform import Rotation
 sys.path.insert(0,str(Path(contract['pins']['root_exporter']['path']).parent))
 from native_reference_abi import export_clip
 from dex_supermimic.tasks.pen_spin.tooling.recipe import resolve_recipe
 from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingReader
 import torch
 def forbidden(*a,**kw):raise AssertionError('Native ABI export is FK-only')
 for n in ('mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_collision','mj_geomDistance','mj_contactForce'):setattr(mujoco,n,forbidden)
 keys=('time_s','q_goal_rad','mapped_object_pose_w')
 with np.load(args.candidate,allow_pickle=False) as z:t,q,pose=[np.array(z[k],dtype=np.float64,copy=True) for k in keys]
 if t.shape!=(16,) or q.shape!=(16,20) or pose.shape!=(16,7) or not all(np.isfinite(a).all() for a in (t,q,pose)) or not np.array_equal(t,np.arange(16)*.02):raise ValueError('Require complete16frame artifact, no partial fill')
 with np.load(contract['pins']['canonical_q0']['path'],allow_pickle=False) as z:q0=z['joint_pos_rad'][0]
 with np.load(contract['pins']['frozen_geometry']['path'],allow_pickle=False) as z:expected_pose=z['mapped_object_pose_w'][:16]
 if not np.array_equal(q[0],q0) or not np.array_equal(pose,expected_pose):raise ValueError('Fixed q0/pose/window mismatch')
 args.output.mkdir(parents=True);group=args.output/'clips/legacy_observation_H123';clipid='clip007_f0200_n0016_prefix16_H123'
 native,sidecar,info=export_clip(group,clipid,t,q,pose,model_path=contract['pins']['model']['path'],constants_path=contract['pins']['constants']['path'],
  provenance={'is_real_ego':False,'source_robot_labels_used':False,'observation_source':contract['pins']['observation_fixture'],
   'geometric_derivative':{'path':str(args.candidate.resolve()),'sha256':args.candidate_sha256,'loaded_keys':list(keys)},
   'source_identity':contract['source_identity'],'unknown_confidence':None,'shared10_registry_enrolled':False,
   'native_FD_not_continuous_curve_init_velocity':True,'init_measured':None,'u_executed':None})
 write(group/'index.json',{'schema':'wuji_pen_spin_clips:1','group_id':'legacy_observation_H123','group_name':'Legacy distinctprefix geometry ABI only','frame_dt':.02,'clip_count':1,
  'clips':[{'clip_id':clipid,'file':native.name,'sha256':sha(native)}]})
 recipe=args.output/'native_recipe.json';write(recipe,{'schema':'wuji_pen_spin_recipe:1','clips':['legacy_observation_H123/'+clipid]})
 selection=resolve_recipe(recipe,clips_root=args.output/'clips',verify_clip_hashes=True);loaded=SingleHandTrackingReader(selection.clip_sets[0],device='cpu').read()
 if loaded['trajectory_lengths'].tolist()!=[16]:raise ValueError('Reader length')
 q32=q.astype(np.float32);nativeqd=np.empty_like(q32);nativeqd[:-1]=np.diff(q32,axis=0)/np.float32(.02);nativeqd[-1]=nativeqd[-2]
 if not np.array_equal(loaded['joint_pos'].numpy(),q32) or not np.array_equal(loaded['joint_vel'].numpy(),nativeqd):raise ValueError('Reader q/native FD mismatch')
 with np.load(native,allow_pickle=False) as z:wrist=z['wrist_pose_w'];links=z['link_pos_w'];obj=z['object_pose_w']
 rw=Rotation.from_quat(wrist[:,[4,5,6,3]]).as_matrix();ro=Rotation.from_quat(obj[:,[4,5,6,3]]).as_matrix()
 got=loaded['object_pose'].numpy();gotR=Rotation.from_quat(got[:,[4,5,6,3]]).as_matrix()
 errors={'object_pos_m':float(np.max(abs(got[:,:3]-np.einsum('tji,tj->ti',rw,obj[:,:3]-wrist[:,:3])))),
  'object_R':float(np.max(abs(gotR-np.einsum('tji,tjk->tik',rw,ro)))),
  'native_BODY21_m':float(np.max(abs(loaded['link_pos'].numpy()-np.einsum('tji,tkj->tki',rw,links-wrist[:,None,:3]))))}
 if max(errors.values())>2e-6:raise ValueError('Independent relative frame ABI mismatch: '+str(errors))
 assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
 report={'status':'COMPLETE_PREFIX16_NATIVE_ABI_CPU_ONLY','contract_sha256':sha(args.contract),'candidate_sha256':args.candidate_sha256,
  'native_npz_sha256':sha(native),'sidecar_sha256':sha(sidecar),'recipe_sha256':sha(recipe),'source_identity':contract['source_identity'],
  'frames':16,'prefix_starts':[0],'prefix_ends':[16],'q_FP32_exact':True,'native_FD_FP32_exact':True,'relative_frame_errors':errors,
  'native_FD_initial_qd_FP32_rad_s':nativeqd[0].tolist(),'continuous_curve_init_velocity':'NOT_READ_OR_SUBSTITUTED; separate external design contract',
  'native_body_FK_source':info['link_source'],'init_measured':None,'actual_u':None,'physical_reference_qualified':False,'training_authorized':False,
  'shared10_registry_enrolled':False,'physics_steps':0,'GPU_used':False,'cuda_initialized':torch.cuda.is_initialized()}
 write(args.output/'report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
