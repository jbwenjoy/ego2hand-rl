"""Both fixed H123 q64 saved-contact object screens; no collision/physics/GPU."""
from pathlib import Path
import os,sys,json,hashlib,importlib.util,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import mujoco
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1]
LP=Path('D:/workspace/supermimic-observation-grounding-study/scripts/diagnose_rest_wrench_feasibility_cpu_v2.py')
LP_SHA='ba3f0f934ff7cc2ce6d674c92f3a755487bcc0994f04736664c5ceb87f31fdca'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
 start=time.perf_counter();out=ROOT/'outputs/h123_object_wrench_screen_v1';out.mkdir(exist_ok=False)
 assert sha(LP)==LP_SHA
 spec=importlib.util.spec_from_file_location('pinned_h123_native_lp',LP);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());cases={}
 for seed in ['baseline','midpoint']:
  directory=ROOT/f'outputs/frame0_h123_seeded_{seed}_v3';recipefile=ROOT/f'configs/frame0-h123-seeded-{seed}-v3.json';recipe,original=read(recipefile),read(directory/'report.json')
  assert sha(recipefile)==original['recipe_sha256'] and sha(directory/'initialization_candidate.npz')==original['output_npz_sha256']
  checks={k:sha(v['path'])==v['sha256'] for k,v in recipe['pins'].items()};assert all(checks.values())
  m=mujoco.MjModel.from_binary_path(recipe['pins']['model']['path']);obj=m.body('object/cube').id;og=set(np.flatnonzero(m.geom_bodyid==obj))
  assert m.npair==0 and not m.flg_adhesion and not np.any(m.geom_adhesion) and m.opt.cone==mujoco.mjtCone.mjCONE_PYRAMIDAL
  with np.load(recipe['pins']['context']['path']) as z:mass=float(z['body_mass'][0,obj]);frictions=z['geom_friction'][0].astype(float)
  with np.load(directory/'initialization_candidate.npz') as z:
   pose=z['mapped_object_pose_w'][0].copy();selfmask=(z['geometry_pair_kind']=='hand_self')&z['generated_contact_mask'][0];self_count=int(selfmask.sum())
  records=read(directory/'native_contacts.json');contacts=[]
  for i,c in enumerate(records):
   assert c['phase']=='native_geometry_query_not_solved_force';g1,g2=c['geom'];assert (g1 in og)!=(g2 in og);p1,p2=m.geom_priority[[g1,g2]]
   if p1==p2:condim=int(max(m.geom_condim[g1],m.geom_condim[g2]));f=np.maximum(frictions[g1],frictions[g2])
   else:g=g1 if p1>p2 else g2;condim=int(m.geom_condim[g]);f=frictions[g]
   f5=np.array([f[0],f[0],f[1],f[2],f[2]]);assert condim==c['effective_condim'] and np.array_equal(f5,c['friction5'])
   contacts.append({'id':str(i),'frame':(1. if g2 in og else -1.)*np.array(c['frame_rows_geom0_to_geom1_w']),'point':np.array(c['position_w']),'friction':f5,'condim':condim})
  com=pose[:3]+helper.quat_matrix(pose[3:])@m.body_ipos[obj];target=np.r_[-mass*m.opt.gravity.astype(np.float32).astype(float),np.zeros(3)]
  full,W=helper.solve(contacts,com,target);force=linprog(np.ones(W.shape[1]),A_eq=W[:3],b_eq=target[:3],bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  force_report={'feasible':bool(force.success),'status':int(force.status),'message':str(force.message)}
  if force.success:force_report.update(edge_weights_N=force.x.tolist(),force_residual_N=(W[:3]@force.x-target[:3]).tolist(),COM_torque_Nm=(W[3:]@force.x).tolist())
  case={'candidate':{'path':str(directory/'initialization_candidate.npz'),'sha256':sha(directory/'initialization_candidate.npz')},'native_contacts':{'path':str(directory/'native_contacts.json'),'sha256':sha(directory/'native_contacts.json')},'recipe_sha256':sha(recipefile),'source_checks':checks,'contact_count':len(contacts),'edge_count':int(W.shape[1]),'self_generated_pairs':self_count,'required_object_contact_wrench_world':target.tolist(),'COM_world':com.tolist(),'force_only_LP':force_report,'full6D_LP':full,'normal_force_cap_N':None,'physical_reference_qualified':False,'PD_preload_exported':False}
  cases[seed]=case;np.savez_compressed(out/f'{seed}_matrices.npz',native_world_wrench_matrix=W,required_wrench=target,COM_world=com)
 report={'status':'COMPLETE_BOTH_PREDECLARED_H123_STATIC_SCREENS','cases':cases,'source_sha256':sha(__file__),'native_pyramid_helper':{'path':str(LP),'sha256':LP_SHA},'scope':'Frozen saved q64 native geometry and exact installed pyramidal edges; static existence only, no solved force/softcontact depth law/wholehand PD or dynamic qualification. Both initial conditions retained without best selection. Screen is not a prerequisite selecting physical launches.','new_collision_queries':0,'new_physics_steps':0,'GPU_used':False,'Torch_or_Warp_imported':any(k in sys.modules for k in ['torch','warp','mujoco_warp']),'wall_seconds_excluding_imports':time.perf_counter()-start}
 write(out/'report.json',report)
 print(json.dumps({k:{'contacts':v['contact_count'],'rays':v['edge_count'],'force_feasible':v['force_only_LP']['feasible'],'full6D_feasible':v['full6D_LP']['feasible_in_native_object_wrench_screen']} for k,v in cases.items()},indent=2))
if __name__=='__main__':main()
