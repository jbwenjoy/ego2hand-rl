"""Materialize split-specific native catalogs and UNQUALIFIED init designs on CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from dex_supermimic.tasks.pen_spin.tooling.recipe import resolve_recipe
from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingReader,object_velocity_from_pose
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shared_reference import open_catalog
from shared_reference.reference_registry import open_reference_registry
from shared_reference.target_registry import TargetRegistry,MissingTargetBundleError

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}

def main():
 started=time.monotonic();out=ROOT/'outputs/shared_registry_cpu_v1';out.mkdir(exist_ok=False)
 batch=ROOT/'outputs/geometry_native_batch_v1';ready=read(batch/'READY.json')
 g1=read(ROOT/'catalogs/audit_all10.json');entries=[]
 for row in g1['entries']:
  sid=row['source_id'];short=sid.split('/')[-1];name=short+'_f0000_n0151'
  native=batch/'clips/observation_derived_geometry'/f'{name}.npz';geometry=batch/'geometry'/short/'geometric_retarget.npz'
  with np.load(native,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
  q32=values['reference_qpos'].astype(np.float32);qd0=(q32[1]-q32[0])/np.float32(.02)
  lin,ang=object_velocity_from_pose(torch.from_numpy(values['object_pose_w'].astype(np.float32)),frame_dt=.02)
  initial=out/'init_requested'/f'{short}.npz';initial.parent.mkdir(exist_ok=True)
  np.savez_compressed(initial,q_goal0_rad=values['reference_qpos'][0],native_FD_goal_qd0_rad_s=qd0,
   q_init_requested_FP32_rad=q32[0],object_pose_init_requested_FP32_w=values['object_pose_w'][0].astype(np.float32),
   zero_qd_rad_s=np.zeros(20,np.float32),zero_object_velocity=np.zeros(3,np.float32),
   derived_object_linvel0_w_m_s=lin[0].numpy(),derived_object_angvel0_w_rad_s=ang[0].numpy(),
   default_fixed_wrist_pose_w=values['wrist_pose_w'][0])
  entries.append({'source_id':sid,'source_sha256':row['source_sha256'],'observation_export_sha256':row['export_sha256'],
   'source_frame_range_half_open':[0,151],'frames':151,'canonical_native':pin(native),'geometry':pin(geometry),
   'native_provenance':pin(native.with_suffix('.provenance.json')),'init_requested_candidates':pin(initial),
   'init_measured':None,'physical_initialization_status':'UNQUALIFIED'})
 tracks={};audits=[]
 for name in ('audit_all10','within_lineage_train8','within_lineage_validation009','within_lineage_test010','seen_cost_all10'):
  cp=ROOT/'catalogs'/f'{name}.json';cat=open_catalog(cp,expected_sha256=sha(cp));g=read(cp)
  directory=out/'native_catalogs'/name;clips=directory/'clips/shared_geometry';clips.mkdir(parents=True)
  indexrows=[];copies=[]
  for sid in cat.source_ids:
   e=next(e for e in entries if e['source_id']==sid);src=Path(e['canonical_native']['path']);dst=clips/src.name;shutil.copyfile(src,dst)
   indexrows.append({'clip_id':src.stem,'file':src.name,'sha256':sha(dst)})
   copies.append({'source_id':sid,**pin(dst)})
  index=clips/'index.json';recipe=directory/'native_recipe.json'
  write(index,{'schema':'wuji_pen_spin_clips:1','group_id':'shared_geometry','group_name':name+' UNQUALIFIED geometry','frame_dt':.02,'clip_count':len(indexrows),'clips':indexrows})
  write(recipe,{'schema':'wuji_pen_spin_recipe:1','clips':['shared_geometry/'+r['clip_id'] for r in indexrows]})
  selection=resolve_recipe(recipe,clips_root=directory/'clips',verify_clip_hashes=True)
  assert len(selection.clip_sets)==1
  loaded=SingleHandTrackingReader(selection.clip_sets[0],device='cpu').read();lengths=loaded['trajectory_lengths'].numpy()
  assert lengths.tolist()==[151]*len(cat.source_ids)
  actualids=[c.clip_id for c in selection.clip_sets[0].clips]
  assert actualids==['shared_geometry/'+r['clip_id'] for r in indexrows]
  ends=np.cumsum(lengths);starts=np.r_[0,ends[:-1]];h=hashlib.sha256();rows=[]
  for key in ('joint_pos','object_pose','link_pos'):h.update(loaded[key].contiguous().numpy().tobytes())
  h.update(ends.astype(np.int64).tobytes())
  for e,start,end in zip([next(e for e in entries if e['source_id']==s) for s in cat.source_ids],starts,ends):
   with np.load(e['canonical_native']['path'],allow_pickle=False) as z:
    q=z['reference_qpos'].astype(np.float32);v=np.empty_like(q);v[:-1]=(q[1:]-q[:-1])/np.float32(.02);v[-1]=v[-2]
   assert np.array_equal(q,loaded['joint_pos'][start:end].numpy()) and np.array_equal(v,loaded['joint_vel'][start:end].numpy())
   rows.append({'source_id':e['source_id'],'start':int(start),'end':int(end),'q_and_per_clip_FD_exact':True})
  tracks[name]={'G1_catalog':pin(cp),'source_order':list(cat.source_ids),'allowed_uses':g['allowed_uses'],'frame_dt_s':.02,
   'trajectory_lengths':lengths.tolist(),'trajectory_starts':starts.tolist(),'trajectory_ends':ends.tolist(),
   'native_recipe':pin(recipe),'native_index':pin(index),'clips_root':str(directory/'clips'),'materialized_copies':copies,
   'native_reference_signature_CPU':h.hexdigest(),'native_training_entry_bound':False}
  audits.append({'track':name,'native_reader_CPU_loaded':True,'clips':rows,'frames':int(ends[-1]),'signature':h.hexdigest()})
  print(json.dumps({'track':name,'native_reader_CPU_loaded':True,'clips':len(rows),'frames':int(ends[-1])}),flush=True)
 physics={'model':pin('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb'),
  'context':pin('D:/workspace/supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled/fixed_physics_context.npz'),
  'controller_pure':pin('D:/workspace/supermimic-c2-training-study/src/target_inset.py')}
 layout={'WM_state':58,'absolute_executed_u':20,'actor_observation':155,'native_selfcollision_head':15,'context':46,'disturbance':174,
  'joint_names':[f'right_finger{f}_joint{j}' for f in range(1,6) for j in range(1,5)]}
 identity={'references':[{k:e[k] for k in ('source_id','source_frame_range_half_open','source_sha256','canonical_native')} for e in entries],
  'tracks':{k:{j:v[j] for j in ('source_order','native_reference_signature_CPU')} for k,v in tracks.items()},'physics':physics,'layout':layout}
 registry={'schema':'shared_derived_reference_registry:1','training_entry_integration':'PENDING','training_authorized':False,
  'physical_reference_qualified':False,'lineage':g1['lineage'],'split_group':g1['split_group'],
  'upstream_batch_READY':pin(batch/'READY.json'),'upstream_protocol':g1['upstream_bundle'],
  'proposed_controlled_physics':physics,'declared_layout':layout,'entries':entries,'tracks':tracks,
  'registry_identity_sha256':hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
  'initialization_contract':{'init_requested':'Geometric q0 and explicit zero_rest or nativeFD kinematic velocity design, no actual state measurement',
   'init_measured':None,'executed_u':None,'default_mocap':'No runtime snapshot captured; future worker must restore exact nominal default mocap and verify actual context/FP32 state',
   'freejoint_velocity_conversion':'Object linear world; derived angular world; future native freejoint angular must be explicitly converted using actual pose and API convention',
   'window_nontransfer':'new clip007[0,151) cannot inherit old physical clip007[200,351) initialization'},
  'pending':['independent per-reference target bundles','physical initialization/reset qualification','native multi-track training-entry integration','actual transfer/replay runtime binding']}
 rp=ROOT/'configs/derived_reference_registry_v1.json';write(rp,registry)
 reg=open_reference_registry(rp,expected_sha256=sha(rp))
 targets={'schema':'shared_independent_target_registry:1','reference_registry_sha256':sha(rp),'support_basis_sha256':None,
  'bindings':[{'source_id':e['source_id'],'native_reference_sha256':e['canonical_native']['sha256'],'source_frame_range_half_open':[0,151],
   'frames':151,'target_defined_from_current_policy':False,'no_force_or_teacher_claim':True,'status':'MISSING_INDEPENDENT_BUNDLE','target':None} for e in entries]}
 tp=ROOT/'configs/independent_targets_pending_v1.json';write(tp,targets);tr=TargetRegistry(reg,tp,expected_sha256=sha(tp))
 t=tracks['within_lineage_train8']
 try:tr.gather('within_lineage_train8',memory_start=t['trajectory_starts'],memory_end=t['trajectory_ends'],memory_time=[0]*8)
 except MissingTargetBundleError:pass
 else:raise AssertionError('Missing targets must fail closed')
 assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
 report={'status':'CPU_REFERENCE_REGISTRY_AND_FAIL_CLOSED_CONTRACTS_ONLY','registry':pin(rp),'pending_targets':pin(tp),'native_reader_tracks':audits,
  'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'cuda_initialized':torch.cuda.is_initialized(),'physics_steps':0,'training_executed':False,
  'all_initialization_qualified':False,'native_training_integration':'PENDING','wall_seconds':time.monotonic()-started,
  'sources':{str(p.relative_to(ROOT)):pin(p) for p in [Path(__file__),*sorted((ROOT/'src/shared_reference').glob('*.py'))]}}
 write(out/'report.json',report)
 print(json.dumps({'status':report['status'],'report':str(out/'report.json')}),flush=True)

if __name__=='__main__':main()
