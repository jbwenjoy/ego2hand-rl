"""CPU negatives for split, window, pending targets, replay provenance and budget guards."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import copy,hashlib,json,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shared_reference.reference_registry import open_reference_registry,resolve_init_requested,validate_initialization_identity,require_training_ready
from shared_reference.replay_metadata import make_transition_metadata,replay_window_mask
from shared_reference.target_registry import TargetRegistry,MissingTargetBundleError
from shared_reference.budget_transfer import shared_budget,assert_matched_resource_counters,assert_exact_transfer_identity
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,allow_nan=False),encoding='utf-8');return sha(p)
def main():
 rp=ROOT/'configs/derived_reference_registry_v1.json';tp=ROOT/'configs/independent_targets_pending_v1.json'
 reg=open_reference_registry(rp,expected_sha256=sha(rp));rows=[]
 def check(name,func):
  try:func();rows.append({'name':name,'pass':True})
  except Exception as e:rows.append({'name':name,'pass':False,'error':repr(e)})
 def equal(a,b):assert a==b,(a,b)
 def rejects(func,exception=ValueError):
  try:func()
  except exception:return
  raise AssertionError('Expected rejection')
 train='within_lineage_train8';t=reg.track(train);sid='01_single_axis/clip_007';e=reg.entry(sid)
 check('train8 physically excludes009010',lambda:equal(sorted(Path(x['path']).stem[:8] for x in t['materialized_copies']),[f'clip_{i:03d}' for i in range(1,9)]))
 check('single conservative lineage all10',lambda:equal(reg.payload['lineage'],reg.payload['split_group']))
 zero=resolve_init_requested(reg,sid,velocity_design='zero_rest');fd=resolve_init_requested(reg,sid,velocity_design='derived_native_FD')
 check('requested velocity designs distinct from measured and u',lambda:equal((np.count_nonzero(zero['qd_init_requested_rad_s'])==0,np.linalg.norm(fd['qd_init_requested_rad_s'])>0,zero['init_measured'],zero['u_executed0_rad'],zero['physical_qualified']),(True,True,None,None,False)))
 ev={'source_id':sid,'source_frame_range_half_open':[0,151],'source_sha256':e['source_sha256'],'native_npz_sha256':e['canonical_native']['sha256'],
  'model_sha256':reg.payload['proposed_controlled_physics']['model']['sha256'],'physics_context_sha256':reg.payload['proposed_controlled_physics']['context']['sha256']}
 check('matching initialization remains unqualified',lambda:equal(validate_initialization_identity(reg,sid,ev)['physical_qualified'],False))
 bad=copy.deepcopy(ev);bad['source_frame_range_half_open']=[200,351]
 check('reject old007 physical window despite same source',lambda:rejects(lambda:validate_initialization_identity(reg,sid,bad)))
 badsha=copy.deepcopy(ev);badsha['native_npz_sha256']='0'*64
 check('reject other derived q artifact',lambda:rejects(lambda:validate_initialization_identity(reg,sid,badsha)))
 check('CPU registry never authorizes training',lambda:rejects(lambda:require_training_ready(reg,train,targets_ready=True),RuntimeError))
 u=np.arange(40,dtype=np.float32).reshape(2,20)/100
 def meta(sources=(sid,'01_single_axis/clip_001'),start=(0,151),end=(151,302),time=(3,7),use='train',track=train):
  return make_transition_metadata(reg,track,sources,use=use,memory_start=start,memory_end=end,memory_time=time,episode_ids=[4,9],actual_u=u)
 m=meta()
 check('native phase uses start plus current local not nextphase',lambda:equal((m['native_global_reference_frame'].tolist(),m['source_frame'].tolist()),([3,158],[3,7])))
 check('actual executedu retained independently of desiredq',lambda:equal(np.array_equal(m['actual_u'],u) and not np.array_equal(u[0],zero['q_goal0_rad']),True))
 check('trajectory IDs remain metadata outside modelfeatures',lambda:equal((m['metadata_is_network_input'],m['track_trajectory_id'].dtype.kind),(False,'i')))
 check('reject external009 in training replay',lambda:rejects(lambda:meta(sources=(sid,'01_single_axis/clip_009'))))
 check('reject correct source wrong native prefix',lambda:rejects(lambda:meta(start=(151,0))))
 check('reject validation catalog train use',lambda:rejects(lambda:meta(track='within_lineage_validation009',sources=('01_single_axis/clip_009',)*2,start=(0,0),end=(151,151))))
 window={'track_trajectory_id':np.array([[0,0,1,1]]),'episode_id':np.array([[1,1,2,2]]),
  'raw_local_time':np.array([[149,150,0,1]]),'native_start':np.array([[0,0,151,151]]),'native_end':np.array([[151,151,302,302]])}
 done=np.array([[False,True,False,False]]);valid=np.ones((1,4),bool)
 check('terminal transition retained but reset suffix masked',lambda:equal(replay_window_mask(window,done=done,valid=valid).tolist(),[[1,1,0,0]]))
 check('reject unmarked trajectory episode switch',lambda:rejects(lambda:replay_window_mask(window,done=np.zeros((1,4),bool),valid=valid)))
 drift={k:v[:,:2].copy() for k,v in window.items()};drift['raw_local_time'][0,1]+=1
 check('reject continuing phase drift',lambda:rejects(lambda:replay_window_mask(drift,done=np.zeros((1,2),bool),valid=np.ones((1,2),bool))))
 straight={k:np.repeat(v[:,:1],4,axis=1) for k,v in window.items()};straight['raw_local_time']=np.array([[0,1,2,3]])
 check('invalid data poisons remainder of sampled window',lambda:equal(replay_window_mask(straight,done=np.zeros((1,4),bool),valid=np.array([[True,False,True,True]])).tolist(),[[1,0,0,0]]))
 tr=TargetRegistry(reg,tp,expected_sha256=sha(tp));requests=tr.requests(train,memory_start=[0,151],memory_end=[151,302],memory_time=[-2,999])
 check('target lookup native prefix and clamped currentphase',lambda:equal([(r['source_id'],r['local_frame']) for r in requests],[(sid,0),('01_single_axis/clip_001',150)]))
 check('all real targets missing no fabricated zeros',lambda:equal((len(tr.arrays),len(tr.bindings)),(0,10)))
 check('missing targets failclosed',lambda:rejects(lambda:tr.gather(train,memory_start=[0],memory_end=[151],memory_time=[0]),MissingTargetBundleError))
 # Synthetic payloads exercise lookup only; tempdir is deleted and never enrolled as scientific targets.
 with tempfile.TemporaryDirectory(prefix='shared_contract_fixture_') as td:
  td=Path(td);manifest=read(tp);basis=np.zeros((512,6),np.float32);basis[:,0]=1
  manifest['support_basis_sha256']=hashlib.sha256(basis.tobytes()).hexdigest()
  for i,r in enumerate(manifest['bindings']):
   with np.load(reg.entry(r['source_id'])['canonical_native']['path'],allow_pickle=False) as z:n={k:z[k] for k in z.files}
   p=td/f'target{i}.npz';np.savez_compressed(p,time_s=np.arange(151)*.02,support_basis=basis,support512=np.full((151,1,1,512),i+1,np.float32),
    contact_known=np.ones(151,bool),known_no_contact=np.zeros(151,bool),phase=np.arange(151)%5,
    input_joint_pos_rad=n['reference_qpos'],input_object_pose_w=n['object_pose_w'],input_wrist_pose_w=n['wrist_pose_w'])
   r.update(status='FROZEN_KINEMATIC_REFERENCE_GEOMETRY',target={'path':str(p),'sha256':sha(p)})
  mp=td/'targets.json';save(mp,manifest);fixture=TargetRegistry(reg,mp,expected_sha256=sha(mp))
  result=fixture.gather(train,memory_start=[0,151],memory_end=[151,302],memory_time=[4,7])
  check('synthetic target pertrajectory gather not global localclock alias',lambda:equal((result['support512'][:,0,0,0].tolist(),result['phase'].tolist()),([7.,1.],[4,2])))
  bad=copy.deepcopy(manifest);bad['bindings'][6]['source_frame_range_half_open']=[200,351];bp=td/'bad.json';save(bp,bad)
  check('target oldwindow injection rejected',lambda:rejects(lambda:TargetRegistry(reg,bp,expected_sha256=sha(bp))))
  changed=reg.as_dict();changed['tracks']['within_lineage_validation009']['allowed_uses'].append('train');cp=td/'badregistry.json';save(cp,changed)
  check('registry heldout use relabel rejected',lambda:rejects(lambda:open_reference_registry(cp,expected_sha256=sha(cp))))
 counters={a:shared_budget(8) for a in ('A1L','P_ONLY','A2')}
 check('DRAFT matched counters only',lambda:equal(assert_matched_resource_counters(counters,n=8)['training_executed_or_verified'],False))
 bad=copy.deepcopy(counters);bad['A2']['teacher_transitions']=1
 check('reject hidden extra teacher cost',lambda:rejects(lambda:assert_matched_resource_counters(bad,n=8)))
 bad=copy.deepcopy(counters);bad['A2']['actor_optimizer_steps']+=1
 check('reject unequal updates',lambda:rejects(lambda:assert_matched_resource_counters(bad,n=8)))
 desc={'registry_identity_sha256':reg.payload['registry_identity_sha256'],'native_reference_signature':t['native_reference_signature_CPU'],
  'model_sha256':ev['model_sha256'],'physics_context_sha256':ev['physics_context_sha256'],
  'controller_sha256':reg.payload['proposed_controlled_physics']['controller_pure']['sha256'],'control_hz':50,
  'layout':reg.as_dict()['declared_layout'],'data_exposure_source_ids':list(t['source_order']),'format_version':'metadata_identity_only_fixture'}
 check('matching transfer metadata grants no runtime resume',lambda:equal(assert_exact_transfer_identity(reg,train,desc)['native_training_loader_bound'],False))
 old=copy.deepcopy(desc);old['format_version']=103
 check('reject oldformat103 as shared transfer',lambda:rejects(lambda:assert_exact_transfer_identity(reg,train,old)))
 leaked=copy.deepcopy(desc);leaked['data_exposure_source_ids'].append('01_single_axis/clip_009')
 check('reject hidden external trajectory exposure',lambda:rejects(lambda:assert_exact_transfer_identity(reg,train,leaked)))
 out=ROOT/'outputs/shared_registry_contract_checks_v1';out.mkdir(exist_ok=False)
 report={'status':'PASS' if all(r['pass'] for r in rows) else 'FAIL','checks':rows,'count':len(rows),
  'registry_sha256':sha(rp),'targets_sha256':sha(tp),'script_sha256':sha(__file__),'synthetic_target_scope':'temporary lookup fixtures only, deleted; no qualified contact targets produced',
  'physics_steps':0,'training_executed':False,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']}
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps(report,indent=2));assert report['status']=='PASS'
if __name__=='__main__':main()
