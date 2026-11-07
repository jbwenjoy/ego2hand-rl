"""One bounded CPU adapter construction/geometry-autograd contract attempt."""
from pathlib import Path
from unittest.mock import patch
import copy,hashlib,json,os,random,sys,time,traceback
import numpy as np

START=time.perf_counter();ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/full151_intent_binding_CPU_v1'
if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':raise ValueError('Explicit CPU mask -1 required')
if OUT.exists():raise FileExistsError('Preserve every CPU attempt/source snapshot')
OUT.mkdir(exist_ok=False);SNAP=OUT/'source_before';SNAP.mkdir()
def pin(p):
 p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
frozen=Path('D:/workspace/supermimic-rough-prefix-learning-study/vendor/frozen_contact_math')
code={'binding':pin(ROOT/'src/full151_intent/binding.py'),'binding_package':pin(ROOT/'src/full151_intent/__init__.py'),
 'native_io':pin(ROOT/'src/full151_native/io.py'),'native_bundle_loader':pin(ROOT/'src/full151_native/bundle.py'),'native_metadata':pin(ROOT/'src/full151_native/metadata.py'),
 **{'frozen__'+n:pin(frozen/n) for n in ('__init__.py','contact_objective.py','contact_geometry.py','contacts.py','penalty_only_objective.py')}}
for i,(k,v) in enumerate({**code,'checker':pin(__file__)}.items()):(SNAP/f'{i:02d}_{Path(v["path"]).name}').write_bytes(Path(v['path']).read_bytes())
target_dir=ROOT/'outputs/full151_declared_H123_surface_targets_v1'
spec={'schema':'full151_declared_H123_objective_spec:1','binding_manifest':pin(target_dir/'binding_manifest.json'),'target_receipt':pin(target_dir/'receipt.json'),
 'independent_target_review':pin(ROOT/'outputs/full151_declared_targets_numpy_independent_v1/report.json'),
 'input_bundle':pin(ROOT/'outputs/full151_native_input_v1/bundle.json'),'expected_code_pins':code,
 'hyperparameters':{'bandwidth_m':.001,'tolerance':.1,'var':.1},'execution_scope':'CPU_BINDING_VALIDATION_ONLY','training_authorized':False,'GPU_authorized':False}
(OUT/'objective_spec.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
checks={};errors={};denied={};details={}
def check(name,condition):
 try:assert bool(condition() if callable(condition) else condition);checks[name]=True
 except BaseException:checks[name]=False;errors[name]=traceback.format_exc()
def rejects(fn):
 try:fn()
 except (ValueError,KeyError):return True
 return False
def bounded():
 if time.perf_counter()-START>90:raise TimeoutError('CPU adapter contract deadline90s; preserve partial, no retry')
import torch
torch.set_num_threads(2)
import mujoco
for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_collision','mj_geomDistance','mj_contactForce','mj_kinematics','mj_comPos'):
 denied[name]=0
def deny(name):
 def call(*a,**k):denied[name]+=1;raise AssertionError('Native operation forbidden: '+name)
 return call
patches=[patch.object(mujoco,k,deny(k)) for k in denied]
for p in patches:p.start()
sys.path.insert(0,str(ROOT/'src'))
try:
 from full151_intent import binding as b
 check('spec_codepins_equal_frozen_source',spec['expected_code_pins']==b.code_pins())
 rng_py=random.getstate();rng_np=np.random.get_state();rng_torch=torch.get_rng_state().clone()
 bounded();module=b.build_contact_objective(spec)
 after_np=np.random.get_state()
 check('one_actual_constructor_preserves_global_Python_NumPy_Torch_RNG',rng_py==random.getstate() and rng_np[0]==after_np[0] and np.array_equal(rng_np[1],after_np[1]) and rng_np[2:]==after_np[2:] and torch.equal(rng_torch,torch.get_rng_state()))
 check('frozen_forward_and_full512_method_inherited_unchanged',type(module).forward is b.GeometricContactObjective.forward and type(module).full_basis_support_for_evaluation is b.GeometricContactObjective.full_basis_support_for_evaluation)
 check('eight_required_buffers_no_parameters',all(k in module._buffers for k in ('basis512','basis_subset_indices','reference_support512','reference_support64','contact_known','known_no_contact','phase','phase_weight')) and len(list(module.parameters()))==0)
 check('target151_and64stride8_registered',module.frame_count==151 and tuple(module.reference_support512.shape)==(151,1,1,512) and tuple(module.reference_support64.shape)==(151,1,1,64) and torch.equal(module.reference_support64,module.reference_support512[...,::8]))
 check('phase3_is_declared_not_human_truth',module.metadata['phase3_semantics'].startswith('declared') and module.metadata['human_contact_observation_known'] is False and module.metadata['source_confidence_known'] is False and module.phase.eq(3).all())
 check('nominalCOMradius_once_not_crosssection',module.metadata['object_nominal_enclosing_radius_m']==.1290454076850486 and module.geometry_proxy.radius==module.metadata['object_nominal_enclosing_radius_m'] and module.metadata['object_com_offset_body_m']==[0.,0.,0.])
 check('host_actual_geometry_only_no_physics_upload',module.metadata['actual_geometry_host_copy_fields']==['geom_size','geom_pos','geom_friction'] and module.metadata['host_model_uploaded_to_physics'] is False)
 check('producer_unready_flag_not_overwritten_or_promoted',read(target_dir/'binding_manifest.json')['objective_loading_permitted'] is False and module.metadata['producer_objective_loading_permitted'] is False and module.metadata['training_authorized'] is False and module.metadata['GPU_authorized'] is False)
 tr=read(ROOT/'outputs/full151_native_A1L_policy_eval_primary2000_v1/native_reward_transitions.json')
 batch_results=[]
 for size in (32,64):
  bounded();x=torch.tensor([t['x1_reward_snapshot'] for t in tr[:size]],dtype=torch.float32,requires_grad=True)
  m={'start':torch.zeros(size,1),'time':torch.arange(size).float()[:,None],'end':torch.full((size,1),151.)}
  bonus,terms=module(x,m);bonus.sum().backward()
  check(f'savedstate{size}_eager_forward_autograd_finite',tuple(bonus.shape)==(size,) and bonus.dtype==torch.float32 and torch.isfinite(bonus).all() and x.grad is not None and torch.isfinite(x.grad).all())
  check(f'savedstate{size}_staticgeometry_no_velocity_gradient',torch.count_nonzero(x.grad[:,20:40])==0 and torch.count_nonzero(x.grad[:,52:])==0)
  batch_results.append({'size':size,'bonus_minmax':[float(bonus.min()),float(bonus.max())],'gradient_absmax':float(x.grad.abs().max()),'target_frames':terms['geometry_intent_reference_frame'].detach().tolist()})
  if size==32:
   penalty=b.PenaltyOnlyPrior(module).eval().requires_grad_(False)
   pvalue,pterms=penalty(x.detach(),m)
   check('P_ONLY_uses_exact_base_P_no_dt_and_no_supportreward',torch.equal(pvalue,-.1*pterms['geometry_intent_penetration_prior']) and penalty.metadata['support_coefficient']==0. and penalty.metadata['dt_applied_inside_module'] is False)
   check('A2_exact_frozen_support_minus_same_penalty_formula',torch.equal(bonus.detach(),terms['geometry_intent_phase_weight']*terms['geometry_intent_support_match']-.1*terms['geometry_intent_penetration_prior']))
 x=torch.tensor([tr[k]['x1_reward_snapshot'] for k in (0,149,149)],dtype=torch.float32)
 phase=torch.tensor([[0.],[149.],[150.]])
 bounded();value,terms=module(x,{'start':torch.zeros(3,1),'time':phase,'end':torch.full((3,1),151.)})
 check('frame0_149_150_targets_exist_no_alias_to149',torch.equal(terms['geometry_intent_reference_frame'],phase[:,0]) and torch.isfinite(value).all())
 check('terminal150_requires_external_hardactive_mask',module.metadata['active_real_imagination_intervals']==[0,150] and torch.equal(phase[:,0]<150,torch.tensor([True,True,False])))
 # A synthetic module-copy fixture tests the unknown-attraction branch without
 # modifying the frozen target file or the canonical constructed objective.
 bounded();fixture=copy.deepcopy(module)
 deep=torch.tensor([tr[0]['x1_reward_snapshot']],dtype=torch.float32)
 point=fixture.geometry_proxy.hand_points_palm(deep[:,:20])[0,0].detach()
 deep[:,40:43]=point;deep[:,43:52]=torch.eye(3).flatten()
 fixture.phase_weight.zero_()
 unknown,ut=fixture(deep,{'start':torch.zeros(1,1),'time':torch.zeros(1,1),'end':torch.full((1,1),151.)})
 check('synthetic_unknown_cancels_attraction_not_deep_penalty',ut['geometry_intent_penetration_prior'].item()>0 and torch.equal(unknown,-.1*ut['geometry_intent_penetration_prior']) and torch.all(module.phase_weight==1))
 bounded();ampx=torch.tensor([tr[0]['x1_reward_snapshot']],dtype=torch.float32,requires_grad=True)
 with torch.autocast(device_type='cpu',dtype=torch.bfloat16):ampbonus,_=module(ampx,{'start':torch.zeros(1,1),'time':torch.zeros(1,1),'end':torch.full((1,1),151.)})
 ampbonus.sum().backward()
 check('outerCPUautocast_keeps_geometry_FP32_gradient_finite',ampbonus.dtype==torch.float32 and torch.isfinite(ampx.grad).all())
 a=load(target_dir/'intent_targets.npz')
 wrong={k:v[:16] if v.shape[:1]==(151,) else v for k,v in a.items()}
 check('old16_or_padding_refused',rejects(lambda:b.validate_arrays(wrong)))
 wrong={**a,'robot_q':np.zeros((151,20))}
 check('unexpected_robotq_key_injection_refused',rejects(lambda:b.validate_arrays(wrong)))
 wrong=copy.deepcopy(a);wrong['human_contact_known'][0,0]=True
 check('human_truth_fabrication_refused',rejects(lambda:b.validate_arrays(wrong)))
 wrong=copy.deepcopy(a);wrong['declared_intent_frame_known'][0]=False
 check('frame_known_not_any_active_refused',rejects(lambda:b.validate_arrays(wrong)))
 wrong=copy.deepcopy(a);wrong['known_no_contact'][0]=True
 check('inferred_release_refused',rejects(lambda:b.validate_arrays(wrong)))
 wrong=copy.deepcopy(a);wrong['phase_weight'][0]=.5
 check('hidden_phase_switch_weight_refused',rejects(lambda:b.validate_arrays(wrong)))
 wrong=copy.deepcopy(spec);wrong['expected_code_pins']['binding']['sha256']='0'*64
 check('source_tampering_refused_before_another_geometry_constructor',rejects(lambda:b.Full151DeclaredH123ContactObjective(wrong)))
 wrong=copy.deepcopy(spec);wrong['binding_manifest']['sha256']='0'*64
 check('binding_target_source_bytes_changed_refused',rejects(lambda:b.Full151DeclaredH123ContactObjective(wrong)))
 check('GPUfactory_unqualified_refused',rejects(lambda:b.build_contact_objective(spec,device='cuda:0')))
 check('no_native_simulation_or_query_API_called',not any(denied.values()))
 check('CPUonly_noCUDAinit_noWarpEnvWM',not torch.cuda.is_initialized() and not torch.cuda.is_available() and torch.cuda.device_count()==0 and not any(k in sys.modules for k in ('warp','mujoco_warp')))
 check('all_frozen_source_and_input_bytes_unchanged',all(pin(v['path'])==v for v in code.values()) and pin(target_dir/'binding_manifest.json')==spec['binding_manifest'] and pin(target_dir/'receipt.json')==spec['target_receipt'])
 details={'saved_batch_results':batch_results,'unknown_fixture_penetration_prior':float(ut['geometry_intent_penetration_prior'].item()),'one_geometry_constructor_actual':True,'Pwrapper_shares_same_base':True,'FP32_source_formula_unchanged':True}
finally:
 for p in reversed(patches):p.stop()
report={'schema':'full151_declared_H123_binding_CPU_contract:1','status':'PASS_CPU_REPRESENTATION_AND_GEOMETRY_AUTOGRAD_ONLY' if all(checks.values()) else 'CPU_CONTRACT_FAILURE_PRESERVED','passed':sum(checks.values()),'total':len(checks),'checks':checks,'errors':errors,'details':details,'spec':pin(OUT/'objective_spec.json'),'source_pins':code,'checker':pin(__file__),'denied_native_API_calls':denied,'wall_seconds':time.perf_counter()-START,
 'scope':['One canonical CPU constructor, inherited frozen forward/score/P, savedstate32/64 geometry autograd, one shared-base P wrapper and explicit synthetic unknown/deep fixture. No real Env, native query, Warp, physics, Actor/WM, optimizer, GPU or training integration.','Constructor reads the validated full151 input bundle for identity/ABI only; observation-derived target buffers are immutable and not defined from reference robot q or policy contacts.','No useful dynamics/force/forceclosure qualification. Prior proxy and compiled precision failures retained. All151 declared-active is a design hypothesis; the separate128/151 dynamic-cone infeasibility is not erased and no eligibility mask is changed here.','Phase150 target is finite for diagnostics but future training must retain full151runtime hard active phase<150 reward/bootstrap/critic masking; this fixture is not that training integration qualification.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'report':pin(OUT/'report.json'),'passed':report['passed'],'total':report['total'],'status':report['status']}))
if not all(checks.values()):raise SystemExit(2)
