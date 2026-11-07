"""Build CPUFalse metadata with explicit pending final1000; no array/NN/native."""
from pathlib import Path
import hashlib,json,runpy,sys
H=Path(__file__).resolve().parent;P=H.parent
def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
m=runpy.run_path(str(H/'evaluate_learned_actors_v1.py'));train=P/'phone_tripod_supermimic_v1/protocol_CPU_GPU_True_run1.json';tp=read(train)
zero=read(H/'zero_CPUTrue_v2_run1.json');qualified=H/'zero_qualified_run2/report.json';q=read(qualified)
sources=dict(tp['sources'])
sources.update(kernel=pin(H/'controlled_native_kernel_v1.py'),controller=pin(H/'check_zero_controller_v2.py'),kernel_derivation=pin(H/'kernel_derivation_v1.json'),kernel_frozen_loop=pin(P/'functional_tripod_dynamics_v1/run_functional_tripod_dynamics_v3.py'))
inputs=dict(training_seed_protocol=pin(train),training_final_protocol=None,final_result=None,final_health=None,final_terminal=None,
    task_bundle=tp['inputs']['task_bundle'],geometry_catalog=tp['inputs']['geometry_catalog'],model=zero['inputs']['model'],binding=zero['inputs']['binding'],qualified_zero_report=pin(qualified))
p=dict(schema=m['SCHEMA'],role='fixed_final_paired1000_learned_A1_A2_native_original2s_double_dt_evaluation',source=pin(H/'evaluate_learned_actors_v1.py'),sources=sources,inputs=inputs,
    CPU_run_authorized=False,NN_forward_authorized=False,native_execution_authorized=False,GPU_run_authorized=False,training_authorized=False,device='cpu',fixed_paired_round=1000,modes=m['MODES'],dt_s=m['DTS'],
    original_cases=[dict(dt_s=dt,command=next(v for v in q['cases'] if v['dt_s']==dt)['command']) for dt in m['DTS']],
    budget=dict(cases=4,MJB_reads=1,MjData=4,reset_forward=4,maximum_native_steps=38400,maximum_Actor_FW=400,own_checkpoint_CPU_loads=1,Actor_ctors=2,unused_Critic_ctors=2,Critic_FW=0,WM_ctors=0,WM_FW=0,observer_forward=0,optimizer_steps=0,backward_calls=0,GPU=0),
    output=str(H/'learned_fixed1000_run1'),
    tracking=dict(reference_grid=101,control_H_s=.02,alignment='post_native_tick=k*n, k0 initial; no floatclock index or interpolation',orientation='full_SO3_geodesic_BODY_rotation_to_command',position='actual_COM_world_m',missing_tail='NaN_and_unknown_mask_preserved',comparator='phone_frozen_at_actual_initial_COM_and_BODY_R; no new rollout',new_tracking_success_gate=None),
    scope='Own fixed paired1000 only, A1/A2 each2native dt; original2s loop/tail/contact/warning/abort gates unchanged, old2pad parallel gates retained. No external aid or extra observer forward. Actor CPU obs222 current COM/world58 and F64history; future object command is allowed, force labels are not inputs. A2 engineered material/normal proxy was a training prior, not observed contact or force and is not evaluated in this runner. Tracking is descriptive, not holding5mm success. Offline matched training is not completed online SuperMimic/CHORD and evaluation cannot establish ego generalization.',
    preparation=pin(__file__))
write(H/'learned_CPUFalse_v1.json',p);m['metadata_preflight'](H/'learned_CPUFalse_v1.json');assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
write(H/'learned_metadata_v1.json',dict(schema='phone_tripod_final_Actor_metadata:1',status='PASS_METADATA_PENDING_FINAL1000_NOT_EXECUTABLE',protocol=pin(H/'learned_CPUFalse_v1.json'),source=p['source'],Torch_imported=False,MuJoCo_imported=False,NN_arrays_native_calls=0))
print(json.dumps(dict(protocol=pin(H/'learned_CPUFalse_v1.json'),source=p['source'],metadata=pin(H/'learned_metadata_v1.json'))))
