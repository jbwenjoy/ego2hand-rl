"""Static-model/source and saved-array inventory only; no WM inference/physics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import ast,hashlib,json,sys
import numpy as np
import torch
import mujoco
ROOT=Path(__file__).resolve().parents[2]
DEX=Path('D:/workspace/dex-supermimic-research')
OUT=Path(__file__).resolve().parent
def pin(p):p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
FORBIDDEN={n:0 for n in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_collision','mj_geomDistance','mj_contactForce')}
for name in FORBIDDEN:
    def deny(*a,_name=name,**kw):FORBIDDEN[_name]+=1;raise RuntimeError('Forbidden '+_name)
    setattr(mujoco,name,deny)
sources={}
for rel in ('sim.py','pen.py','state.py','rotations.py','runner.py','networks.py'):
    p=DEX/'src/dex_supermimic/rl/dex_supermimic'/rel;sources['native_'+rel]=pin(p)
for rel in ('envs/manager_based_rl_env.py','sim/sim.py','scene/scene.py','sensor/contact_sensor.py','sensor/sensor.py','entity/data.py','entity/entity.py'):
    sources['mjlab_'+rel]=pin(DEX/'.venv/Lib/site-packages/mjlab'/rel)
for rel in ('forward.py','sensor.py','io.py','collision_core.py','types.py','smooth.py'):
    sources['mujoco_warp_'+rel]=pin(DEX/'.venv/Lib/site-packages/mujoco_warp/_src'/rel)
for p in (ROOT/'src/rough_prefix/policy_probe.py',ROOT/'src/rough_prefix/evaluation_v2.py',ROOT/'src/rough_prefix/bundle_v2.py',Path('D:/workspace/supermimic-chord-study/src/chord_study/world_model.py'),Path('D:/workspace/supermimic-chord-study/scripts/probe_world_model_gradients.py'),DEX/'src/dex_supermimic/assets/wuji_hand2.py',DEX/'src/dex_supermimic/tasks/pen_spin/config/wuji_hand2/env_cfgs.py'):
    sources[str(p)]=pin(p)
bundle=read(ROOT/'outputs/input_bundle_v1/bundle.json');model_path=Path(bundle['model']['path']);model=mujoco.MjModel.from_binary_path(str(model_path));body={}
for name in ('robot/mocap_base','robot/right_palm_link','object/cube'):
    i=model.body(name).id;body[name]={'id':i,'parent_id':int(model.body_parentid[i]),'mocap_id':int(model.body_mocapid[i]),'position_body':model.body_pos[i].tolist(),'quaternion_wxyz_body':model.body_quat[i].tolist(),'joints':int(model.body_jntnum[i]),'COM_offset_body_m':model.body_ipos[i].tolist()}
finger_sensors=[]
for i in range(model.nsensor):
    name=model.sensor(i).name
    if 'finger_collision' in name:finger_sensors.append({'name':name,'sensor_type':int(model.sensor_type[i]),'primary_geom_id':int(model.sensor_objid[i]),'primary_geom_name':model.geom(int(model.sensor_objid[i])).name,'secondary_type':int(model.sensor_reftype[i]),'secondary_body_id':int(model.sensor_refid[i]),'intprm':model.sensor_intprm[i].tolist(),'delay_s':float(model.sensor_delay[i])})
query_types={int(mujoco.mjtSensor.mjSENS_GEOMDIST),int(mujoco.mjtSensor.mjSENS_GEOMNORMAL),int(mujoco.mjtSensor.mjSENS_GEOMFROMTO)}
static={'model':pin(model_path),'nq':model.nq,'nv':model.nv,'nu':model.nu,'nflex':model.nflex,'bodies':body,'finger_collision_sensors':finger_sensors,'geom_query_sensor_count':sum(int(t) in query_types for t in model.sensor_type),'scope':'Compiled constants read through MjModel only; no MjData/Env/query/forward/step'}
bridge=npz(Path(bundle['source_pins']['contact_reference']['path']));inventory={};models={}
for method,name in [('A1L','a1l'),('P_ONLY','p_only'),('A2','a2')]:
    run=ROOT/f'outputs/eval_primary_{name}_seed0_v1';state=npz(run/'raw_states.npz');events=read(run/'control_events.json');solver=read(run/'solver_samples.json');raw=[json.loads(x) for x in (run/'raw_contacts.jsonl').read_text().splitlines() if x];xi=npz(run/'reader_context.npz')['reader_context'];checkpoint=ROOT/f'outputs/primary_{name}_seed0_v1/train/model_final.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    inventory[method]={'run':str(run),'raw_states':pin(run/'raw_states.npz'),'control_events':pin(run/'control_events.json'),'solver_samples':pin(run/'solver_samples.json'),'raw_contacts':pin(run/'raw_contacts.jsonl'),'reader_context':pin(run/'reader_context.npz'),'state_shapes':{k:list(v.shape) for k,v in state.items()},'event_keys':list(events[0]),'raw_contact_keys':list(raw[0]),'state61_count':len(state['index']),'solver60_count':len(solver),'events15_count':len(events),'canonical_raw_indices':list(range(0,61,4)),'canonical_time_max_abs_error_s':float(np.max(np.abs(state['raw_time_s'][::4]-np.arange(16)*.02))),'actualu_shapes':[list(np.asarray(e['actualu']).shape) for e in events],'mocap_constant_all61':bool(np.array_equal(state['mocap_pos_w'],np.broadcast_to(state['mocap_pos_w'][:1],state['mocap_pos_w'].shape)) and np.array_equal(state['mocap_quat_wxyz'],np.broadcast_to(state['mocap_quat_wxyz'][:1],state['mocap_quat_wxyz'].shape))),'mocap_equals_bridge_palm0_FP32':bool(np.array_equal(state['mocap_pos_w'][0,0].astype(np.float32),bridge['wrist_pose_w'][0,:3].astype(np.float32)) and np.array_equal(state['mocap_quat_wxyz'][0,0].astype(np.float32),bridge['wrist_pose_w'][0,3:].astype(np.float32))),'all_saved_exogenous_force0':bool((state['external_force_max_abs']==0).all()),'native_state58_directly_saved':False,'native_reward_snapshot_physics_directly_saved':False,'native_self15_sensor_directly_saved':False,'raw_contact_type_saved':'type' in raw[0],'raw_contact_efc_address_saved':'efc_address' in raw[0],'raw_contact_rows':len(raw),'zero_normal_force_rows':sum(c['force6_contact_frame'][0]==0 for c in raw),'positive_gap_rows':sum(c['distance_m']>0 for c in raw),'15_target_source':'control_events[k].actualu[0]; raw ctrl[4k+1:4k+5] heldsame; startup ctrl index0 separate','self15_truth_status':'UNKNOWN_DIRECT_NATIVE_SENSOR_NOT_SAVED','reader_xi_shape':list(xi.shape)}
    wm=saved['modules']['wm'];models[method]={'checkpoint':pin(checkpoint),'format_version':saved['format_version'],'iteration':saved['iteration'],'context_dim_task':saved['context_dim'],'contact_dim':saved['contact_dim'],'WM_input_width':wm['net.0.weight'].shape[1],'dynamics_width_without_u':wm['net.0.weight'].shape[1]-20,'reader_physical_context_width':xi.shape[1],'disturbance_width':wm['net.0.weight'].shape[1]-20-58-xi.shape[1],'WM_output_width':wm['net.9.weight'].shape[0],'WM_delta_width':wm['delta_norm.mean'].numel(),'hidden':list(saved['config']['wm_hidden']),'all_WM_tensors_finite':all(bool(torch.isfinite(v).all()) for v in wm.values()),'native_module':'WorldModel(dynamics_dim=278,contact_dim=15,hidden=(512,512,512),amp=False) CPU strict state_dict load in future104consumer; no load/inference executed here'}
report={'schema':'native104_saved_WM_raw_interface_review:1','status':'STATE_CONTROL_CPU_RECONSTRUCTION_FEASIBLE_SELF15_TRUTH_UNKNOWN','static_model':static,'source_pins':sources,'saved_inventory':inventory,'three_final_WM_headers':models,'reviewer_source':pin(__file__),'executed_native_state_reconstruction':False,'WM_forward_calls':0,'new_data_collection':False,'GPU_used':False,'CUDA_initialized':torch.cuda.is_initialized(),'forbidden_calls':FORBIDDEN,'scope':'Read-only source/static model/saved headers review. Data is adequate for a separately strict104 native-state CPU reconstruction and descriptive prediction errors; no native self15 calibration or causal-gradient qualification is asserted.'}
if (OUT/'report.json').exists():raise FileExistsError('Review already exists')
dump(OUT/'report.json',report);dump(OUT/'source_pins.json',sources);dump(OUT/'static_model_metadata.json',static)
print(json.dumps({'status':report['status'],'report':pin(OUT/'report.json'),'native_forbidden_calls':FORBIDDEN,'WM_forward_calls':0,'GPU_used':False},indent=2))
