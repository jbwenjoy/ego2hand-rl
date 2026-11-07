"""Three completed saved policy runs: common input/init/cadence, not benefit."""
from pathlib import Path
import hashlib,itertools,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
closure=['scripts/evaluate_rough_prefix_pen_v2.py','scripts/run_rough_evaluation_v2.py','src/rough_prefix/evaluation_v2.py','src/rough_prefix/policy_probe.py','src/rough_prefix/checkpoint_v2.py','src/rough_prefix/controller.py','src/rough_prefix/runtime_v2.py','src/rough_prefix/native_v2.py','src/rough_prefix/bundle_v2.py','src/rough_prefix/io.py','src/rough_prefix/metadata.py','vendor/runtime_io.py','vendor/measured_contacts.py']
rows=[];checks={};pairs={}
for method,name in [('A1L','a1l'),('P_ONLY','p_only'),('A2','a2')]:
    run=ROOT/f'outputs/eval_smoke_{name}_v2';config=ROOT/f'configs/{name}-policy-evaluation-v2.json';review=ROOT/f'outputs/policy_saved_audit_{name}_v2/report.json';c=read(config);h=read(review);s=load(run/'raw_states.npz');events=read(run/'control_events.json');solver=read(run/'solver_samples.json');current={}
    for rel in closure:
        want=(ROOT/rel).resolve();pp=[v for v in c['source_pins'].values() if Path(v['path']).resolve()==want]
        if len(pp)!=1:raise ValueError('Explicit execution dependency missing/duplicate '+rel)
        current[rel]=pp[0]
    checks[method+'_completed_engineering_saved_audit']=h['engineering_qualified'] is True and h['passed']==h['total']==38 and all(h['checks'].values()) and h['config']==pin(config) and h['checkpoint']==c['checkpoint'] and h['evaluation']==pin(run/'evaluation.json')
    rows.append({'method':method,'config':c,'config_pin':pin(config),'audit_pin':pin(review),'run':run,'state':s,'closure':current,'events':events,'solver':solver,'controller':read(run/'controller_descriptor.json'),'context':load(run/'actual_context_readbacks.npz'),'planned':load(run/'requested_initialization_and_control.npz')})
for a,b in itertools.combinations(rows,2):
    label=a['method']+'__'+b['method'];ca,cb=a['config'],b['config'];exclude={'method','output','checkpoint','training_config','health_report','actor_load_report','source_pins'}
    checks[label+'_common_evaluation_config_exact']=canonical({k:v for k,v in ca.items() if k not in exclude})==canonical({k:v for k,v in cb.items() if k not in exclude})
    checks[label+'_explicit_execution_closure_exact']=a['closure']==b['closure']
    fields=('full_qpos','full_qvel','object_pose_w','object_twist_native','mocap_pos_w','mocap_quat_wxyz','ctrl_rad')
    checks[label+'_actual_initial_full_state_bytes_equal']=all(same(a['state'][k][0],b['state'][k][0]) for k in fields)
    checks[label+'_actual8_both_readbacks_bytes_equal']=set(a['context'])==set(b['context']) and all(same(a['context'][k],b['context'][k]) for k in a['context'])
    checks[label+'_requested_init_control_mapping_equal']=set(a['planned'])==set(b['planned']) and all(same(a['planned'][k],b['planned'][k]) for k in a['planned'])
    checks[label+'_same61_clock']=same(a['state']['raw_time_s'],b['state']['raw_time_s']) and same(a['state']['expected_time_s'],b['state']['expected_time_s'])
    ec=lambda row:[{k:e[k] for k in ('phase','preaction_phase','postaction_phase','terminated','truncated')} for e in row['events']]
    sc=lambda row:[{k:s[k] for k in ('step','prestate_index','poststate_index','solver_t_s','poststate_t_s','control_interval_used','phase')} for s in row['solver']]
    checks[label+'_same15_phase_and60_solver_cadence']=ec(a)==ec(b) and sc(a)==sc(b)
    cc=('native_reference_signature','baseline_u15_FP32_sha256','target_lower_rad','target_upper_rad','margin_rad','pure_projection_sha256','controller_source_sha256','active_intervals','terminal_observation')
    checks[label+'_same_reference_baseline_C2_contract']=canonical({k:a['controller'][k] for k in cc})==canonical({k:b['controller'][k] for k in cc})
    pairs[label]={'common_input_bundle':ca['input_bundle'],'common_protocol':ca['protocol'],'distinct_checkpoint_pins':[ca['checkpoint'],cb['checkpoint']],'interpretation':'Same input/init/nativecadence doesnot imply equal learned actions, finalweights, actual trajectories, compute or task benefit.'}
checks['pure_CPU_no_environment_imports']=not any(k.split('.')[0] in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules)
report={'schema':'native_policy_saved_three_pair_comparability:1','status':'PASS_COMMON_INPUT_INIT_CADENCE_ONLY' if all(checks.values()) else 'FAIL','checks':checks,'passed':sum(checks.values()),'total':len(checks),'participants':[{k:r[k] for k in ('method','config_pin','audit_pin')} for r in rows],'pairs':pairs,'source':pin(__file__),'source_training_initial_fairness':pin(ROOT/'outputs/smoke_three_arm_initial_fairness_v1/report.json'),'scope':'Three pairwise engineering identity comparisons. All models are seed0 smoke checkpoints; these are not independent trainingseeds or a benefit/success-rate estimate. Individual physicalquality failures remain unchanged in the original evaluation/audit receipts.','GPU_used':False,'physics_steps':0,'environment_created':False}
p=Path(__file__).with_name('report.json')
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(p)},indent=2));raise SystemExit(int(not all(checks.values())))
