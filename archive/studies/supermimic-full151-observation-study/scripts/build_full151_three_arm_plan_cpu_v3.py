"""Freeze one common-source, fixed-budget A1L/P_ONLY/A2 run family."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_native.io import checked,pin,read,write
from full151_native.probe import GATES,CLOCK_CONTRACT
METHODS=['A1L','P_ONLY','A2']
COMMON=dict(num_envs=128,buffer_steps=256,warmup_steps=64,wm_window=8,wm_batches=2,wm_batch=256,
    pi_batch=128,horizon=8,max_horizon=8,critic_epochs=2,critic_minibatches=2,holdout_fraction=.1,
    log_interval=1,amp=True,compile=False,compile_policy=False,compile_policy_dynamic=False,
    action_noise=.1,utd=.4,wm_hidden=[512,512,512])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--role',choices=['engineering_smoke','primary_matched'],required=True);ap.add_argument('--authorize-gpu',action='store_true');ap.add_argument('--paired-smoke',type=Path);a=ap.parse_args()
    if any(k in sys.modules for k in ('torch','warp','mujoco_warp')):raise AssertionError('Metadata-only builder imported a device framework')
    tag='smoke40' if a.role=='engineering_smoke' else 'primary2000'
    protocol_path=ROOT/f'protocols/full151-v3-three-arm-{tag}-v1.json'
    config_paths={m:ROOT/f'configs/full151-v3-{tag}-{m}-seed0-v1.json' for m in METHODS}
    if protocol_path.exists() or any(p.exists() for p in config_paths.values()):raise FileExistsError('Freeze once; never overwrite')
    base=read(ROOT/'configs/full151-a0-authorized-v1.json');bundle=base['input_bundle']
    ready={
      'native_A0_engineering_enrollment':pin(ROOT/'outputs/full151_native_A0_engineering_enrollment_v1.json'),
      'learning_CPU_READY':pin(ROOT/'outputs/full151_prior_learning_preparation_v1/READY.json'),
      'binding_CPU_READY':pin(ROOT/'outputs/READY_full151_intent_binding_v3.json'),
      'health_auditor_CPU_READY':pin(ROOT/'outputs/READY_full151_three_arm_health_v2.json'),
      'matched_cone_report':pin(ROOT/'outputs/full151_matched_actual_material_cones_CPU_v1/report.json'),
      'matched_cone_saved_numeric_report':pin(ROOT/'outputs/full151_matched_material_cones_saved_numeric_independent_v1/report.json')}
    contracts={};ready_objects={}
    for name in ('learning_CPU','binding_CPU','health_auditor_CPU'):
        r=read(checked(ready[name+'_READY']));ready_objects[name]=r
        ready[name+'_report']=r['cpu_report'];report=read(checked(r['cpu_report']))
        if not report['checks'] or not all(v is True for v in report['checks'].values()) or report['passed']!=report['total'] or r['cpu_checks']!={'passed':report['passed'],'total':report['total']}:raise ValueError('Actual positive CPU proof required')
        contracts[name]={'schema':report['schema'],'status':report['status']}
    if a.role=='primary_matched':
        if a.paired_smoke is None:raise ValueError('Actual independently paired smoke required')
        ready['paired_smoke_enrollment']=pin(a.paired_smoke)
    source={};seen={}
    def include(key,value):
        path=checked(value).resolve();marker=str(path).casefold()
        if marker in seen:
            if seen[marker]!=value['sha256']:raise ValueError('Same file has conflicting expected bytes')
            return
        seen[marker]=value['sha256'];source[key]=value
    for key,value in base['source_pins'].items():
        if key not in ('worker','supervisor','plan_builder','protocol'):include('base__'+key,value)
    for key,value in ready.items():include('readiness__'+key,value)
    for name,r in ready_objects.items():
        for key,value in r['source_pins'].items():include(name+'__'+key,value)
        for key,value in read(checked(r['cpu_report']))['source_pins'].items():include(name+'_report__'+key,value)
    # Explicit --no-ignore is necessary for installed backends under .venv.
    roots=[Path('D:/workspace/dex-supermimic-research/src/dex_supermimic'),Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab'),Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mujoco_warp')]
    paths=subprocess.run(['rg','--files','--hidden','--no-ignore',*(str(p) for p in roots)],capture_output=True,text=True,check=True).stdout.splitlines()
    for i,path in enumerate(sorted(x for x in paths if x.endswith('.py'))):include(f'native_all_python_{i:04d}',pin(path))
    for folder in ('full151_native','full151_intent'):
        for path in sorted((ROOT/'src'/folder).glob('*.py')):include(folder+'__'+path.name,pin(path))
    for key,path in {'worker':ROOT/'scripts/execute_full151_learning_v3.py','supervisor':ROOT/'scripts/run_full151_learning_v3.py','builder':Path(__file__),
        'eval_worker':ROOT/'scripts/execute_full151_policy_eval_v2.py','eval_supervisor':ROOT/'scripts/run_full151_policy_eval_v2.py','eval_builder':ROOT/'scripts/build_full151_policy_eval_plan_cpu_v2.py'}.items():include(key,pin(path))
    for key,value in read(ROOT/'configs/full151-matched-material-cones-CPU-authorized-v1.json')['source_pins'].items():include('cone__'+key,value)
    br=ready_objects['binding_CPU'];cpu_spec=read(checked(br['CPU_spec']))
    spec_path=ROOT/'configs/full151-v2-declared-H123-GPU-objective-spec-v1.json';gpu_spec={**cpu_spec,'GPU_authorized':True}
    if spec_path.exists():
        if read(spec_path)!=gpu_spec:raise ValueError('Frozen shared objective specification differs')
    else:write(spec_path,gpu_spec)
    objective_spec=pin(spec_path);include('objective_spec',objective_spec);include('CPU_tested_objective_spec',br['CPU_spec'])
    for name in ('binding_manifest','target_receipt','independent_target_review','prior_CPU_binding_review','actual_material_contract','matched_material_screen','matched_material_saved_review','matched_material_lifecycle_review','legacy_proxy_dynamic_cone_review'):
        include('objective__'+name,gpu_spec[name])
    manifest=read(checked(gpu_spec['binding_manifest']));include('declared_target',manifest['target'])
    for key,value in manifest['source_pins'].items():include('intent_input__'+key,value)
    for key,value in gpu_spec['expected_code_pins'].items():include('objective_code__'+key,value)
    if a.role=='primary_matched':
        paired=read(a.paired_smoke);include('paired_smoke_review',paired['saved_pair_review']);include('smoke_protocol',paired['smoke_protocol'])
        for method,hpin in paired['health'].items():
            include('smoke_health__'+method,hpin);h=read(checked(hpin));include('smoke_config__'+method,h['config']);include('smoke_auditor__'+method,h['auditor'])
            for key,value in read(checked(h['config']))['source_pins'].items():
                if key!='protocol':include('smoke_source__'+method+'__'+key,value)
    smoke=COMMON|dict(max_iterations=40,wm_only_iters=20,save_interval=40)
    primary=COMMON|dict(max_iterations=2000,wm_only_iters=200,save_interval=100)
    protocol={'schema':'full151_knownrough_matched_training_protocol:2','allowed_methods':METHODS,'allowed_roles':['engineering_smoke','primary_matched'],'allowed_seeds':[0],
        'input_bundle':bundle,'objective_spec':objective_spec,'readiness':ready,'CPU_report_contracts':contracts,
        'engineering_smoke_training':smoke,'primary_matched_training':primary,'reference_states':151,'control_intervals':150,
        'physical_reference_qualified':False,'contact_targets_are_human_truth':False,'clock_contract':CLOCK_CONTRACT,'evaluation_gates':GATES,
        'hypothesis':'H123 source-surface declaration fixed before H12345 comparison; no highercoverage winner selection. mu.1 support fingerprint unchanged; native material condim4 conditional151 witness is separate evidence, not actualcontact/force/handreach.',
        'method_difference':'A1L: no objective import/instance; P_ONLY: same -.1 penetration; A2: same penetration +1 support. Only imagination soft=True reward +.02*unscaledbonus; native phase<150 mask/realcollection/WMfit/controller/optimizers unchanged.',
        'fairness':'Same seed0/128worlds/context/reference/controller/budget/source table; raw four parameter initializations independently matched. Entire prior import/build/to RNG compared before firstcollect. No warmstart/modelresume or quality checkpoint selection.',
        'primary_budget_counts':{'real_transitions':648192,'wm':4000,'fail_head':4000,'actor':1800,'critic':7200},
        'smoke_budget_counts':{'real_transitions':20992,'wm':80,'fail_head':80,'actor':20,'critic':80},
        'qualification_limits':'Prior GPU eager engineering only after actualsmoke; historical proxy/compiled precision failure retained; physical/contact gradients and realego benefits unqualified. Current source SCP geometry and A0 task failures retained.',
        'lifecycle':'One fresh output per fixed config. Smoke soft540/hard600; primary soft7100/hard7200. No retry/compile fallback. User conditional Goalpause after first healthy newlystarted primary checkpoint and ETA; owned detached job may continue.'}
    write(protocol_path,protocol);source['protocol']=pin(protocol_path)
    for method,path in config_paths.items():
        cfg={'schema':'full151_knownrough_native_training_run:2','role':a.role,'method':method,'seed':0,'input_bundle':bundle,
            'output':str(ROOT/f'outputs/full151_v3_{tag}_{method}_seed0_v1'),'training':smoke if a.role=='engineering_smoke' else primary,
            'contact_binding':None if method=='A1L' else objective_spec,'source_pins':source,'protocol':pin(protocol_path),
            'GPU_run_authorized':bool(a.authorize_gpu),'training_authorized':True,
            'hard_process_seconds':600 if a.role=='engineering_smoke' else 7200,'soft_worker_seconds':540 if a.role=='engineering_smoke' else 7100,
            'automatic_retry':False,'compile_fallback':False,'controller':{'kind':'FULL151_KNOWNROUGH_BASELINE_EMA_C2','format_version':105,'margin_rad':.020,'resume':'from_scratch_only'},
            'evaluation_gates':GATES,'readiness':ready}
        write(path,cfg)
    print(json.dumps({'protocol':pin(protocol_path),'configs':{m:pin(p) for m,p in config_paths.items()},'source_pins':len(source),'objective_spec':objective_spec,'GPU_started':False,'Torch_imported':False,'Warp_imported':False}))
if __name__=='__main__':main()
