"""Freeze one source-pinned native full151 learning budget; no device imports."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_native.io import checked,pin,read,write
from full151_native.probe import GATES,CLOCK_CONTRACT

COMMON=dict(num_envs=128,buffer_steps=256,warmup_steps=64,wm_window=8,wm_batches=2,wm_batch=256,
    pi_batch=128,horizon=8,max_horizon=8,critic_epochs=2,critic_minibatches=2,holdout_fraction=.1,
    log_interval=1,amp=True,compile=False,compile_policy=False,compile_policy_dynamic=False,
    action_noise=.1,utd=.4,wm_hidden=[512,512,512])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--role',choices=['engineering_smoke','primary_matched'],required=True);ap.add_argument('--authorize-gpu',action='store_true');ap.add_argument('--smoke-health',type=Path);a=ap.parse_args()
    if any(x in sys.modules for x in ('torch','warp','mujoco_warp')):raise AssertionError('Plan cannot import device frameworks')
    tag='smoke40' if a.role=='engineering_smoke' else 'primary2000'
    config_path=ROOT/f'configs/full151-a1l-{tag}-authorized-v1.json'
    protocol_path=ROOT/f'protocols/full151-native-a1l-{tag}-v1.json'
    if config_path.exists() or protocol_path.exists():raise FileExistsError('Freeze once; no overwrite')
    old=read(ROOT/'configs/full151-a0-authorized-v1.json');bundle=old['input_bundle']
    ready={'native_A0_engineering_enrollment':pin(ROOT/'outputs/full151_native_A0_engineering_enrollment_v1.json'),
           'training_CPU_READY':pin(ROOT/'outputs/full151_learning_preparation_v1/READY.json')}
    if a.role=='primary_matched':
        if a.smoke_health is None:raise ValueError('Actual smoke health required')
        ready['engineering_smoke_health']=pin(a.smoke_health)
    smoke=COMMON|dict(max_iterations=40,wm_only_iters=20,save_interval=40)
    primary=COMMON|dict(max_iterations=2000,wm_only_iters=200,save_interval=100)
    protocol={'schema':'full151_knownrough_native_training_protocol:1','allowed_roles':['engineering_smoke','primary_matched'],
        'allowed_methods':['A1L'],'allowed_seeds':[0],'input_bundle':bundle,'reference_states':151,'control_intervals':150,
        'engineering_smoke_training':smoke,'primary_matched_training':primary,'readiness':ready,
        'physical_reference_qualified':False,'geometric_candidate_qualified':False,'known_rough_reference':True,'contact_objective':None,
        'clock_contract':CLOCK_CONTRACT,'evaluation_gates':GATES,'initialization':'Native fromscratch RNG; no warmstart/resume/checkpoint input. New105 identity only.',
        'capacity_scope':'Frozen sticky observer starts after Env constructor; constructor/startup graph before observer UNKNOWN. Read before each reset/checkpoint and one final. Fixed192/4096; no automatic expansion.',
        'quality_scope':'Parent SCP and A0 task FAIL retained. Training uses native soft error termination .05m/60degrees and reference phase150 terminal. Record phase histogram; a budget of164 worldsteps does not imply all150 phases visited.',
        'comparison_scope':'Current source observation-derived full151 A1L only. P_ONLY/A2 independent full151 intent/wrench bindings MISSING. Prior results known; no blind prospective causal improvement claim.',
        'learning_scope':'Real Manager/DexRunner, native replay collection/WM/failure/Actor/critic; actual four module optimizer hook counts and raw initialization saved. H8 hard-bound masking preserves native soft survival semantics.',
        'pause_scope':'After starting primary training, independently inspect first healthy indexed checkpoint100 and estimate remaining time using measured smoke policy cost plus live WM-only cost; then pause Goal while owned hidden supervisor continues.',
        'attempt_policy':'One run per fresh output/config; no retry/fallback. Smoke soft540/hard600; primary soft7100/hard7200. Failures and old bytes retained.'}
    source={};seen={}
    def include(key,value):
        marker=str(checked(value).resolve()).casefold()
        if marker in seen:
            if seen[marker]!=value['sha256']:raise ValueError('Samefile unequal source hashes')
            return
        seen[marker]=value['sha256'];source[key]=value
    for key,value in old['source_pins'].items():
        if key not in ('worker','supervisor','plan_builder','protocol'):include(key,value)
    for key,value in ready.items():include('readiness__'+key,value)
    for path in sorted((ROOT/'src/full151_native').glob('*.py')):include('learning_full151__'+path.name,pin(path))
    for key,path in {'worker':ROOT/'scripts/execute_full151_learning_v1.py','supervisor':ROOT/'scripts/run_full151_learning_v1.py','plan_builder':Path(__file__)}.items():include(key,pin(path))
    for name in ('check_full151_training_cpu.py','audit_full151_checkpoint_cpu_v1.py'):include('checker__'+name,pin(ROOT/'scripts'/name))
    include('training_CPU_report',pin(ROOT/'outputs/full151_training_CPU_v1/report.json'))
    if a.role=='primary_matched':
        h=read(a.smoke_health)
        for key in ('checkpoint','config','auditor'):checked(h[key])
        # Large checkpoint is a read-only prerequisite, not an execution source
        # and cannot become a training input or be loaded by this worker.
        include('smoke_config',h['config']);include('smoke_auditor',h['auditor'])
    write(protocol_path,protocol);source['protocol']=pin(protocol_path)
    config={'schema':'full151_knownrough_native_training_run:1','role':a.role,'method':'A1L','seed':0,'input_bundle':bundle,
        'output':str(ROOT/f'outputs/full151_native_A1L_{tag}_v1'),'training':smoke if a.role=='engineering_smoke' else primary,
        'contact_binding':None,'source_pins':source,'protocol':pin(protocol_path),'GPU_run_authorized':bool(a.authorize_gpu),
        'training_authorized':True,'hard_process_seconds':600 if a.role=='engineering_smoke' else 7200,
        'soft_worker_seconds':540 if a.role=='engineering_smoke' else 7100,'automatic_retry':False,'compile_fallback':False,
        'controller':{'kind':'FULL151_KNOWNROUGH_BASELINE_EMA_C2','format_version':105,'margin_rad':.020,'resume':'from_scratch_only'},
        'evaluation_gates':GATES,'readiness':ready}
    write(config_path,config)
    print(json.dumps({'config':pin(config_path),'protocol':pin(protocol_path),'pins':len(source),'GPU_started':False,'Torch_imported':False,'Warp_imported':False}))

if __name__=='__main__':main()
