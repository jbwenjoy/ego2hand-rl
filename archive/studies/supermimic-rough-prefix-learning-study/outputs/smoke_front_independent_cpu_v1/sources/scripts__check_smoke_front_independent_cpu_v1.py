"""Three default-only CLI subprocesses plus minimal static engineering diff review."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, os, subprocess, time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/smoke_front_independent_cpu_v1'
PYTHON = Path(r'D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe')
LIFE = Path(r'D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py')

def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    OUT.mkdir(exist_ok=False)
    snapshots = OUT / 'sources'; snapshots.mkdir()
    sources = ['scripts/train_rough_prefix_pen_v2.py', 'scripts/train_rough_prefix_pen_v3.py',
               'scripts/run_rough_prefix_v3.py', 'scripts/run_rough_prefix_v4.py',
               'scripts/build_native_config_cpu_v6.py', 'scripts/build_native_config_cpu_v7.py',
               'src/rough_prefix/training.py', 'src/rough_prefix/native_v2.py',
               'protocols/rough-prefix-v3.json', 'scripts/check_smoke_front_independent_cpu_v1.py']
    before = {name: pin(ROOT/name) for name in sources}
    for name in sources:
        (snapshots/name.replace('/', '__')).write_bytes((ROOT/name).read_bytes())
    spec = importlib.util.spec_from_file_location('reviewed_cpu_cleanup', LIFE)
    life = importlib.util.module_from_spec(spec); spec.loader.exec_module(life)
    assert pin(LIFE)['sha256'] == '40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa'
    checks = {}; runs = []
    protocol = read(ROOT/'protocols/rough-prefix-v3.json')
    checks['protocol_only_engineering_seed0'] = protocol['allowed_roles'] == ['engineering_smoke'] and protocol['allowed_seeds'] == [0]
    checks['physical_false_and_no_shared_enrollment'] = protocol['physical_reference_qualified'] is False and protocol['shared10_enrollment'] is False
    old = (ROOT/'scripts/run_rough_prefix_v3.py').read_text()
    new = (ROOT/'scripts/run_rough_prefix_v4.py').read_text()
    checks['supervisor_only_entry_route_changed'] = old.replace("ENTRY=ROOT/'scripts/train_rough_prefix_pen_v2.py'", "ENTRY=ROOT/'scripts/train_rough_prefix_pen_v3.py'") == new
    entry = (ROOT/'scripts/train_rough_prefix_pen_v3.py').read_text()
    checks['entry_default_returns_before_run_training_import'] = entry.index('if not a.run:print(json.dumps(proof,indent=2));return') < entry.index('from rough_prefix.training import run_training')
    checks['completion_schema_config_result_written_after_wrapper_returns'] = "result=run_training(p,b,output,emit)\n        write(output/'training_completion.json',{'schema':'rough_prefix_training_completion:1','config_sha256':sha(path),'result':result})" in entry
    training = (ROOT/'src/rough_prefix/training.py').read_text()
    checks['completion_result_final_capacity_is_separate_from_native_result'] = "if not scope.qualified_final():" in training and "'rough_capacity_observation': _pin(Path(output) / 'capacity_observation.json')" in training and "'rough_capacity_qualified_final': True" in training
    builder = (ROOT/'scripts/build_native_config_cpu_v7.py').read_text()
    checks['builder_v7_freezes_five_prior_artifacts_before_execution'] = all(s in builder for s in ['manifest.json','contact_targets.npz','rules.json','raw_geometry.json','time_windows.json',"if value['sha256']!=spec['expected_hashes'][key]:raise ValueError"])
    all_cfg = []
    for stem, method in [('a1l','A1L'), ('p_only','P_ONLY'), ('a2','A2')]:
        cfgpath = ROOT/f'configs/{stem}-engineering_smoke-check_v1.json'; cfg = read(cfgpath)
        before[str(cfgpath)] = pin(cfgpath); (snapshots/cfgpath.name).write_bytes(cfgpath.read_bytes())
        all_cfg.append(cfg)
        checks[stem+'_input_scope'] = cfg['role']=='engineering_smoke' and cfg['seed']==0 and cfg['method']==method and cfg['GPU_run_authorized'] is False and cfg['training_authorized'] is True and cfg['training']==protocol['engineering_smoke_training']
        checks[stem+'_all_sourcepins_current'] = all(pin(v['path'])['sha256']==v['sha256'] for v in cfg['source_pins'].values())
        checks[stem+'_prior_binding_matches_arm'] = (cfg['contact_binding'] is None) == (method=='A1L')
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = '-1'
        args = [str(PYTHON), str(ROOT/'scripts/train_rough_prefix_pen_v3.py'), '--config', str(cfgpath)]
        started = time.monotonic(); child = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            stdout, stderr = child.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            cleanup = life.stop_owned(child)
            (OUT/(stem+'_CPU_timeout_cleanup.json')).write_text(json.dumps(cleanup,indent=2)+'\n')
            raise RuntimeError('CPU preflight timed out; no GPU path or retry')
        (OUT/(stem+'_stdout.json')).write_bytes(stdout); (OUT/(stem+'_stderr.txt')).write_bytes(stderr)
        proof = json.loads(stdout)
        checks[stem+'_actual_default_CPU_preflight_pass'] = child.returncode==0 and proof['status']=='CPU_INPUT_SOURCE_PREFLIGHT_ONLY' and proof['method']==method and proof['config_sha256']==pin(cfgpath)['sha256']
        checks[stem+'_no_framework_device_physics'] = proof['Torch_imported'] is False and proof['Warp_imported'] is False and proof['GPU_started'] is False and proof['physics_steps']==0
        runs.append({'method':method, 'args':args, 'returncode':child.returncode, 'elapsed_seconds':time.monotonic()-started, 'proof':proof, 'config':pin(cfgpath), 'GPU_flag':cfg['GPU_run_authorized']})
    checks['three_arms_identical_training_budget'] = all(c['training']==all_cfg[0]['training'] for c in all_cfg)
    budget = all_cfg[0]['training']; acc=0.;steps=budget['warmup_steps']
    for _ in range(budget['max_iterations']):
        acc+=1/budget['utd']; n=int(acc);acc-=n;steps+=n
    budget_counts = {'collect_calls':budget['max_iterations']+1, 'steps_per_world':steps, 'real_transitions':steps*budget['num_envs'], 'wm_optimizer_steps':budget['max_iterations']*budget['wm_batches'], 'failure_head_optimizer_steps':budget['max_iterations']*budget['wm_batches'], 'actor_optimizer_steps':budget['max_iterations']-budget['wm_only_iters'], 'critic_optimizer_steps':(budget['max_iterations']-budget['wm_only_iters'])*budget['critic_epochs']*budget['critic_minibatches']}
    checks['declared_budget_arithmetic'] = budget_counts == {'collect_calls':11,'steps_per_world':41,'real_transitions':5248,'wm_optimizer_steps':20,'failure_head_optimizer_steps':20,'actor_optimizer_steps':8,'critic_optimizer_steps':32}
    checks['frozen_review_sources_unchanged'] = all(pin(p['path'])['sha256']==p['sha256'] for p in before.values())
    report = {'schema':'rough_smoke_front_independent_CPU:1', 'status':'PASS_CPU_PREFLIGHT_AND_STATIC_DIFF' if all(checks.values()) else 'FAIL', 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'checks':checks, 'passed':sum(checks.values()), 'total':len(checks), 'runs':runs, 'source_pins':before, 'expected_training_counts_NOT_MEASURED':budget_counts, 'scope':'Three real default CLI CPU subprocesses only. No --run, supervisor watch, Env/Runner construction, GPU, physics, optimizer or learning executed. Original 47/21 fixture checks not rerun. Formal v7 configurations remain separate hash-bound authorization.', 'physics_steps':0, 'optimizer_steps':0, 'GPU_used':False, 'experimental_source_modified':False}
    path=OUT/'report.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'report':pin(path)}))
    return 0 if all(checks.values()) else 1

if __name__=='__main__':raise SystemExit(main())
