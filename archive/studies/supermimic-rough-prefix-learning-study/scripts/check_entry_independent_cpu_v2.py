"""Default CLI subprocess + intrinsic preflight negatives; never --run/watch/GPU."""
from __future__ import annotations
import argparse,ast,copy,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,default=ROOT/'configs/a0-native_a0-check_v3.json');ap.add_argument('--entry',type=Path,default=ROOT/'scripts/train_rough_prefix_pen.py');ap.add_argument('--supervisor',type=Path,default=ROOT/'scripts/run_rough_prefix_v2.py');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir();config=a.config.resolve();entry=a.entry.resolve();supervisor=a.supervisor.resolve();p=read(config)
    source_paths={'config':config,'entry':entry,'supervisor':supervisor,'builder':Path(p['source_pins']['config_builder']['path']),'protocol':Path(p['protocol']['path'])}
    source_pins={k:{'path':str(v),'sha256':sha(v)} for k,v in source_paths.items()}
    for key,value in source_paths.items():(out/'sources'/(key+value.suffix)).write_bytes(value.read_bytes())
    results=[];children=[];target=Path(p['output']);target_existed=target.exists()
    def check(name,value,detail=None):results.append({'name':name,'passed':bool(value),**({'detail':detail} if detail else {})})
    hidden=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    def child(name,script):
        command=[sys.executable,str(script),'--config',str(config)]
        r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=25,creationflags=hidden,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
        (out/(name+'_stdout.log')).write_text(r.stdout,encoding='utf-8');(out/(name+'_stderr.log')).write_text(r.stderr,encoding='utf-8')
        children.append({'command':command,'returncode':r.returncode,'scope':'CPU default only; no --run'})
        check(name+'_exit0',r.returncode==0,r.stderr[-1000:] if r.returncode else None)
        d=json.loads(r.stdout) if r.returncode==0 else {}
        check(name+'_actual_CPU_report_no_GPU_frameworks',d.get('status')=='CPU_INPUT_SOURCE_PREFLIGHT_ONLY' and d.get('Torch_imported') is False and d.get('Warp_imported') is False and d.get('GPU_started') is False and d.get('physics_steps')==0)
        check(name+'_config_and_entry_exact',d.get('config_sha256')==sha(config) and d.get('entry_sha256')==sha(entry))
        return d
    try:
        entry_proof=child('entry_default',entry);supervisor_proof=child('supervisor_default',supervisor)
        check('supervisor_venv_base_junction_identity_preserved',supervisor_proof.get('selected_python_identity',{}).get('venv_selected_samefile') is True and supervisor_proof.get('selected_python_identity',{}).get('base_alias_resolved_samefile') is True)
        check('default_preflight_no_native_output_created',target.exists()==target_existed)
        spec=importlib.util.spec_from_file_location('_rough_entry_independent_preflight',entry);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        def negative(name,change):
            d=copy.deepcopy(p);change(d);path=out/(name+'.json');path.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            try:m.preflight(path)
            except (ValueError,AssertionError,KeyError,FileNotFoundError):check(name+'_rejected',True)
            else:check(name+'_rejected',False)
        negative('old_schema',lambda d:d.__setitem__('schema','supermimic_chord_train:1'))
        negative('old_format103',lambda d:d['controller'].__setitem__('format_version',103))
        negative('old_format3',lambda d:d['controller'].__setitem__('format_version',3))
        negative('unknown_rootfield_checkpoint',lambda d:d.__setitem__('checkpoint','old.pt'))
        negative('A0_optimizers_requested',lambda d:d['training'].__setitem__('max_iterations',10))
        negative('A0_training_authorized',lambda d:d.__setitem__('training_authorized',True))
        negative('A0_wrongworlds',lambda d:d['training'].__setitem__('num_envs',2))
        negative('retry',lambda d:d.__setitem__('automatic_retry',True))
        negative('fallback',lambda d:d.__setitem__('compile_fallback',True))
        negative('bool_seed',lambda d:d.__setitem__('seed',True))
        negative('unknown_seed',lambda d:d.__setitem__('seed',4))
        negative('changed_gate',lambda d:d['evaluation_gates'].__setitem__('directed_shaft_error_max_deg',40.))
        negative('bad_bundle_SHA',lambda d:d['input_bundle'].__setitem__('sha256','0'*64))
        negative('bad_protocol_alias',lambda d:d.__setitem__('protocol',d['input_bundle']))
        key=next(k for k,v in p['source_pins'].items() if Path(v['path']).resolve()==(ROOT/'src/rough_prefix/controller.py').resolve())
        negative('missing_controller_sourcepin',lambda d:d['source_pins'].pop(key))
        negative('outside_owned_output',lambda d:d.__setitem__('output','D:/not_rough_owned/native_probe'))
        def primary(d):d.update(role='primary_matched',method='A1L',training_authorized=True,training={'num_envs':128},hard_process_seconds=300,soft_worker_seconds=280)
        negative('formal_primary_not_authorized_by_protocol',primary)
        def smoke_without_A0(d):d.update(role='engineering_smoke',method='A1L',training_authorized=True,training=read(d['protocol']['path'])['engineering_smoke_training'],hard_process_seconds=300,soft_worker_seconds=280)
        negative('smoke_without_nativeA0_evidence',smoke_without_A0)
        # Authorization is necessary but insufficient: the CLI's default remains
        # CPU even if a separately copied config carries a true authorization.
        d=copy.deepcopy(p);d['GPU_run_authorized']=True;path=out/'authorization_flag_default_only.json';path.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8');_,_,proof=m.preflight(path)
        check('authorization_flag_without_run_remains_CPU',proof['GPU_started'] is False and proof['physics_steps']==0 and 'torch' not in sys.modules and 'mujoco_warp' not in sys.modules)
        tree=ast.parse(entry.read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module=='rough_prefix.native']
        check('native_import_only_nested_in_main',len(calls)==1 and calls[0].lineno>next(n.lineno for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'))
        check('no_frameworks_imported_by_independent_preflight',not any(k in sys.modules for k in ('torch','warp','mujoco_warp','mujoco')))
        check('frozen_original_inputs_sources_unchanged',all(sha(v['path'])==v['sha256'] for v in source_pins.values()) and all(sha(v['path'])==v['sha256'] for v in p['source_pins'].values()))
        fault_checks(supervisor,config,out,check)
        static_closure_checks(p,out,check)
        status='PASS' if all(c['passed'] for c in results) else 'FAIL'
        error=None
    except BaseException as exc:status='HARNESS_FAILURE';error=repr(exc)
    report={'status':status,'checks':results,'check_count':len(results),'passed':sum(c['passed'] for c in results),'error':error,'source_pins':source_pins,'reviewer':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'CPU_default_children':children,'execution':{'watch_calls':0,'physical_workers_launched':0,'GPU':0,'physics_steps':0,'training_updates':0},'static_lifecycle_review':'The wrapper uses unchanged f416 Windows argv0/samefile/actualPID+PPID+UTC/source/config/output/terminal-ownedPID gate and unchanged40b independenttimer/ownedtree boundedcleanup; ROOT/WORKER and159cIO are changed only in the imported module instance. NativeA0 Popenhard120/worker110. Quality outcomes never qualify OS identity; missing/unknown identity exit1. No actual GPU lifecycle exercised here.','open_static_observations':[], 'closure_scope':'New supervisorv2 startup and CPUtimeout fixtures are purestub only; defaultCLI actualCPU subprocesses are separate. Sourceclosure covers project/vendored/reviewed lifecycle and selected modified mjlab modules; no claim that all installed PyTorch/MuJoCo dependency source files were archived.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':status,'passed':report['passed'],'count':len(results),'report_sha256':sha(out/'report.json'),'error':error},indent=2));raise SystemExit(status!='PASS')


def fault_checks(supervisor,config,out,check):
    """Run the actual supervisor functions with fake handles and no OS child."""
    from contextlib import contextmanager
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('_rough_supervisor_v2_faults',supervisor);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    base=read(config);minimal=out/'CPU_timeout_stub_config.json'
    # This only reaches preflight's Popen path with a fake process; it is not a
    # native-run config, producer output, authorization or actual subprocess.
    minimal.write_text(json.dumps({'source_pins':{k:{'path':str(p),'sha256':sha(p)} for k,p in [('supervisor',supervisor),('entry',m.ENTRY),('identity',m.DRIVER)]}},indent=2)+'\n',encoding='utf-8')
    original_popen=m.subprocess.Popen;original_driver=m.driver;old_read=m.read;old_preflight=m.preflight;old_reject=m.reject_other_workers;argv=sys.argv[:]
    calls=[]
    class Pipe:
        def __init__(self):self.closed=False
        def close(self):self.closed=True
    class Handle:
        pid=424242;returncode=None
        def __init__(self):self.stdout=Pipe();self.stderr=Pipe();self.timeouts=[]
        def communicate(self,timeout):self.timeouts.append(timeout);raise subprocess.TimeoutExpired(['CPU_STUB_ONLY'],timeout)
    for kind in ('confirmed','cleanup_raises','cleanup_unverified'):
        h=Handle();stop_calls=[]
        def stop_owned(retained):
            stop_calls.append(retained)
            if kind=='cleanup_raises':raise RuntimeError('injected cleanup failure')
            return {'owned_tree_termination_confirmed':kind=='confirmed','scope':'PURE_STUB_NO_OS_KILL'}
        m.subprocess.Popen=lambda *a,**kw:h
        m.driver=lambda:SimpleNamespace(lifecycle=lambda:SimpleNamespace(stop_owned=stop_owned))
        try:
            try:m.preflight(minimal,Path(sys.executable))
            except RuntimeError as exc:
                text=str(exc)
                check('CPUtimeout_'+kind+'_blocks_GPU_and_drains_bounded',stop_calls==[h] and h.timeouts==[20,1] and h.stdout.closed and h.stderr.closed and 'no GPU permitted' in text)
                if kind=='cleanup_raises':check('CPUtimeout_cleanup_exception_explicit', 'cleanup_error' in text and 'injected cleanup failure' in text)
                if kind=='cleanup_unverified':check('CPUtimeout_unknown_not_promoted_confirmed', "'owned_tree_termination_confirmed': False" in text)
            else:check('CPUtimeout_'+kind+'_must_raise',False)
        finally:m.subprocess.Popen=original_popen;m.driver=original_driver
        calls.append({'kind':kind,'stop_owned_retained_handle_calls':len(stop_calls),'communicate_timeouts_s':h.timeouts,'actual_processes':0})
    @contextmanager
    def gpu_lock_ok():yield
    @contextmanager
    def gpu_lock_fail():raise RuntimeError('injected lock refusal');yield
    def forbidden_watch(*a,**kw):raise AssertionError('No watch allowed in startup fault fixture')
    try:
        for stage in ('lock','reject','config_changed'):
            output=out/('startup_'+stage);logs=output.with_name(output.name+'_watchdog')
            fake=copy.deepcopy(base);fake['GPU_run_authorized']=True;fake['output']=str(output)
            # Only an in-memory main-control-flow fixture sees a trueflag.
            # No file authorization is created, lock is fake, watch is forbidden.
            m.read=lambda path,fake=fake:copy.deepcopy(fake) if Path(path).resolve()==config else old_read(path)
            proof={'config_sha256':sha(config) if stage!='config_changed' else '0'*64,'entry_sha256':sha(m.ENTRY),'status':'CPU_STUB_NOT_PHYSICAL'}
            m.preflight=lambda *a,proof=proof,**kw:proof.copy()
            life=SimpleNamespace(gpu_lock=gpu_lock_fail if stage=='lock' else gpu_lock_ok,watch=forbidden_watch)
            m.driver=lambda life=life:SimpleNamespace(selected_python_identity=lambda p:{'CPU_STUB_ONLY':True},lifecycle=lambda:life)
            def reject():
                if stage=='reject':raise RuntimeError('injected otherworker refusal')
            m.reject_other_workers=reject
            sys.argv=[str(supervisor),'--config',str(config),'--run']
            try:m.main()
            except (RuntimeError,ValueError):
                state=read(logs/'state.json');identity=read(logs/'engineering_identity.json')
                check('startup_'+stage+'_explicit_nochild_terminal',state['status']=='failed_no_child_started' and state['watch_invoked'] is False and state['Popen_attempted'] is False and state['automatic_retry'] is False and identity['engineering_identity_qualified'] is False and identity['status']=='NO_CHILD_IDENTITY_NOT_APPLICABLE' and identity['physical_outcome']=='NOT_RUN')
                check('startup_'+stage+'_no_PID_absence_fabrication',not any(k in state for k in ('actual_pid','launcher_pid')) and 'No child launch' in identity['PID_or_CIM_absence_claim'])
            else:check('startup_'+stage+'_must_fail',False)
            calls.append({'stage':stage,'actual_watch_calls':0,'actual_GPU_locks':0,'actual_processes':0})
    finally:
        m.subprocess.Popen=original_popen;m.driver=original_driver;m.read=old_read;m.preflight=old_preflight;m.reject_other_workers=old_reject;sys.argv=argv
    (out/'pure_stub_lifecycle_cases.json').write_text(json.dumps(calls,indent=2)+'\n',encoding='utf-8')


def static_closure_checks(p,out,check):
    paths={Path(v['path']).resolve():v['sha256'] for v in p['source_pins'].values()}
    pure=Path('D:/workspace/supermimic-c2-training-study/src/target_inset.py').resolve()
    check('executed_external_C2_source_pinned_for_archive',paths.get(pure)==sha(pure)=='2d88859c62d2833633f8b09a3be42d11c943e0c3422095e1d941100cb1bf86f4')
    deps=[ROOT/'src/rough_prefix'/name for name in ('native.py','probe.py','runtime.py','controller.py','bundle_v2.py','io.py','metadata.py')]+[ROOT/'vendor/runtime_io.py',ROOT/'vendor/measured_contacts.py',ROOT/'scripts/train_rough_prefix_pen.py',ROOT/'scripts/run_rough_prefix_v2.py']
    check('A0_project_execution_dependency_closure_pinned',all(paths.get(x.resolve())==sha(x) for x in deps))
    check('capacity_not_an_executed_A0_dependency',not any(x.name=='capacity.py' for x in paths))
    native=ROOT/'src/rough_prefix/native.py';tree=ast.parse(native.read_text());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_native')
    branches=[n for n in ast.walk(func) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='is_a0']
    def names(stmts):
        return [n.func.id for s in stmts for n in ast.walk(s) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    a0calls=[name for n in branches for name in names(n.body)]
    check('A0_branch_avoids_actor_trainingCfg_Runner_optimizer_construction',not any(n in a0calls for n in ('DexSuperMimicCfg','RecordedRunner','DexSuperMimicRunner','Actor','WorldModel','Adam','AdamW')) and 'SimpleNamespace' in a0calls and 'PenSimReader' in a0calls and 'run_probe' in a0calls)
    text=native.read_text();check('A0_quality_earlystop_removed_no_manual_pending_clear',"env_cfg.terminations.pop('object_position_error')" in text and "env_cfg.terminations.pop('object_orientation_error')" in text and '_manual_reset_pending' not in text)
    check('all_frozen_config_pins_current_final',all(sha(x)==digest for x,digest in paths.items()))
    (out/'static_A0_dependency_closure.json').write_text(json.dumps({'A0_direct_project_dependencies':[str(x) for x in deps],'external_pure':str(pure),'native_source_sha256':sha(native),'A0_branch_constructor_call_names':a0calls,'scope':'Static branch/source graph only; PenSimReader is not instantiated here; installed package sources beyond selected pinned modified modules are environment dependencies.'},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()

