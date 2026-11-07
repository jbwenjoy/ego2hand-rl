"""Minimal saved-diff + two default CPU CLI checks; no native --run or physics."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,default=ROOT/'configs/a0-native_a0-check_v4.json');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);snap=out/'source';snap.mkdir();p=read(a.config);checks={};children=[]
    files={'reviewer':Path(__file__),'config':a.config,'protocol':Path(p['protocol']['path']),'builder':Path(p['source_pins']['config_builder']['path']),'old_native':ROOT/'src/rough_prefix/native.py','native_v2':ROOT/'src/rough_prefix/native_v2.py','old_entry':ROOT/'scripts/train_rough_prefix_pen.py','entry_v2':ROOT/'scripts/train_rough_prefix_pen_v2.py','old_supervisor':ROOT/'scripts/run_rough_prefix_v2.py','supervisor_v3':ROOT/'scripts/run_rough_prefix_v3.py','runtime_v2':ROOT/'src/rough_prefix/runtime_v2.py','installed_API_review':ROOT/'outputs/reset_observation_API_independent_review_v2/report.json','prior_47_CPU':ROOT/'outputs/entry_supervisor_independent_cpu_v2/report.json'}
    pins={k:{'path':str(v.resolve()),'sha256':sha(v)} for k,v in files.items()}
    for k,v in files.items():(snap/(k+v.suffix)).write_bytes(v.read_bytes())
    def check(name,value):checks[name]=bool(value)
    check('native_only_imports_newruntime',files['native_v2'].read_text()==files['old_native'].read_text().replace('from .runtime import rough_runtime','from .runtime_v2 import rough_runtime'))
    check('CLI_only_newnative_and_sourceclosure_routes',files['entry_v2'].read_text()==files['old_entry'].read_text().replace("'src/rough_prefix/native.py'","'src/rough_prefix/native_v2.py'").replace("'src/rough_prefix/runtime.py'","'src/rough_prefix/runtime_v2.py'").replace('from rough_prefix.native import run_native','from rough_prefix.native_v2 import run_native'))
    expected=files['old_supervisor'].read_text().replace("ENTRY=ROOT/'scripts/train_rough_prefix_pen.py'","ENTRY=ROOT/'scripts/train_rough_prefix_pen_v2.py'").replace('train_rough_prefix_pen|track_prefix_warp','train_rough_prefix_pen(?:_v[0-9]+)?|track_prefix_warp')
    check('supervisor_only_entry_and_advisoryregex_routes',files['supervisor_v3'].read_text()==expected)
    protocol=read(files['protocol']);parent=read(protocol['parent_protocol']['path']);old_config=read(protocol['engineering_correction']['prior_config']['path']);old_runtime=read(protocol['engineering_correction']['prior_runtime']['path'])
    check('protocol_v2_nativeA0only_no_smokes_primary',protocol['allowed_roles']==['native_a0'] and protocol['allowed_seeds']==[0] and protocol['protocol_revision']==2)
    check('parent_protocol_and_prior_lifecycle_proof_preserved',sha(protocol['parent_protocol']['path'])==protocol['parent_protocol']['sha256'] and read(files['prior_47_CPU'])['status']=='PASS')
    check('scientific_controller_initialization_horizon_unchanged',all(protocol[k]==parent[k] for k in ('input_bundle','retarget_variant_id','source_window_half_open','reference_states','control_intervals','duration_s','control_dt_s','physics_dt_s','controller','runtime','native_A0')) and p['evaluation_gates']==old_config['evaluation_gates'])
    check('old_failedattempt0step0update_preserved_noquality_selection',old_runtime['physics_steps']==old_runtime['optimizer_updates']==0 and old_runtime['status']=='failed_engineering_partial' and protocol['engineering_correction']['prior_evaluation']=='NOT_ASSESSED_0_PROBE_STEPS' and protocol['engineering_correction']['new_attempts']==1)
    check('all_prior_failure_evidence_pins_current',all(sha(v['path'])==v['sha256'] for v in protocol['engineering_correction'].values() if isinstance(v,dict) and set(v)=={'path','sha256'}))
    check('config_CPUonly_no_training_no_optimizer',p['GPU_run_authorized'] is False and p['training_authorized'] is False and p['role']=='native_a0' and p['method']=='A0' and p['training']=={'num_envs':1} and p['automatic_retry'] is False and p['compile_fallback'] is False)
    allpaths={Path(v['path']).resolve():v['sha256'] for v in p['source_pins'].values()}
    check('frozen_source_graph_current',all(sha(k)==v for k,v in allpaths.items()))
    check('new_entry_native_runtime_supervisor_and_installedObs_pinned',all(q.resolve() in allpaths for q in (files['entry_v2'],files['native_v2'],files['runtime_v2'],files['supervisor_v3'],Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab/managers/observation_manager.py'),Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab/utils/buffers/circular_buffer.py'),Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab/utils/buffers/delay_buffer.py'))))
    check('controller_probe_math_sources_unchanged',sha(ROOT/'src/rough_prefix/controller.py')=='7f934ba79d95d2e78e3260ed6ed863ef1eaea9cb43fded8fd356a5e3e043fddf' and sha(ROOT/'src/rough_prefix/probe.py')=='7bc6c813d2803b4ed2a8f492cb6586ddd0900534bb81668992a88deb046f82f6')
    check('runtime_v2_exact_reviewed_source',sha(files['runtime_v2'])=='4cbef95884334bfccff8f801d8ff5b083f7399e7ff2f3b530c2fe3548d8ff497' and read(files['installed_API_review'])['status']=='PASS_STATIC_AND_SOURCEBOUND_CPU_PROOF_REVIEW')
    target=Path(p['output']);before=target.exists();hidden=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
    error=None
    try:
        for name in ('entry_v2','supervisor_v3'):
            command=[sys.executable,str(files[name]),'--config',str(a.config.resolve())];t=time.monotonic()
            h=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=hidden,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
            try:stdout,stderr=h.communicate(timeout=30)
            except BaseException:
                life_path=Path('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py')
                spec=importlib.util.spec_from_file_location('_CPU_only_owned_cleanup',life_path);life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
                cleanup=life.stop_owned(h)
                try:h.communicate(timeout=1)
                except BaseException:
                    if h.stdout:h.stdout.close()
                    if h.stderr:h.stderr.close()
                (out/(name+'_CPU_cleanup.json')).write_text(json.dumps(cleanup,indent=2)+'\n');raise
            (out/(name+'_stdout.log')).write_text(stdout,encoding='utf-8');(out/(name+'_stderr.log')).write_text(stderr,encoding='utf-8')
            children.append({'command':command,'retained_CPU_launcher_pid':h.pid,'returncode':h.returncode,'wall_seconds':time.monotonic()-t,'scope':'Default CPU CLI only, no --run; not an actual native worker PID claim'})
            d=json.loads(stdout) if h.returncode==0 else {}
            check(name+'_real_default_CPU_exit0',h.returncode==0)
            check(name+'_real_no_Torch_Warp_GPU_or_physics',d.get('status')=='CPU_INPUT_SOURCE_PREFLIGHT_ONLY' and d.get('Torch_imported') is False and d.get('Warp_imported') is False and d.get('GPU_started') is False and d.get('physics_steps')==0)
            check(name+'_actual_config_newentry_hash_bound',d.get('config_sha256')==sha(a.config) and d.get('entry_sha256')==sha(files['entry_v2']))
        check('default_did_not_create_native_output',target.exists()==before)
        check('all_reviewed_sources_unchanged_after_defaultCPU',all(sha(v['path'])==v['sha256'] for v in pins.values()) and all(sha(k)==v for k,v in allpaths.items()))
    except BaseException as exc:error=repr(exc);check('actual_default_CPU_harness_completed',False)
    report={'status':'PASS_ENGINEERING_CORRECTION_CPU_ONLY' if all(checks.values()) and error is None else 'FAIL_RETAINED','checks':checks,'check_count':len(checks),'passed':sum(checks.values()),'error':error,'source_pins':pins,'CPU_default_children':children,'reviewed_utc':datetime.now(timezone.utc).isoformat(),'boundary':'Only declared import/source routing plus the previously reviewed observation-reset API fix. Original failed A0 remains0steps and no quality outcome; no new outcome-based selection. Actual defaultCPUpreflight executed, old47faultcases not rerun. The retained frozen old47 lifecycle, new25realObs proof and7static review remain separate. This receipt neither authorizes nor executes a native/GPU experiment.','execution':{'physical_workers':0,'GPU':0,'physics_steps':0,'training_updates':0,'fault_cases_repeated':0,'ObservationManager_calls':0}}
    f=out/'report.json';f.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'passed':report['passed'],'count':len(checks),'sha256':sha(f),'error':error},indent=2));raise SystemExit(error is not None or not all(checks.values()))
if __name__=='__main__':main()
