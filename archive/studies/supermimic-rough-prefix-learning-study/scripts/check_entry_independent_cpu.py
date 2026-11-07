"""Default CLI subprocess + intrinsic preflight negatives; never --run/watch/GPU."""
from __future__ import annotations
import argparse,ast,copy,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,default=ROOT/'configs/a0-native_a0-check_v1.json');ap.add_argument('--entry',type=Path,default=ROOT/'scripts/train_rough_prefix_pen.py');ap.add_argument('--supervisor',type=Path,default=ROOT/'scripts/run_rough_prefix.py');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir();config=a.config.resolve();entry=a.entry.resolve();supervisor=a.supervisor.resolve();p=read(config)
    source_paths={'config':config,'entry':entry,'supervisor':supervisor,'builder':ROOT/'scripts/build_native_config_cpu.py','protocol':Path(p['protocol']['path'])}
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
        status='PASS' if all(c['passed'] for c in results) else 'FAIL'
        error=None
    except BaseException as exc:status='HARNESS_FAILURE';error=repr(exc)
    report={'status':status,'checks':results,'check_count':len(results),'passed':sum(c['passed'] for c in results),'error':error,'source_pins':source_pins,'reviewer':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'CPU_default_children':children,'execution':{'watch_calls':0,'physical_workers_launched':0,'GPU':0,'physics_steps':0,'training_updates':0},'static_lifecycle_review':'The wrapper uses unchanged f416 Windows argv0/samefile/actualPID+PPID+UTC/source/config/output/terminal-ownedPID gate and unchanged40b independenttimer/ownedtree boundedcleanup; ROOT/WORKER and159cIO are changed only in the imported module instance. NativeA0 Popenhard120/worker110. Quality outcomes never qualify OS identity; missing/unknown identity exit1. No actual GPU lifecycle exercised here.','open_static_observations':['Current supervisor pre-watch startup failures after logs creation lack a separate failed_no_child terminal receipt; child not launched and exit1 are fail-closed but explicit metadata is recommended.','External frozen C2 pure path is hashchecked inside controller, but current builder does not explicitly archive that external file in source_pins/sourceZIP; add it for CPU-first dependency closure.']}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':status,'passed':report['passed'],'count':len(results),'report_sha256':sha(out/'report.json'),'error':error},indent=2));raise SystemExit(status!='PASS')
if __name__=='__main__':main()
