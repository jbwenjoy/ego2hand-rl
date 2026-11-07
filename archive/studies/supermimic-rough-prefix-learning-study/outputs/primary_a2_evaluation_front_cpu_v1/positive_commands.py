"""One CPU-only A2 final2000 config build and default Sup2/Entry2 preflight."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parents[1]
os.environ['CUDA_VISIBLE_DEVICES']='-1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    checks={};commands=[];error=None;source_pins={};cfgpin=None
    builder=STUDY/'scripts/build_primary_evaluation_config_cpu_v2.py';sup=STUDY/'scripts/run_rough_evaluation_v2.py';protocol_path=STUDY/'protocols/rough-evaluation-v3.json';train_path=STUDY/'configs/a2-primary_matched-authorized_v1.json';health_path=STUDY/'outputs/health_primary_a2_final_v1/report.json';load_path=STUDY/'outputs/actual_actor_load_primary_a2_final_v1/report.json'
    try:
        snapshot=ROOT/'source';snapshot.mkdir(exist_ok=False)
        protocol=read(protocol_path)
        for key,p in {'builder':builder,'protocol':protocol_path,'commands':Path(__file__),**{k:Path(v['path']) for k,v in protocol['shared_executed_consumer'].items()}}.items():
            source_pins[key]=pin(p);(snapshot/Path(p).name).write_bytes(Path(p).read_bytes())
        if pin(builder)['sha256']!='831acb03c2fc7de41fb7a256a18d3cdc0f66f14c2bb82f1e2da3c32757b4def8' or pin(protocol_path)['sha256']!='5caf3fa4323c85d7ae41a4ed741a7aedf648f853613fa57a1b0f504d2cf3a6a1':raise ValueError('Frozen builder/protocol changed')
        health=read(health_path)
        if health['status']!='PASS_SAVED_ENGINEERING_HEALTH' or not all(v is True for v in health['checks'].values()) or health['iteration']!=2000 or health['final_iteration_scope'] is not True:raise ValueError('Actual final health must be green before build')
        checks['actual_finalhealth_green']=True
        def child(name,args):
            command=[sys.executable,*map(str,args)]
            if '--run' in command:raise AssertionError('CPU preparation cannot watch/evaluate')
            tick=time.perf_counter();result=subprocess.run(command,capture_output=True,text=True,timeout=60,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'},creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            (ROOT/(name+'.stdout.json')).write_text(result.stdout,encoding='utf-8');(ROOT/(name+'.stderr.txt')).write_text(result.stderr,encoding='utf-8');commands.append({'name':name,'command':command,'exit_code':result.returncode,'wall_seconds':time.perf_counter()-tick});checks[name+'_exit0']=result.returncode==0
            if result.returncode:raise RuntimeError(name+' failed '+result.stderr)
            return json.loads(result.stdout)
        cfgpin=child('builder',[builder,'--training-config',train_path,'--health',health_path,'--actor-load',load_path,'--version','v1'])
        config=read(cfgpin['path']);checks['actual_Gfalse_A2final_selected']=config['GPU_run_authorized'] is False and config['method']=='A2' and config['seed']==0 and config['checkpoint_iteration']==2000 and config['checkpoint']['sha256']=='6cb4659e34e02b230b1e6449b186fb99817f6e0c3e732c453f6946d883363210' and cfgpin==pin(cfgpin['path'])
        proof=child('supervisor_default',[sup,'--config',cfgpin['path']])
        checks['actual_default_Sup2_Entry2_CPU_only']=proof['status']=='CPU_INPUT_SOURCE_PREFLIGHT_ONLY' and proof['method']=='A2' and proof['checkpoint']==config['checkpoint'] and proof['Torch_imported'] is False and proof['Warp_imported'] is False and proof['GPU_started'] is False and proof['physics_steps']==0
        shared=protocol['shared_executed_consumer']
        checks['shared_six_consumer_declarations_archived']=len(shared)==6 and all(config['source_pins']['protocol_shared_consumer__'+k]==v and Path(v['path']).resolve()==(STUDY/k).resolve() and pin(v['path'])['sha256']==v['sha256'] for k,v in shared.items())
        checks['all_dependency_paths_and_SHAs_current']=all(Path(v['path']).is_absolute() and pin(v['path'])['sha256']==v['sha256'] for v in config['source_pins'].values())
        train=read(train_path);checks['training_protocol4_alias_exact']=config['source_pins']['training_protocol']==train['protocol']==protocol['training_protocol']
        checks['original_cadence_budget_criteria']=config['worlds']==1 and config['checkpoint_iteration']==2000 and config['hard_process_seconds']==protocol['hard_process_seconds']==120 and config['soft_worker_seconds']==protocol['soft_worker_seconds']==110 and config['automatic_retry'] is False and config['compile_fallback'] is False and config['evaluation_gates']==protocol['evaluation_gates']
        checks['no_physical_output_or_watchdog_created']=not Path(config['output']).exists() and not Path(config['output']+'_watchdog').exists()
        checks['all_frozen_snapshot_sources_unchanged']=all(pin(v['path'])==v for v in source_pins.values())
        checks['no_Torch_Warp_import_or_watch']=not any(k in sys.modules for k in ('torch','warp','mujoco_warp')) and all('--run' not in c['command'] for c in commands)
    except BaseException as exc:error={'error':repr(exc),'traceback':traceback.format_exc()};checks['campaign_complete']=False
    report={'schema':'primary_A2_final_evaluation_positive_CPU_preparation:1','status':'PASS_CPU_ONLY' if all(checks.values()) else 'FAIL_OR_INCOMPLETE','checks':checks,'passed':sum(checks.values()),'total':len(checks),'commands':commands,'source_pins':source_pins,'config':cfgpin,'health':pin(health_path),'actual_actor_load':pin(load_path),'error':error,'GPU_run_authorized':False,'watch_called':False,'Env_or_Runner_constructed':False,'physics_steps':0,'scope':'Actual frozen builder once and Sup2 default once (internally Entry2); positive current-source/cadence/budget checks only. No 32-case replay, source changes or physical evaluation.'}
    save(ROOT/'report.json',report);print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'error':error},indent=2));raise SystemExit(report['status']!='PASS_CPU_ONLY')
if __name__=='__main__':main()
