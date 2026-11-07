"""CPU-only actual primary2000 evaluation builder/defaults and narrow negatives."""
from __future__ import annotations
import argparse,copy,difflib,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));os.environ['CUDA_VISIBLE_DEVICES']='-1'
P3_SHA='5caf3fa4323c85d7ae41a4ed741a7aedf648f853613fa57a1b0f504d2cf3a6a1'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);snap=out/'source';snap.mkdir();checks=[];commands=[];error=None;started=time.perf_counter()
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
    def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    def save(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def check(name,value,detail=None):checks.append({'name':name,'passed':bool(value),**({'detail':detail} if detail is not None else {})})
    builder=ROOT/'scripts/build_primary_evaluation_config_cpu_v2.py';entry=ROOT/'scripts/evaluate_rough_prefix_pen_v2.py';supervisor=ROOT/'scripts/run_rough_evaluation_v2.py';protocol=ROOT/'protocols/rough-evaluation-v3.json';train=ROOT/'configs/a1l-primary_matched-authorized_v1.json';health=ROOT/'outputs/health_primary_a1l_final_v1/report.json';load=ROOT/'outputs/actual_actor_load_primary_a1l_final_v1/report.json';pins={};config_pin=None
    try:
        if sha(protocol)!=P3_SHA:raise ValueError('Root final2000 evaluation protocol differs')
        for p in (builder,entry,supervisor,ROOT/'src/rough_prefix/evaluation_v2.py',ROOT/'src/rough_prefix/policy_probe.py',ROOT/'src/rough_prefix/checkpoint_v2.py',Path(__file__),protocol):
            (snap/p.name).write_bytes(p.read_bytes());pins[p.name]=pin(p)
        parent_builder=ROOT/'scripts/build_primary_evaluation_config_cpu.py'
        (out/'builder_v1_to_v2.diff').write_text(''.join(difflib.unified_diff(parent_builder.read_text().splitlines(True),builder.read_text().splitlines(True),fromfile='executed_builder_v1',tofile='builder_v2')),encoding='utf-8')
        def child(name,args):
            command=[sys.executable,*map(str,args)]
            if '--run' in command:raise AssertionError('No watch/GPU command allowed')
            tick=time.perf_counter();result=subprocess.run(command,capture_output=True,text=True,timeout=60,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'},creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            (out/(name+'.stdout.txt')).write_text(result.stdout,encoding='utf-8');(out/(name+'.stderr.txt')).write_text(result.stderr,encoding='utf-8');record={'name':name,'command':command,'exit_code':result.returncode,'wall_seconds':time.perf_counter()-tick};commands.append(record);check(name+'_actual_exit0',result.returncode==0,record)
            if result.returncode:raise RuntimeError(name+' '+result.stderr)
            return json.loads(result.stdout)
        built=child('builder',[builder,'--training-config',train,'--health',health,'--actor-load',load,'--protocol',protocol,'--version','check_v2']);config=Path(built['path']);config_pin=pin(config);p=read(config)
        check('actual_builder_Gfalse_primary_final2000',p['GPU_run_authorized'] is False and p['checkpoint_iteration']==2000 and p['training_config']==pin(train) and p['checkpoint']['sha256']=='5a3f48d6cbf24e0a177f57e21ed6ce83992ac01bbc3af273653633bae2cea59c')
        for name,file in (('entry',entry),('supervisor',supervisor)):
            proof=child(name,[file,'--config',config]);save(out/(name+'_proof.json'),proof);check(name+'_default_CPU_boundary',proof['status']=='CPU_INPUT_SOURCE_PREFLIGHT_ONLY' and proof['checkpoint']==p['checkpoint'] and proof['Torch_imported'] is False and proof['Warp_imported'] is False and proof['GPU_started'] is False and proof['physics_steps']==0)
        spec=importlib.util.spec_from_file_location('_primary_eval_builder_CPU_contract',builder);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        neg=out/'negative_inputs';neg.mkdir();base_train=read(train);base_health=read(health)
        def reject(name,train_change=None,health_path=None,load_path=None,protocol_change=None,load_change=None):
            t,h,l,pr=train,health_path or health,load_path or load,protocol
            if train_change is not None:
                altered=copy.deepcopy(base_train);train_change(altered);t=neg/(name+'_training.json');save(t,altered)
                aligned=copy.deepcopy(base_health);aligned['config']=pin(t);h=neg/(name+'_health.json');save(h,aligned)
            if protocol_change is not None:
                altered=read(protocol);protocol_change(altered);pr=neg/(name+'_protocol.json');save(pr,altered)
            if load_change is not None:
                altered=read(load);load_change(altered);l=neg/(name+'_load.json');save(l,altered)
            try:module.build_config(t,h,l,pr,'negative_'+name)
            except (ValueError,KeyError,TypeError,FileNotFoundError) as exc:check('reject_'+name,True,repr(exc))
            else:check('reject_'+name,False)
        reject('nonprimary_role',train_change=lambda p:p.update(role='engineering_smoke'))
        reject('not_final2000_budget',train_change=lambda p:p['training'].update(max_iterations=1999))
        reject('model100_health',health_path=ROOT/'outputs/health_primary_a1l_100_v1/report.json')
        reject('wrong_health_checkpoint',health_path=ROOT/'outputs/health_smoke_a1l_v2_checker_v2/report.json')
        reject('wrong_load_checkpoint',load_path=ROOT/'outputs/actual_actor_load_a1l_v2/report.json')
        reject('wrong_receipt_checkpoint',load_change=lambda p:p['receipt']['checkpoint'].update(sha256='0'*64))
        reject('threshold_change',protocol_change=lambda p:p['evaluation_gates'].update(directed_shaft_error_max_deg=21.))
        reject('old10_protocol_selection',protocol_change=lambda p:p.update(checkpoint_iterations=[10]))
        reject('float2000_selection',protocol_change=lambda p:p.update(checkpoint_iterations=[2000.]))
        def bad_source(p):next(iter(p['source_pins'].values())).update(sha256='0'*64)
        reject('source_pin_drift',train_change=bad_source)
        def wrong_training_protocol(p):
            value=pin(ROOT/'protocols/rough-prefix-v3.json');p['protocol']=value;p['source_pins']['protocol']=value
        reject('wrong_training_protocol',train_change=wrong_training_protocol)
        reject('declared_eval_revision2',protocol_change=lambda p:p.update(protocol_revision=2))
        reject('declared_eval_revision_float',protocol_change=lambda p:p.update(protocol_revision=3.))
        reject('declared_eval_nonprimary_training_role',protocol_change=lambda p:p.update(training_role='engineering_smoke'))
        reject('declared_eval_wrong_training_protocol',protocol_change=lambda p:p.update(training_protocol=pin(ROOT/'protocols/rough-prefix-v3.json')))
        reject('missing_shared_consumer',protocol_change=lambda p:p['shared_executed_consumer'].pop('src/rough_prefix/policy_probe.py'))
        reject('extra_shared_consumer',protocol_change=lambda p:p['shared_executed_consumer'].update(extra=pin(ROOT/'src/rough_prefix/probe.py')))
        reject('shared_consumer_wrong_path',protocol_change=lambda p:p['shared_executed_consumer'].update({'src/rough_prefix/policy_probe.py':pin(ROOT/'src/rough_prefix/probe.py')}))
        reject('shared_consumer_wrong_SHA',protocol_change=lambda p:p['shared_executed_consumer']['src/rough_prefix/policy_probe.py'].update(sha256='0'*64))
        check('all_six_predeclared_consumers_archived_exact',{key.removeprefix('protocol_shared_consumer__'):value for key,value in p['source_pins'].items() if key.startswith('protocol_shared_consumer__')}==read(protocol)['shared_executed_consumer'])
        check('executed_v1_builder_preserved',sha(parent_builder)=='7ffac77aafff8b91a0ce8896c1a84ff313048966ad98b01764b1b9a981a897ff')
        check('trained_protocol4_alias_preserved',p['source_pins']['training_protocol']==base_train['protocol'])
        check('selected_health_and_load_remain_actualfinal',read(health)['final_iteration_scope'] is True and read(health)['iteration']==2000 and read(load)['receipt']['iteration']==2000)
        check('shared_executed_consumer_sources_unchanged',all(sha(q['path'])==q['sha256'] for name,q in pins.items()))
        check('no_future_physical_or_watchdog_output_created',not Path(p['output']).exists() and not Path(p['output']+'_watchdog').exists())
        check('no_Torch_Warp_import_or_watch',not any(k in sys.modules for k in ('torch','warp','mujoco_warp')) and all('--run' not in c['command'] for c in commands))
    except BaseException as exc:error={'error':repr(exc),'traceback':traceback.format_exc()};check('CPU_campaign_complete',False,error)
    report={'schema':'rough_primary_final_evaluation_front_CPU_contract:1','status':'PASS_CPU_ONLY' if all(c['passed'] for c in checks) else 'FAIL_OR_INCOMPLETE','check_count':len(checks),'passed':sum(c['passed'] for c in checks),'checks':checks,'source_pins':pins,'config':config_pin,'commands':commands,'error':error,'Torch_imported':'torch' in sys.modules,'Warp_imported':'warp' in sys.modules,'watch_called':False,'Env_or_Runner_constructed':False,'GPU_run_authorized':False,'physics_steps':0,'wall_seconds':time.perf_counter()-started,'scope':'Actual new builder plus unchanged tested Entry2/Sup2 default CPU; metadata guard tests only. No physics/evaluation invocation or authorization.'}
    save(out/'report.json',report);print(json.dumps({'status':report['status'],'checks':report['check_count'],'passed':report['passed'],'failed':[c for c in checks if not c['passed']]},indent=2));raise SystemExit(report['status']!='PASS_CPU_ONLY')


if __name__=='__main__':main()
