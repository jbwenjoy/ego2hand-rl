"""Default-only evaluation preflight and meaningful fail-closed metadata cases."""
from pathlib import Path
import ast, copy, hashlib, importlib.util, json, os, subprocess, sys, time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/evaluation_front_independent_cpu_v1'
PYTHON=Path(r'D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe')
LIFE=Path(r'D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py')
def pin(p):
    p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
    OUT.mkdir(exist_ok=False);snap=OUT/'sources';snap.mkdir();checks={};runs=[];negatives=[]
    def ck(k,v):checks[k]=bool(v)
    specs=['src/rough_prefix/evaluation.py','src/rough_prefix/policy_probe.py','src/rough_prefix/checkpoint_v2.py','scripts/evaluate_rough_prefix_pen.py','scripts/run_rough_evaluation.py','scripts/build_evaluation_config_cpu.py','protocols/rough-evaluation-v1.json','scripts/check_evaluation_front_independent_cpu.py']
    sourcepins={k:pin(ROOT/k) for k in specs}
    for k in specs:(snap/k.replace('/','__')).write_bytes((ROOT/k).read_bytes())
    spec=importlib.util.spec_from_file_location('reviewed_eval_CPU_cleanup',LIFE);life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
    ck('reviewed_CPU_failure_cleanup_source',pin(LIFE)['sha256']=='40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa')
    def child(label,args):
        start=time.monotonic();p=subprocess.Popen([str(PYTHON),*map(str,args)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'},creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:out,err=p.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            cleanup=life.stop_owned(p);write(OUT/(label+'_timeout.json'),cleanup);raise RuntimeError('CPU preflight timeout, no GPU or retry')
        (OUT/(label+'_stdout.txt')).write_bytes(out);(OUT/(label+'_stderr.txt')).write_bytes(err)
        record={'label':label,'args':list(map(str,args)),'returncode':p.returncode,'elapsed_seconds':time.monotonic()-start};runs.append(record)
        if p.returncode:raise RuntimeError('CPU child failed '+label+': '+err.decode(errors='replace'))
        return ast.literal_eval(out.decode()) if label=='build_CPU_config' else json.loads(out)
    cfgpath=ROOT/'configs/a1l-policy-evaluation-check-v1.json'
    built=child('build_CPU_config',[ROOT/'scripts/build_evaluation_config_cpu.py','--method','A1L','--version','check-v1','--health',ROOT/'outputs/health_smoke_a1l_v2_checker_v2/report.json','--actor-load',ROOT/'outputs/actual_actor_load_a1l_v2/report.json'])
    cfg=read(cfgpath);ck('builder_default_GPU_false',cfg['GPU_run_authorized'] is False and built==pin(cfgpath))
    direct=child('actual_default_CLI',[ROOT/'scripts/evaluate_rough_prefix_pen.py','--config',cfgpath])
    supervised=child('actual_default_supervisor',[ROOT/'scripts/run_rough_evaluation.py','--config',cfgpath])
    for label,p in [('direct',direct),('supervisor',supervised)]:
        ck(label+'_actual_default_no_framework_or_physics',p['status']=='CPU_INPUT_SOURCE_PREFLIGHT_ONLY' and p['config_sha256']==pin(cfgpath)['sha256'] and p['Torch_imported'] is False and p['Warp_imported'] is False and p['GPU_started'] is False and p['physics_steps']==0)
    ck('no_worker_or_watchdog_output_created',not Path(cfg['output']).exists() and not Path(cfg['output']+'_watchdog').exists())
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
    s=importlib.util.spec_from_file_location('eval_front_CPU_only',ROOT/'scripts/evaluate_rough_prefix_pen.py');front=importlib.util.module_from_spec(s);s.loader.exec_module(front)
    def negative(name,mutate):
        c=copy.deepcopy(cfg);mutate(c);p=OUT/(name+'.json');write(p,c)
        try:front.preflight(p)
        except (ValueError,KeyError,FileNotFoundError,AssertionError) as exc:negative={'name':name,'rejected':True,'reason':repr(exc)}
        else:negative={'name':name,'rejected':False}
        negatives.append(negative);ck('reject_'+name,negative['rejected'])
    negative('legacy_schema',lambda c:c.update(schema='rough_prefix_native_run:1'))
    negative('wrong_role',lambda c:c.update(role='engineering_smoke'))
    negative('bool_world_not_int1',lambda c:c.update(worlds=True))
    negative('seed_outside_protocol',lambda c:c.update(seed=1))
    negative('nonfinal_iteration9',lambda c:c.update(checkpoint_iteration=9))
    negative('different_wall_budget',lambda c:c.update(hard_process_seconds=121))
    negative('changed_numeric_gate',lambda c:c['evaluation_gates'].update(directed_shaft_error_max_deg=21.))
    negative('source_bytes_drift',lambda c:c['source_pins']['protocol'].update(sha256='0'*64))
    negative('missing_training_protocol',lambda c:c['source_pins'].pop('training_protocol'))
    negative('missing_policy_dependency',lambda c:c['source_pins'].pop('evaluation__src/rough_prefix/policy_probe.py'))
    model10=pin(ROOT/'outputs/smoke_a1l_v2/train/model_10.pt')
    negative('iteration_checkpoint_instead_of_final',lambda c:c.update(checkpoint=model10))
    def receipt_bad(c,key,kind):
        receipt=read(c[key]['path'])
        if kind=='FAIL':
            receipt['status']='FAIL_OR_MISSING_EVIDENCE'
        else:
            def replace(v):
                if v==cfg['checkpoint']:return model10
                if isinstance(v,dict):return {k:replace(x) for k,x in v.items()}
                if isinstance(v,list):return [replace(x) for x in v]
                return v
            receipt=replace(receipt)
        p=OUT/(key+'_'+kind+'_receipt.json');write(p,receipt);c[key]=pin(p)
    negative('health_not_pass',lambda c:receipt_bad(c,'health_report','FAIL'))
    negative('health_wrong_checkpoint_bytes',lambda c:receipt_bad(c,'health_report','wrong'))
    negative('actor_load_wrong_checkpoint_bytes',lambda c:receipt_bad(c,'actor_load_report','wrong'))
    negative('unowned_output_path',lambda c:c.update(output=str(ROOT.parent/'outside_eval_output')))
    proto=read(cfg['protocol']['path']);train=read(cfg['training_config']['path'])
    ck('protocol_and_archive_training_protocol_bound',cfg['source_pins']['protocol']==cfg['protocol'] and cfg['source_pins']['training_protocol']==train['protocol'] and proto['input_bundle']==cfg['input_bundle'] and proto['attempts_per_arm']==1 and proto['control_intervals']==15 and proto['solver_substeps']==60 and proto['postinterval_forward_calls']==15)
    ck('all_config_sources_current',all(pin(p['path'])['sha256']==p['sha256'] for p in cfg['source_pins'].values()))
    old=(ROOT/'scripts/run_rough_prefix_v5.py').read_text().replace("ENTRY=ROOT/'scripts/train_rough_prefix_pen_v4.py'","ENTRY=ROOT/'scripts/evaluate_rough_prefix_pen.py'").replace('(train_rough_prefix_pen(?:_v[0-9]+)?|','(evaluate_rough_prefix_pen|train_rough_prefix_pen(?:_v[0-9]+)?|')
    ck('supervisor_only_eval_route_and_worker_reject_added',old==(ROOT/'scripts/run_rough_evaluation.py').read_text())
    e=(ROOT/'src/rough_prefix/evaluation.py').read_text();p=(ROOT/'src/rough_prefix/policy_probe.py').read_text()
    ck('orchestrator_preserves_trained_descriptor_no_runner_construct',"descriptor = state['rough_controller']['root_descriptor']" in e and "trained_controller" in e and 'RecordedRunner(' not in e and 'DexSuperMimicRunner(' not in e)
    ck('actor_no_normalizer_or_observation_extra_update', 'actor(_policy_obs(obs))' in p and 'norm.update(' not in p and 'observation_manager.compute(' not in p and 'env.reset(' not in p)
    ck('independent_EMA_control_and_full15intervals', 'for k in range(15):' in p and "prediction=expected_control(" in p and "sample(len(states))" in p and "forward_calls!=15" in p and "flags=env.sim.wp_data.overflow.numpy().copy()" in p)
    ck('reviewed_sources_unchanged',all(pin(v['path'])['sha256']==v['sha256'] for v in sourcepins.values()))
    ck('independent_CPU_checker_no_framework_import',not any(k.split('.')[0] in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules))
    report={'schema':'rough_policy_evaluation_front_independent_CPU:1','status':'PASS_CPU_FRONT_AND_STATIC_INTERFACE' if all(checks.values()) else 'BLOCKING_OR_FAILED','passed':sum(checks.values()),'total':len(checks),'checks':checks,'negative_cases':negatives,'actual_CPU_subprocesses':runs,'source_pins':sourcepins,'config':pin(cfgpath),'prerequisite_A1L_health':cfg['health_report'],'actual_actor_load':cfg['actor_load_report'],'static_review':['Same trained controller descriptor/signature, not re-labeled new evaluation root.','One Manager, nominal model then actual8 copy/setconst/readback then analytic nonzero reset.','No Runner/optimizer/model update; Actor155->20 eval mode, no noise/norm update.','15 real action intervals with four original steps each and one original postinterval forward; quality failure scored after full finite horizon.','Startup ctrl baseline0 is preserved; independent expected EMA and C2 compare cache and actual.','Contacts/actuator-force saved at completed solver preintegration; q/qd at postintegration before Manager forward.','Final sticky once after step60, partial/UNKNOWN unqualified.','Rates include declared startup target jump plus14 updates under original15rad/s threshold.','Old training protocol independently archived after source-closure finding corrected.'],'scope':'Default CLI/supervisor CPU preflight and metadata negative cases only. Static source/method review is not real learned Actor closed-loop/physics/capacity qualification. No GPU launch, watch, Env, Runner, collision/FK/physics or optimizer.','GPU_used':False,'physics_steps':0,'optimizer_steps':0,'experimental_source_modified':False}
    write(OUT/'report.json',report);print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(OUT/'report.json')}));return 0 if all(checks.values()) else 1
if __name__=='__main__':raise SystemExit(main())
