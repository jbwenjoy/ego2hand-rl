"""Actual CPU-only primary builder/entry/supervisor and meaningful rejections.

All child commands omit --run. No watch, trainer, device, Env or physics call.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ['CUDA_VISIBLE_DEVICES']='-1'
PROTOCOL_SHA='02d266c007545e81469c3c9b1ebcd8192d2bfce11726cda670597fd12cb24a2a'
FROZEN={'train_rough_prefix_pen_v5.py':'38cf260b7818fc6eb065b20ca1402a4dd93040f515862c3d275dfb471958466d','run_rough_prefix_v6.py':'6a2525fc03076d5e97c2d301280a2af5b8a0953b6cada785ede7dc41f0ad08c6','build_primary_config_cpu.py':'9ac128e7c18e625b63d5a1e5792a6f24acf5be04bdf5ffc6510e5293b862e4f8'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);snap=out/'source';snap.mkdir();commands=[];checks=[];errors=[];config_pins={};source_pins={};started=time.perf_counter()
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
    def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    def save(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def check(name,ok,detail=None):checks.append({'name':name,'passed':bool(ok),**({'detail':detail} if detail is not None else {})})
    try:
        for name,expected in FROZEN.items():
            path=ROOT/'scripts'/name
            if sha(path)!=expected:raise ValueError('Approved candidate source changed '+name)
            (snap/name).write_bytes(path.read_bytes());source_pins[name]=pin(path)
        (snap/Path(__file__).name).write_bytes(Path(__file__).read_bytes());source_pins['checker']=pin(__file__)
        protocol_path=ROOT/'protocols/rough-prefix-v4.json'
        if sha(protocol_path)!=PROTOCOL_SHA:raise ValueError('Root protocol4 differs')
        (snap/protocol_path.name).write_bytes(protocol_path.read_bytes());source_pins['protocol4']=pin(protocol_path)
        protocol=read(protocol_path)
        def child(name,args):
            command=[sys.executable,*map(str,args)]
            if '--run' in command:raise AssertionError('No GPU/watch command permitted by CPU checker')
            tick=time.perf_counter();p=subprocess.run(command,capture_output=True,text=True,timeout=60,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'},creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            (out/(name+'.stdout.txt')).write_text(p.stdout,encoding='utf-8');(out/(name+'.stderr.txt')).write_text(p.stderr,encoding='utf-8')
            record={'name':name,'command':command,'exit_code':p.returncode,'elapsed_seconds':time.perf_counter()-tick};commands.append(record)
            check(name+'_exit0',p.returncode==0,record)
            if p.returncode:raise RuntimeError('CPU child failed '+name+' '+p.stderr)
            return json.loads(p.stdout)
        for method in ('A1L','P_ONLY','A2'):
            label=method.lower();child('build_'+label,[ROOT/'scripts/build_primary_config_cpu.py','--method',method,'--version','check_v1'])
            config=ROOT/'configs'/(label+'-primary_matched-check_v1.json');config_pins[method]=pin(config);p=read(config)
            check(label+'_Gfalse_fromscratch_seed0',p['GPU_run_authorized'] is False and p['seed']==0 and p['role']=='primary_matched' and p['controller']['resume']=='from_scratch_only' and p['training']==protocol['primary_matched_training'])
            for variant,entry in (('entry','train_rough_prefix_pen_v5.py'),('supervisor','run_rough_prefix_v6.py')):
                proof=child(variant+'_'+label,[ROOT/'scripts'/entry,'--config',config])
                check(variant+'_'+label+'_actual_default_CPU_boundary',proof['status']=='CPU_INPUT_SOURCE_PREFLIGHT_ONLY' and proof['method']==method and proof['role']=='primary_matched' and proof['Torch_imported'] is False and proof['Warp_imported'] is False and proof['GPU_started'] is False and proof['physics_steps']==0)
                save(out/(variant+'_'+label+'_proof.json'),proof)
            check(label+'_no_formal_output_created',not Path(p['output']).exists() and not Path(p['output']+'_watchdog').exists())
        # Import the frozen Entry5 only after source snapshots and actual CLI
        # defaults. These direct preflight calls cannot reach its main/run path.
        spec=importlib.util.spec_from_file_location('_primary_entry_CPU_contract',ROOT/'scripts/train_rough_prefix_pen_v5.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        base=read(config_pins['A1L']['path']);negative_dir=out/'negative_inputs';negative_dir.mkdir()
        def reject(name,change,protocol_change=None):
            config=copy.deepcopy(base)
            if protocol_change is not None:
                revised=copy.deepcopy(protocol);protocol_change(revised);path=negative_dir/(name+'_protocol.json');save(path,revised);value=pin(path);config['protocol']=value;config['source_pins']['protocol']=value
            change(config);path=negative_dir/(name+'_config.json');save(path,config)
            try:entry.preflight(path)
            except (ValueError,KeyError,AssertionError,TypeError) as exc:check('reject_'+name,True,repr(exc))
            else:check('reject_'+name,False)
        reject('iterations1999',lambda p:p['training'].update(max_iterations=1999))
        reject('warmup63',lambda p:p['training'].update(warmup_steps=63))
        reject('wm_only0',lambda p:p['training'].update(wm_only_iters=0))
        reject('log_interval10',lambda p:p['training'].update(log_interval=10))
        reject('utd_bool',lambda p:p['training'].update(utd=True))
        reject('seed1',lambda p:p.update(seed=1))
        reject('seed_bool',lambda p:p.update(seed=False))
        reject('wrong_role',lambda p:p.update(role='engineering_smoke'))
        reject('wrong_primary_role_spelling',lambda p:p.update(role='primary_matched_arm'))
        reject('warmstart_argument',lambda p:p['training'].update(model_checkpoint='old_checkpoint.pt'))
        reject('old_protocol3',lambda p:(p.update(protocol=pin(ROOT/'protocols/rough-prefix-v3.json')),p['source_pins'].update(protocol=pin(ROOT/'protocols/rough-prefix-v3.json'))))
        reject('revision3_primary_claim',lambda p:None,lambda q:q.update(protocol_revision=3))
        reject('missing_A2_prerequisite',lambda p:None,lambda q:q['primary_matched_prerequisites'].pop('A2'))
        reject('wrong_smoke_checkpoint',lambda p:None,lambda q:q['primary_matched_prerequisites']['A1L'].update(smoke_checkpoint=q['primary_matched_prerequisites']['P_ONLY']['smoke_checkpoint']))
        reject('missing_saved_audit_pin',lambda p:None,lambda q:q['primary_matched_prerequisites']['A1L'].pop('evaluation_saved_audit'))
        reject('authorization_string',lambda p:p.update(GPU_run_authorized='false'))
        def remove_prior(p):
            selected=next(v for v in read(protocol['primary_matched_prerequisites']['A1L']['evaluation_config']['path'])['source_pins'].values() if Path(v['path']).name=='evaluate_rough_prefix_pen_v2.py')
            p['source_pins']={k:v for k,v in p['source_pins'].items() if v!=selected}
        reject('unarchived_prior_execution_dependency',remove_prior)
        # This is an actual positive: all three observed task grades are false,
        # yet their complete engineering prerequisites qualify the primary.
        qualities={method:read(values['evaluation']['path'])['native_policy_pass'] for method,values in protocol['primary_matched_prerequisites'].items()}
        _,_,proof=entry.preflight(Path(config_pins['A1L']['path']))
        check('all_actual_quality_failures_do_not_block_engineering_stage',all(v is False for v in qualities.values()) and proof['status']=='CPU_INPUT_SOURCE_PREFLIGHT_ONLY',qualities)
        check('no_new_source_modified_after_snapshot',all(sha(ROOT/'scripts'/name)==expected for name,expected in FROZEN.items()))
        check('Root_protocol4_unchanged',sha(protocol_path)==PROTOCOL_SHA)
        check('no_Torch_Warp_import_no_watch',not any(name in sys.modules for name in ('torch','warp','mujoco_warp')) and all('--run' not in row['command'] for row in commands))
    except BaseException as exc:
        errors.append({'error':repr(exc),'traceback':traceback.format_exc()});check('CPU_campaign_completed',False,errors[-1])
    report={'schema':'rough_prefix_primary_front_CPU_contract:1','status':'PASS_CPU_ONLY' if all(c['passed'] for c in checks) and not errors else 'FAIL_OR_INCOMPLETE','check_count':len(checks),'passed':sum(c['passed'] for c in checks),'checks':checks,'source_pins':source_pins,'config_pins':config_pins,'actual_direct_child_commands':commands,'error':errors,'GPU_run_authorized':False,'watch_called':False,'Env_or_Runner_constructed':False,'physics_steps':0,'Torch_imported':'torch' in sys.modules,'Warp_imported':'warp' in sys.modules,'wall_seconds':time.perf_counter()-started,'scope':'Actual three builder/entry/supervisor default CPU paths and primary metadata negative gates. No authorized configs, formal outputs, watchdog invocation, device or training.'}
    save(out/'report.json',report);print(json.dumps({'status':report['status'],'checks':report['check_count'],'passed':report['passed'],'failed':[c for c in checks if not c['passed']]},indent=2));raise SystemExit(report['status']!='PASS_CPU_ONLY')


if __name__=='__main__':main()
