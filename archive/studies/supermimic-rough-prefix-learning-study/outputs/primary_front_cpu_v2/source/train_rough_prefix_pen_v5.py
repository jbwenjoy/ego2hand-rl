"""Default CPU input/source preflight; explicit supervised native A0/training."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,sys,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
STARTED=time.perf_counter();ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from rough_prefix.io import read,checked,pin,sha,jsonable,canonical
from rough_prefix.bundle_v2 import load_bundle
from rough_prefix.metadata import write
def utc():return datetime.now(timezone.utc).isoformat()

PRIMARY_TRAINING={'num_envs':128,'max_iterations':2000,'buffer_steps':256,'warmup_steps':64,'wm_only_iters':200,'wm_window':8,'wm_batches':2,'wm_batch':256,'wm_hidden':[512,512,512],'pi_batch':128,'horizon':8,'max_horizon':8,'critic_epochs':2,'critic_minibatches':2,'holdout_fraction':.1,'save_interval':100,'log_interval':1,'amp':True,'compile':False,'compile_policy':False,'compile_policy_dynamic':False,'action_noise':.1,'utd':.4}
PRIMARY_PREREQUISITE_FIELDS={'smoke_config','smoke_checkpoint','smoke_health','smoke_identity','evaluation_config','evaluation','evaluation_capacity','evaluation_identity','evaluation_saved_audit'}

def verify_primary_prerequisites(protocol,config):
    """All three arms qualify on engineering evidence, never task quality."""
    table=protocol['primary_matched_prerequisites']
    if type(table) is not dict or set(table)!={'A1L','P_ONLY','A2'}:raise ValueError('All three fixed engineering arms are prerequisites')
    def included(value):
        if not any(v==value for v in config['source_pins'].values()):raise ValueError('Prerequisite/dependency missing from source archive closure')
    def bool_checks(report,key='checks'):
        values=report.get(key)
        return type(values) is dict and bool(values) and all(v is True for v in values.values())
    def fields_equal(value,expected):
        return canonical({key:value.get(key) for key in expected})==canonical(expected)
    for method,spec in table.items():
        if type(spec) is not dict or set(spec)!=PRIMARY_PREREQUISITE_FIELDS:raise ValueError('Incomplete primary prerequisite fields')
        for value in spec.values():checked(value);included(value)
        smoke=read(checked(spec['smoke_config']));health=read(checked(spec['smoke_health']));smoke_identity=read(checked(spec['smoke_identity']))
        evaluation_config=read(checked(spec['evaluation_config']));evaluation=read(checked(spec['evaluation']));capacity=read(checked(spec['evaluation_capacity']));evaluation_identity=read(checked(spec['evaluation_identity']));audit=read(checked(spec['evaluation_saved_audit']))
        if smoke['schema']!='rough_prefix_native_run:1' or smoke['role']!='engineering_smoke' or smoke['method']!=method or type(smoke['seed']) is not int or smoke['seed']!=0 or smoke['input_bundle']!=config['input_bundle']:raise ValueError('Prerequisite smoke arm/source mismatch')
        if health.get('schema')!='rough104_checkpoint_CPU_health:1' or health.get('status')!='PASS_SAVED_ENGINEERING_HEALTH' or not bool_checks(health) or health.get('config')!=spec['smoke_config'] or health.get('checkpoint')!=spec['smoke_checkpoint'] or not fields_equal(health,{'method':method,'seed':0,'iteration':10}):raise ValueError('Smoke checkpoint engineering health prerequisite failed')
        smoke_output=Path(smoke['output']).resolve()
        if Path(spec['smoke_checkpoint']['path']).resolve()!=smoke_output/'train/model_final.pt' or Path(spec['smoke_identity']['path']).resolve()!=smoke_output.with_name(smoke_output.name+'_watchdog')/'engineering_identity.json':raise ValueError('Smoke proof paths do not match actual run')
        if smoke_identity.get('engineering_identity_qualified') is not True or not bool_checks(smoke_identity):raise ValueError('Smoke process identity unqualified')
        if evaluation_config['schema']!='rough_prefix_policy_evaluation_run:1' or evaluation_config['role']!='native_policy_eval' or not fields_equal(evaluation_config,{'method':method,'seed':0}) or evaluation_config['checkpoint']!=spec['smoke_checkpoint'] or evaluation_config['training_config']!=spec['smoke_config'] or evaluation_config['health_report']!=spec['smoke_health'] or evaluation_config['input_bundle']!=config['input_bundle']:raise ValueError('Evaluation is not the same declared smoke checkpoint/source')
        eval_output=Path(evaluation_config['output']).resolve()
        expected_paths={'evaluation':eval_output/'evaluation.json','evaluation_capacity':eval_output/'capacity_audit.json','evaluation_identity':eval_output.with_name(eval_output.name+'_watchdog')/'engineering_identity.json'}
        if any(Path(spec[key]['path']).resolve()!=path for key,path in expected_paths.items()):raise ValueError('Evaluation proof paths do not match actual run')
        if not fields_equal(evaluation,{'complete':True,'physical_steps':60,'control_intervals':15,'native_postinterval_forward_calls':15,'optimizer_updates':0,'resets_after_start':0}) or evaluation.get('gates',{}).get('capacity') is not True or evaluation.get('gates',{}).get('all_actual_u_exact') is not True:raise ValueError('Evaluation engineering/full-horizon prerequisite failed')
        if not fields_equal(capacity,{'status':'OK','flags_by_world':[0],'read_count':1,'read_after_step':60,'resets_during_probe':0}):raise ValueError('Evaluation capacity prerequisite failed')
        if evaluation_identity.get('engineering_identity_qualified') is not True or not bool_checks(evaluation_identity):raise ValueError('Evaluation process identity unqualified')
        if audit.get('schema')!='rough_native_policy_saved_CPU_audit:1' or audit.get('engineering_qualified') is not True or not bool_checks(audit) or type(audit.get('passed')) is not int or type(audit.get('total')) is not int or audit['passed']!=audit['total'] or audit.get('passed')!=len(audit['checks']) or audit.get('config')!=spec['evaluation_config'] or audit.get('checkpoint')!=spec['smoke_checkpoint'] or audit.get('evaluation')!=spec['evaluation']:raise ValueError('Independent evaluation saved audit prerequisite failed')
        # Archive all prior executed dependencies, not just reports pointing to
        # unarchived mutable files. Equal file pins may be content-deduplicated.
        for prior in (smoke,evaluation_config):
            for value in prior['source_pins'].values():checked(value);included(value)
    return {'all_fixed_arms':['A1L','P_ONLY','A2'],'engineering_prerequisites_qualified':True,'task_quality_used_as_gate':False}

def preflight(path):
    p=read(path)
    allowed={'schema','role','method','seed','input_bundle','output','training','contact_binding','source_pins','protocol','GPU_run_authorized','training_authorized','hard_process_seconds','soft_worker_seconds','automatic_retry','compile_fallback','controller','evaluation_gates'}
    if set(p)!=allowed or p['schema']!='rough_prefix_native_run:1' or p['method'] not in ('A0','A1L','P_ONLY','A2') or p['role'] not in ('native_a0','engineering_smoke','primary_matched'):raise ValueError('New rough-prefix schema/role/arm required')
    if type(p['seed']) is not int or p['seed']<0 or p['automatic_retry'] is not False or p['compile_fallback'] is not False:raise ValueError('Finite seeded no-retry/no-fallback protocol required')
    if p['controller']!={'kind':'ROUGH_PREFIX_BASELINE_EMA_C2','format_version':104,'margin_rad':.020,'resume':'from_scratch_only'}:raise ValueError('Wrong new control/checkpoint contract')
    if p['role']=='native_a0':
        if p['method']!='A0' or p['training']!={'num_envs':1} or p['contact_binding'] is not None or p['training_authorized'] is not False or p['hard_process_seconds']!=120 or p['soft_worker_seconds']!=110:raise ValueError('Only one finite A0 world, no optimizer/training')
    else:
        if p['method']=='A0' or p['training_authorized'] is not True:raise ValueError('Separate training arm/authorization required')
        keys={'num_envs','max_iterations','buffer_steps','warmup_steps','wm_only_iters','wm_window','wm_batches','wm_batch','wm_hidden','pi_batch','horizon','max_horizon','critic_epochs','critic_minibatches','holdout_fraction','save_interval','log_interval','amp','compile','compile_policy','compile_policy_dynamic','action_noise','utd'}
        if set(p['training'])-keys or {'checkpoint','model_checkpoint'}&set(p['training']):raise ValueError('Unsupported/old-checkpoint training arguments')
        if (p['method'] in ('P_ONLY','A2'))!=(p['contact_binding'] is not None):raise ValueError('Prior arm binding mismatch')
        if not 1<=p['soft_worker_seconds']<p['hard_process_seconds']<=7200:raise ValueError('Finite engineering bound required')
    from rough_prefix.probe import GATES
    if p['evaluation_gates']!=GATES:raise ValueError('Original numeric evaluation criteria changed')
    for v in p['source_pins'].values():checked(v)
    for source in (Path(__file__),ROOT/'src/rough_prefix/bundle_v2.py',ROOT/'src/rough_prefix/native_v2.py',ROOT/'src/rough_prefix/training.py',ROOT/'src/rough_prefix/capacity.py',ROOT/'src/rough_prefix/probe.py',ROOT/'src/rough_prefix/runtime_v2.py',ROOT/'src/rough_prefix/controller.py',ROOT/'src/rough_prefix/io.py',ROOT/'src/rough_prefix/metadata.py',ROOT/'vendor/runtime_io.py',ROOT/'vendor/measured_contacts.py'):
        if not any(Path(v['path']).resolve()==source.resolve() and v['sha256']==sha(source) for v in p['source_pins'].values()):raise ValueError('Unfrozen execution dependency '+str(source))
    if p['protocol']!=p['source_pins']['protocol']:raise ValueError('Protocol alias differs')
    protocol=read(checked(p['protocol']))
    if protocol['schema']!='rough_prefix_learning_protocol:1' or protocol['physical_reference_qualified'] is not False or protocol['reference_states']!=16 or protocol['control_intervals']!=15:raise ValueError('Wrong new-mode protocol')
    if p['role'] not in protocol['allowed_roles'] or p['seed'] not in protocol['allowed_seeds']:raise ValueError('Role/seed not authorized by this stage protocol')
    bundle=load_bundle(checked(p['input_bundle']),p['input_bundle']['sha256'])
    if protocol['input_bundle']!=p['input_bundle']:raise ValueError('Protocol input differs')
    if p['role']=='engineering_smoke':
        for key in ('native_A0_evaluation','native_A0_identity'):
            if key not in p['source_pins']:raise ValueError('Native A0 prerequisite proof missing')
        ev=read(checked(p['source_pins']['native_A0_evaluation']));identity=read(checked(p['source_pins']['native_A0_identity']))
        if ev.get('complete') is not True or ev.get('control_intervals')!=15 or ev.get('physical_steps')!=60 or ev.get('gates',{}).get('capacity') is not True or identity.get('engineering_identity_qualified') is not True:raise ValueError('Native A0 engineering/capacity identity unqualified; quality outcome is separate')
        if p['training']!=protocol['engineering_smoke_training']:raise ValueError('Engineering smoke budget differs')
    if p['role']=='primary_matched':
        if type(protocol.get('protocol_revision')) is not int or protocol['protocol_revision']!=4 or protocol['allowed_roles']!=['primary_matched'] or canonical(protocol['allowed_seeds'])!=canonical([0]) or p['seed']!=0:raise ValueError('Primary requires separate revision4, role primary_matched and seed0; earlier protocol3 cannot authorize it')
        if canonical(protocol['primary_matched_training'])!=canonical(PRIMARY_TRAINING) or canonical(p['training'])!=canonical(protocol['primary_matched_training']):raise ValueError('Exact fixed primary budget differs')
        if p['soft_worker_seconds']!=7100 or p['hard_process_seconds']!=7200 or type(p['GPU_run_authorized']) is not bool:raise ValueError('Primary finite process guard/authorization type differs')
        verify_primary_prerequisites(protocol,p)
    if p['contact_binding'] is not None:
        spec=read(checked(p['contact_binding']));checked(spec['binding_manifest'])
        if spec['reference_file']!=bundle['spec']['source_pins']['contact_reference']['path'] or spec['expected_hashes']['reference_npz_sha256']!=bundle['spec']['source_pins']['contact_reference']['sha256'] or spec['expected_hashes']['actual_context_sha256']!=bundle['spec']['context']['sha256']:raise ValueError('Prior/reference/physical context differs')
    output=Path(p['output']).resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('New study-owned output required')
    if any(k in sys.modules for k in ('torch','warp','mujoco_warp')):raise AssertionError('CPU preflight imported GPU frameworks')
    return p,bundle,{'status':'CPU_INPUT_SOURCE_PREFLIGHT_ONLY','config_sha256':sha(path),'entry_sha256':sha(__file__),'role':p['role'],'method':p['method'],'input_signature_sha256':bundle['descriptor']['input_signature_sha256'],'pinned_files':len(p['source_pins']),'Torch_imported':False,'Warp_imported':False,'GPU_started':False,'physics_steps':0}
def freeze_sources(output,config):
    dest=output/'source';dest.mkdir();inventory={}
    with zipfile.ZipFile(dest/'execution_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for i,(key,value) in enumerate(config['source_pins'].items()):
            path=checked(value);z.write(path,f'{i:04d}_{path.name}');inventory[key]=value
    write(dest/'inventory.json',inventory)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path);parser.add_argument('--run',action='store_true');a=parser.parse_args();path=a.config.resolve(strict=True)
    if not a.run:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    p,b,proof=preflight(path)
    if not a.run:print(json.dumps(proof,indent=2));return
    if p['GPU_run_authorized'] is not True:raise ValueError('Separate hash-bound GPU authorization absent')
    output=Path(p['output']).resolve()
    if a.output is None or a.output.resolve()!=output or output.exists():raise ValueError('Explicit exact new output required, no overwrite or retry')
    output.mkdir(parents=True,exist_ok=False)
    runtime={'status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':utc(),'plan_sha256':sha(path),'source_sha256':sha(__file__),'role':p['role'],'method':p['method'],'physics_steps':None,'physics_step_scope':'Training batched substeps derived from actual collector sim_steps; UNKNOWN before terminal counter is available; excludes startup graph capture/forwards','optimizer_updates':0,'automatic_retry':False}
    write(output/'runtime.json',runtime);shutil.copy2(path,output/'config.json');freeze_sources(output,p);write(output/'CPU_preflight.json',proof)
    p={**p,'_config_sha256':sha(path),'_worker_started_perf':STARTED}
    def emit(**updates):runtime.update(updates);runtime['worker_elapsed_seconds']=time.perf_counter()-STARTED;write(output/'runtime.json',runtime)
    try:
        from rough_prefix.training import run_training
        result=run_training(p,b,output,emit)
        write(output/'training_completion.json',{'schema':'rough_prefix_training_completion:1','config_sha256':sha(path),'result':result})
        runtime.update(status='completed_native_A0' if p['role']=='native_a0' else 'completed_rough_training',result_status=result['status'],optimizer_updates=sum(result.get('actual_optimizer_steps',{}).values()))
        if 'sim_steps' in result:
            runtime['collected_sim_transitions']=result['sim_steps']
            runtime['physics_steps']=result['sim_steps']//p['training']['num_envs']*4
            runtime['physical_world_substeps']=result['sim_steps']*4
    except BaseException as exc:
        runtime.update(status='failed_engineering_partial',error=repr(exc));write(output/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'no_retry':True});raise
    finally:
        runtime.update(finished_utc=utc(),wall_seconds=time.perf_counter()-STARTED);write(output/'runtime.json',runtime)
    print(json.dumps(jsonable(result),indent=2))
if __name__=='__main__':main()
