"""Default CPU input/source preflight; explicit supervised native A0/training."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,sys,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
STARTED=time.perf_counter();ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from rough_prefix.io import read,checked,pin,sha,jsonable
from rough_prefix.bundle_v2 import load_bundle
from rough_prefix.metadata import write
def utc():return datetime.now(timezone.utc).isoformat()
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
