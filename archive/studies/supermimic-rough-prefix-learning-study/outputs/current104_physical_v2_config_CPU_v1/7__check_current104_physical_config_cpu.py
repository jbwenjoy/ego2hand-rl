"""New physical-FD metadata contracts only; never pass --run to a GPU entry."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/current104-physical-FD-v1.json'
ENTRY=ROOT/'scripts/probe_current104_physical_fd.py'
SUP=ROOT/'scripts/run_current104_physical_fd.py'
BUILDER=ROOT/'scripts/build_current104_physical_probe_config_cpu.py'
WATCH=Path('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py')
WATCH_SHA='40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa'
HIDDEN=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0


def pin(path):
    path=Path(path);return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def run_cpu(command,out,label):
    """Retain even a CPU redirector handle and bound owned cleanup on failure."""
    p=subprocess.Popen([str(v) for v in command],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                       creationflags=HIDDEN,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
    started=time.perf_counter()
    try:stdout,stderr=p.communicate(timeout=30)
    except BaseException as error:
        cleanup={'retained_launcher_pid':p.pid,'error':repr(error)}
        try:
            if pin(WATCH)['sha256']!=WATCH_SHA:raise ValueError('Owned CPU cleanup source changed')
            cleanup['owned_cleanup']=module(WATCH,'physical_config_CPU_cleanup').stop_owned(p)
        except BaseException as e:cleanup['cleanup_error']=repr(e)
        try:stdout,stderr=p.communicate(timeout=1)
        except BaseException:
            stdout=stderr='CPU pipes unavailable after bounded cleanup'
            if p.stdout:p.stdout.close()
            if p.stderr:p.stderr.close()
        (out/(label+'_cleanup.json')).write_text(json.dumps(cleanup,indent=2)+'\n',encoding='utf8')
        raise RuntimeError('CPU command interrupted/timed out; no GPU or retry') from error
    record={'command':[str(v) for v in command],'retained_launcher_pid':p.pid,'exit_code':p.returncode,
            'wall_seconds':time.perf_counter()-started,'stdout':stdout,'stderr':stderr,'GPU_authorized':False}
    (out/(label+'.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    if p.returncode:raise ValueError('CPU metadata command failed '+label+': '+stderr+stdout)
    return json.loads(stdout)


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'outputs/current104_physical_config_CPU_v1');args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),BUILDER,ENTRY,SUP,ROOT/'protocols/current104-physical-fd-execution-v2.json',
             ROOT/'src/rough_prefix/physical_fd.py',ROOT/'src/rough_prefix/probe_transaction_v2.py']
    before={str(p):pin(p) for p in sources}
    for i,p in enumerate(sources):(out/(str(i)+'__'+p.name)).write_bytes(p.read_bytes())
    checks={};failure=None
    try:
        if CONFIG.exists():
            # Consume the fixed saved result if Root already generated it; no rewrite.
            cfg=json.loads(CONFIG.read_text(encoding='utf-8-sig'));checks['existing_config_consumed_without_rebuild']=True
        else:
            built=run_cpu([sys.executable,BUILDER],out,'builder_default')
            checks['builder_actual_default_CPU_once']=built==pin(CONFIG)
            cfg=json.loads(CONFIG.read_text(encoding='utf-8-sig'))
        checks['builder_false_no_retry_fixed_output']=cfg['GPU_run_authorized'] is False and cfg['automatic_retry'] is False and cfg['output']==str(ROOT/'outputs/current104_physical_FD_v1')
        result=run_cpu([sys.executable,ENTRY,'--config',CONFIG],out,'entry_default')
        checks['entry_actual_default_no_framework_no_physics']=result.get('Torch_imported') is False and result.get('Warp_imported') is False and result.get('GPU_started') is False and result.get('physics_steps')==0
        result=run_cpu([sys.executable,SUP,'--config',CONFIG],out,'supervisor_default')
        checks['supervisor_actual_default_no_GPU_no_watch']=result.get('Torch_imported') is False and result.get('Warp_imported') is False and result.get('GPU_started') is False and result.get('physics_steps')==0
        entry=module(ENTRY,'current104_physical_entry_CPU_contract')
        def reject(name,edit):
            c=copy.deepcopy(cfg);edit(c);path=out/(name+'.json');path.write_text(json.dumps(c,indent=2)+'\n',encoding='utf8')
            try:entry.preflight(path)
            except (ValueError,TypeError,KeyError,AssertionError,FileNotFoundError) as error:
                checks[name]=True;(out/(name+'_rejection.txt')).write_text(repr(error),encoding='utf8')
            else:checks[name]=False
        reject('unknown_header',lambda c:c.update(extra_robot_q='injected'))
        reject('missing_stage',lambda c:c.pop('stage'))
        reject('wrong_role',lambda c:c.update(role='native_policy_eval'))
        reject('wrong_stage',lambda c:c.update(stage='CPU_consumer_before_GPU'))
        reject('worlds_bool_not_int',lambda c:c.update(worlds=True))
        reject('evaluation_seed_bool_not_int',lambda c:c.update(evaluation_seed=False))
        reject('GPU_flag_string_not_bool',lambda c:c.update(GPU_run_authorized='true'))
        reject('GPU_true_without_Root_authorization',lambda c:c.update(GPU_run_authorized=True))
        reject('hard_budget_not240',lambda c:c.update(hard_process_seconds=241))
        reject('soft_budget_not220',lambda c:c.update(soft_worker_seconds=240))
        reject('retry_enabled',lambda c:c.update(automatic_retry=True))
        reject('fallback_enabled',lambda c:c.update(compile_fallback=True))
        reject('outside_output',lambda c:c.update(output=str(ROOT.parent/'outside_probe')))
        reject('wrong_direction_arrays_alias',lambda c:c['directions'].update(arrays=c['directions']['report']))
        reject('dataset_matrix_reordered',lambda c:c['datasets'].reverse())
        reject('dataset_16interval_injection',lambda c:c['datasets'][0].update(control_transitions=16))
        reject('wrong_final_model_iteration',lambda c:c['models'][0].update(iteration=10))
        reject('source_IO_missing',lambda c:c['source_pins'].pop('runtime_IO'))
        reject('corrupt_entry_source_SHA',lambda c:c['source_pins']['entry'].update(sha256='0'*64))
        def old_parent(c):
            v=pin(ROOT/'proposals/current104_samewarp_physical_fd_v1/protocol.json');c['parent_proposal']=v
        reject('old_V1_parent_forbidden',old_parent)
        def old_protocol(c):
            v=pin(ROOT/'protocols/current104-physical-fd-execution-v1.json');c['protocol']=v;c['source_pins']['protocol']=v
        reject('revision1_missing_FP32terms_forbidden',old_protocol)
        checks['all_snapshotted_source_preserved']=all(pin(p)==v for p,v in before.items())
        checks['no_framework_in_checker_process']=not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp'))
    except BaseException as error:failure={'error':repr(error),'traceback':traceback.format_exc()}
    report={'schema':'current104_physical_config_CPU_contract:1','status':'PASS' if checks and all(checks.values()) and failure is None else 'FAIL',
            'checks':checks,'passed':sum(checks.values()),'total':len(checks),'failure':failure,'source_snapshots':before,
            'config':pin(CONFIG) if CONFIG.exists() else None,'WM_forwards':0,'Env_created':False,'physics_steps':0,'GPU_used':False,
            'scope':'Only new builder/entry/supervisor defaultmetadata and malformedconfig guards; not native/GPU/restore or physicalFD qualification.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'report':pin(out/'report.json')}))
    if report['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
