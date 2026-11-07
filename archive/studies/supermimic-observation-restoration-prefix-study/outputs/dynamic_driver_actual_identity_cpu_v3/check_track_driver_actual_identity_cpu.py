"""One tiny stdlib CPU child: venv redirector/base aliases/CIM/exit identity.

Never reads a future bundle, imports a simulator, acquires a GPU or retries a
child. The in-memory WORKER test seam points only at the saved stdlib fixture.
"""
from __future__ import annotations
import argparse,copy,difflib,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/run_track_prefix_v3.py'
spec=importlib.util.spec_from_file_location('dynamic_prefix_file_identity_driver',SOURCE);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

FIXTURE='''"""Stdlib-only runtime identity fixture, no physical or reference inputs."""
import argparse,hashlib,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
assert a.run
assert json.loads(a.plan.read_text())['schema']=='CPU_identity_fixture_plan:1'
a.output.mkdir(parents=True,exist_ok=False)
r={'status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sys_executable':sys.executable,'base_executable':sys._base_executable,'CPU_fixture_only':True,'physics_steps':0}
(a.output/'runtime.json').write_text(json.dumps(r,indent=2))
time.sleep(2.5)
r['status']='completed_CPU_identity_fixture';r['finished_utc']=datetime.now(timezone.utc).isoformat();r['Torch_imported']='torch' in sys.modules;r['Warp_imported']='warp' in sys.modules or 'mujoco_warp' in sys.modules
(a.output/'runtime.json').write_text(json.dumps(r,indent=2))
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();start=time.perf_counter();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for p in (SOURCE,Path(__file__)):(out/p.name).write_bytes(p.read_bytes())
    fixture=out/'stdlib_identity_worker.py';fixture.write_text(FIXTURE,encoding='utf-8');plan=out/'CPU_fixture_plan.json';plan.write_text(json.dumps({'schema':'CPU_identity_fixture_plan:1','GPU_authorized':False,'future_bundle_read':False}),encoding='utf-8');output=out/'actual_CPU_child';logs=out/'actual_CPU_watchdog';logs.mkdir()
    identity=d.selected_python_identity(sys.executable);d.WORKER=fixture;life=d.lifecycle();command=[str(Path(sys.executable)),'-u',str(fixture),'--plan',str(plan),'--output',str(output),'--run']
    observer=lambda pid,parent:life.observe_actual_pid(pid,parent,fixture,plan,output)
    error=None
    try:result=life.watch(command,output,logs,hard_seconds=10,observer=observer)
    except BaseException as exc:
        error=repr(exc);result=json.loads((logs/'state.json').read_text()) if (logs/'state.json').exists() else {'status':'missing_terminal_metadata'}
    try:post=d.terminal_processes([result.get('launcher_pid'),result.get('actual_pid')])
    except Exception as exc:post={'status':'UNVERIFIED','queried_pids':[],'actual_CIM_rows':None,'error':repr(exc)}
    (logs/'post_terminal_CIM.json').write_text(json.dumps(post,indent=2)+'\n',encoding='utf-8')
    runtime=json.loads((output/'runtime.json').read_text()) if (output/'runtime.json').exists() else {}
    expected={'plan':plan,'output':output,'command':command,'plan_sha':sha(plan),'worker_sha':sha(fixture),'python_identity':identity}
    checks=[]
    def qualify(r=result,u=runtime,p=post,identity_override=None):return d.qualify_terminal_identity(r,u,p,**{**expected,**({'python_identity':identity_override} if identity_override is not None else {})})
    def check(name,call):
        try:
            if not call():raise AssertionError('False result')
            checks.append({'name':name,'passed':True})
        except Exception as exc:checks.append({'name':name,'passed':False,'error':repr(exc)})
    actual=qualify();(logs/'engineering_identity.json').write_text(json.dumps(actual,indent=2)+'\n',encoding='utf-8')
    check('one_real_child_completed_exact_identity',lambda:error is None and actual['engineering_identity_qualified'])
    check('real_worker_reports_selected_venv',lambda:os.path.samefile(runtime['sys_executable'],identity['chosen_python']))
    check('real_worker_base_matches_CIM_and_expected',lambda:os.path.samefile(runtime['base_executable'],result['actual_process_OS_observation']['executable_path']) and os.path.samefile(runtime['base_executable'],identity['base_executable_resolved']))
    check('venv_redirector_distinct_from_base_file',lambda:not os.path.samefile(identity['chosen_python'],identity['base_executable_raw']))
    check('real_3_12_alias_and_3_12_15_resolved_samefile',lambda:identity['base_executable_raw']!=identity['base_executable_resolved'] and os.path.samefile(identity['base_executable_raw'],identity['base_executable_resolved']))
    check('exact_actual_PID_parent_creation_present',lambda:type(result.get('actual_pid')) is int and result['actual_process_OS_observation']['pid']==runtime['pid'] and result['actual_process_OS_observation']['parent_pid']==runtime['parent_pid'] and bool(result['actual_process_OS_observation']['creation_date_utc']))
    check('postterminal_actual_and_launcher_absent',lambda:post['status']=='NO_OWNED_PID_REMAINING' and post['actual_CIM_rows']==[])
    check('Windows_quoted_executable_parser',lambda:d.command_executable('"C:\\Path With Space\\python.exe" -u some.py')=='C:\\Path With Space\\python.exe')
    # Exercise raw and resolved aliases on actual file identities, without a
    # second child. OS executable and command token need not use one spelling.
    alias=copy.deepcopy(result);alias['actual_process_OS_observation']['executable_path']=identity['base_executable_resolved'];alias['actual_process_OS_observation']['command_line']='"'+identity['base_executable_raw']+'" -u '+str(fixture)+' --plan '+str(plan)+' --output '+str(output)+' --run'
    check('raw_command_alias_resolved_CIM_positive',lambda:qualify(alias)['engineering_identity_qualified'])
    reverse=copy.deepcopy(alias);reverse['actual_process_OS_observation']['executable_path']=identity['base_executable_raw'];reverse['actual_process_OS_observation']['command_line']='"'+identity['base_executable_resolved']+'" -u '+str(fixture)+' --plan '+str(plan)+' --output '+str(output)+' --run'
    check('resolved_command_raw_CIM_positive',lambda:qualify(reverse)['engineering_identity_qualified'])
    def negative(name,modify):
        r,u,p=copy.deepcopy((result,runtime,post));modify(r,u,p)
        check(name,lambda:not qualify(r,u,p)['engineering_identity_qualified'] and not qualify(r,u,p)['next_condition_permitted_by_identity'])
    negative('unobserved_rejected',lambda r,u,p:r.__setitem__('actual_process_OS_observation',{'status':'unobserved'}))
    negative('cleanup_unknown_rejected',lambda r,u,p:r.__setitem__('status','timeout_cleanup_unverified'))
    negative('exit_nonzero_rejected',lambda r,u,p:r.__setitem__('exit_code',7))
    negative('missing_actual_PID_rejected',lambda r,u,p:r.pop('actual_pid'))
    negative('runtime_PID_mismatch_rejected',lambda r,u,p:u.__setitem__('pid',1))
    negative('runtime_parent_mismatch_rejected',lambda r,u,p:u.__setitem__('parent_pid',1))
    negative('missing_creation_rejected',lambda r,u,p:r['actual_process_OS_observation'].pop('creation_date_utc'))
    negative('wrong_plan_digest_rejected',lambda r,u,p:u.__setitem__('plan_sha256','a'*64))
    negative('wrong_source_digest_rejected',lambda r,u,p:u.__setitem__('source_sha256','a'*64))
    negative('CIM_exe_is_redirector_rejected',lambda r,u,p:r['actual_process_OS_observation'].__setitem__('executable_path',identity['chosen_python']))
    negative('command_exe_is_redirector_rejected',lambda r,u,p:r['actual_process_OS_observation'].__setitem__('command_line','"'+identity['chosen_python']+'" -u '+str(fixture)+' --plan '+str(plan)+' --output '+str(output)+' --run'))
    negative('unrelated_command_rejected',lambda r,u,p:r['actual_process_OS_observation'].__setitem__('command_line','"'+identity['base_executable_raw']+'" unrelated.py'))
    negative('postterminal_unknown_rejected',lambda r,u,p:p.__setitem__('status','UNVERIFIED'))
    negative('postterminal_live_owned_PID_rejected',lambda r,u,p:p.__setitem__('actual_CIM_rows',[{'pid':runtime.get('pid') }]))
    negative('postterminal_missing_actual_query_rejected',lambda r,u,p:p.__setitem__('queried_pids',[result.get('launcher_pid')]))
    wrong=copy.deepcopy(identity);wrong['base_executable_raw']=str(fixture);wrong['base_executable_resolved']=str(fixture);wrong['base_executable_sha256']=sha(fixture)
    check('expected_unrelated_base_rejected',lambda:not qualify(identity_override=wrong)['engineering_identity_qualified'])
    check('old_sources_43_and27_receipts_retained',lambda:sha(ROOT/'scripts/track_prefix_warp.py')=='8834a386f001cc7a138fba3906774bc11d6a2a996c0695ddfc772eff11a90db7' and sha(ROOT/'scripts/run_track_prefix.py')=='1ef052e110615f8edaa9a87956c833166e85f8e7a392facef3e946207721c7b7' and sha(ROOT/'scripts/run_track_prefix_v2.py')=='b260ad4fef0329d283b2c0d1e3425d0c86c0854fa00f56f8879996070ad34d15')
    check('no_Torch_Warp_or_physics_imports',lambda:not any(k in sys.modules for k in ('torch','warp','mujoco_warp','mujoco')) and runtime.get('Torch_imported') is False and runtime.get('Warp_imported') is False)
    (out/'driver_v2_to_v3.diff').write_text(''.join(difflib.unified_diff((ROOT/'scripts/run_track_prefix_v2.py').read_text().splitlines(keepends=True),SOURCE.read_text().splitlines(keepends=True),fromfile='run_track_prefix_v2.py',tofile=SOURCE.name)),encoding='utf-8')
    report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','check_count':len(checks),'passed':sum(c['passed'] for c in checks),'checks':checks,'source_sha256':sha(SOURCE),'checker_sha256':sha(__file__),'fixture_sha256':sha(fixture),'fixture_plan_sha256':sha(plan),'selected_python_identity':identity,'actual_identity':actual,'watchdog_status':result.get('status'),'actual_pid':result.get('actual_pid'),'launcher_pid':result.get('launcher_pid'),'actual_OS_observation':result.get('actual_process_OS_observation'),'post_terminal_CIM':post,'watchdog_elapsed_s':result.get('elapsed_s'),'wall_seconds':time.perf_counter()-start,'attempted_CPU_children':1,'automatic_retry':False,'future_bundle_read':False,'future_physics_outputs_created':False,'physics_steps':0,'GPU_started':False,'real_CPU_probe_error':error,'scope':'One actual stdlib child proves this Windows venv/base/CIM/lifecycle route only, plus actual-file alias counterfactual metadata checks. No native dynamics/control/solver qualification.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ('checks','actual_identity','selected_python_identity')},indent=2));raise SystemExit(report['status']!='PASS')
if __name__=='__main__':main()
