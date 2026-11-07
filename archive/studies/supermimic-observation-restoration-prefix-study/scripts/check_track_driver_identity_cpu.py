"""Pure CPU positive/negative identity-gate checks; no child, CIM or GPU started."""
from __future__ import annotations
import argparse,ast,copy,hashlib,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/run_track_prefix_v2.py'
spec=importlib.util.spec_from_file_location('dynamic_prefix_strict_identity_driver',SOURCE);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();start=time.perf_counter();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    plan=ROOT/'configs/FUTURE_NOT_READ.json';output=ROOT/'outputs/FUTURE_NOT_CREATED';worker=ROOT/'scripts/track_prefix_warp.py';command=['python.exe','-u',str(worker),'--plan',str(plan),'--output',str(output),'--run'];plan_sha='a'*64;worker_sha=sha(worker)
    runtime={'pid':4242,'parent_pid':4040,'started_utc':'2026-11-03T00:00:00.200000+00:00','plan_sha256':plan_sha,'source_sha256':worker_sha}
    result={'status':'completed','exit_code':0,'timeout':False,'supervisor_pid':4000,'launcher_pid':4040,'actual_pid':4242,'command':command,'worker_output':str(output),'Popen_prelaunch_utc':'2026-11-03T00:00:00.000000+00:00','runtime_metadata':runtime.copy(),'actual_process_OS_observation':{'status':'exact_OS_identity_observed','pid':4242,'parent_pid':4040,'creation_date_utc':'2026-11-03T00:00:00.1000000Z','observed_utc':'2026-11-03T00:00:00.300000+00:00','executable_path':'D:/Python/python.exe','command_line':'D:/Python/python.exe -u '+str(worker)+' --plan '+str(plan)+' --output '+str(output)+' --run'}}
    post={'status':'NO_OWNED_PID_REMAINING','queried_pids':[4040,4242],'actual_CIM_rows':[]};checks=[]
    def qualify(r=result,u=runtime,p=post):return d.qualify_terminal_identity(r,u,p,plan=plan,output=output,command=command,plan_sha=plan_sha,worker_sha=worker_sha)
    def check(name,call):
        try:
            if not call():raise AssertionError('False result')
            checks.append({'name':name,'passed':True})
        except Exception as exc:checks.append({'name':name,'passed':False,'error':repr(exc)})
    check('exact_observed_terminal_identity_pass',lambda:qualify()['engineering_identity_qualified'])
    def negative(name,modify):
        r,u,p=copy.deepcopy((result,runtime,post));modify(r,u,p)
        check(name,lambda:not qualify(r,u,p)['engineering_identity_qualified'] and not qualify(r,u,p)['next_condition_permitted_by_identity'])
    negative('completed_but_unobserved_rejected',lambda r,u,p:r.__setitem__('actual_process_OS_observation',{'status':'unobserved'}))
    negative('completed_but_missing_actualPID_rejected',lambda r,u,p:r.pop('actual_pid'))
    negative('cleanup_unverified_rejected',lambda r,u,p:r.__setitem__('status','timeout_cleanup_unverified'))
    negative('nonzero_exit_rejected',lambda r,u,p:r.__setitem__('exit_code',1))
    negative('timeout_rejected',lambda r,u,p:r.__setitem__('timeout',True))
    negative('runtime_wrong_PID_rejected',lambda r,u,p:u.__setitem__('pid',4343))
    negative('runtime_wrong_parent_rejected',lambda r,u,p:u.__setitem__('parent_pid',1234))
    negative('OS_wrong_PID_rejected',lambda r,u,p:r['actual_process_OS_observation'].__setitem__('pid',1234))
    negative('OS_wrong_parent_rejected',lambda r,u,p:r['actual_process_OS_observation'].__setitem__('parent_pid',1234))
    negative('OS_missing_creation_rejected',lambda r,u,p:r['actual_process_OS_observation'].pop('creation_date_utc'))
    negative('OS_naive_creation_rejected',lambda r,u,p:r['actual_process_OS_observation'].__setitem__('creation_date_utc','2026-11-03T00:00:00.1'))
    negative('OS_creation_after_runtime_rejected',lambda r,u,p:r['actual_process_OS_observation'].__setitem__('creation_date_utc','2026-11-03T00:00:01Z'))
    negative('OS_creation_before_launch_rejected',lambda r,u,p:r['actual_process_OS_observation'].__setitem__('creation_date_utc','2026-11-02T23:59:59Z'))
    negative('OS_wrong_command_rejected',lambda r,u,p:r['actual_process_OS_observation'].__setitem__('command_line','D:/Python/python.exe other.py --run'))
    negative('OS_missing_executable_rejected',lambda r,u,p:r['actual_process_OS_observation'].pop('executable_path'))
    negative('watch_command_mismatch_rejected',lambda r,u,p:r.__setitem__('command',['other']))
    negative('runtime_wrong_planSHA_rejected',lambda r,u,p:u.__setitem__('plan_sha256','b'*64))
    negative('runtime_wrong_sourceSHA_rejected',lambda r,u,p:u.__setitem__('source_sha256','b'*64))
    negative('captured_terminal_runtime_mismatch_rejected',lambda r,u,p:r['runtime_metadata'].__setitem__('started_utc','2026-11-03T00:00:00.1Z'))
    negative('post_CIM_unavailable_rejected',lambda r,u,p:p.__setitem__('status','UNVERIFIED'))
    negative('post_CIM_ownedPID_live_rejected',lambda r,u,p:p.__setitem__('actual_CIM_rows',[{'pid':4242}]))
    negative('post_CIM_did_not_query_actualPID_rejected',lambda r,u,p:p.__setitem__('queried_pids',[4040]))
    rr,uu,pp=copy.deepcopy((result,runtime,post));rr['actual_pid']=4040;rr['actual_process_OS_observation'].update(pid=4040,parent_pid=4000);uu.update(pid=4040,parent_pid=4000);rr['runtime_metadata']=uu.copy();pp['queried_pids']=[4040]
    check('no_venv_redirector_same_PID_parent_scope',lambda:qualify(rr,uu,pp)['engineering_identity_qualified'])
    tree=ast.parse(SOURCE.read_text());check('same_45second_watch_call',lambda:any(isinstance(n,ast.Call) and ast.unparse(n.func)=='life.watch' and any(k.arg=='hard_seconds' and isinstance(k.value,ast.Constant) and k.value.value==45 for k in n.keywords) for n in ast.walk(tree)))
    check('old_worker_and_driver_remain_immutable',lambda:sha(worker)=='8834a386f001cc7a138fba3906774bc11d6a2a996c0695ddfc772eff11a90db7' and sha(ROOT/'scripts/run_track_prefix.py')=='1ef052e110615f8edaa9a87956c833166e85f8e7a392facef3e946207721c7b7')
    check('no_Torch_Warp_GPU_or_future_directories',lambda:not any(k in sys.modules for k in ('torch','warp','mujoco_warp')) and not output.exists() and not plan.exists())
    report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','check_count':len(checks),'passed':sum(c['passed'] for c in checks),'checks':checks,'source_sha256':sha(SOURCE),'checker_sha256':sha(__file__),'worker_sha256':worker_sha,'scope':'Pure metadata fixtures and AST, no OSchild/CIM/GPU/physics and no future candidate read. RealCIM/actual process binding remains a future execution gate.','future_outputs_created':False,'processes_launched':0,'physics_steps':0,'GPU_started':False,'wall_seconds':time.perf_counter()-start}
    for source in (SOURCE,Path(__file__)):(out/source.name).write_bytes(source.read_bytes())
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(report,indent=2));raise SystemExit(report['status']!='PASS')
if __name__=='__main__':main()
