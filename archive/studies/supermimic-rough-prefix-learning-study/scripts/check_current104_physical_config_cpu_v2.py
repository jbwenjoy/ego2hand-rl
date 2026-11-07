"""Routing-only CPU check of newhelper3 attempt; no old numerical rerun."""
from __future__ import annotations
import ast,copy,hashlib,importlib.util,json,os,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/current104-physical-FD-v2.json'
ENTRY=ROOT/'scripts/probe_current104_physical_fd_v2.py'
SUP=ROOT/'scripts/run_current104_physical_fd_v2.py'
BUILDER=ROOT/'scripts/build_current104_physical_probe_config_cpu_v2.py'
OUT=ROOT/'outputs/current104_physical_v2_config_CPU_v1'


def pin(path):
    path=Path(path);return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='-1';OUT.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),BUILDER,ENTRY,SUP,ROOT/'src/rough_prefix/physical_fd_v2.py',
             ROOT/'src/rough_prefix/probe_transaction_v3.py',ROOT/'protocols/current104-physical-fd-execution-v3.json',
             ROOT/'scripts/check_current104_physical_config_cpu.py']
    before={str(p):pin(p) for p in sources}
    for i,p in enumerate(sources):(OUT/(str(i)+'__'+p.name)).write_bytes(p.read_bytes())
    run_cpu=module(sources[-1],'frozen_CPU_retained_command_helper').run_cpu
    checks={};failure=None
    try:
        if CONFIG.exists():raise FileExistsError('Fresh versioned config required; do not rebuild/overwrite')
        result=run_cpu([sys.executable,BUILDER],OUT,'builder2_default')
        checks['builder2_actual_once_fixed_v2']=result==pin(CONFIG)
        cfg=json.loads(CONFIG.read_text())
        checks['Gfalse_fixedfresh_output2']=cfg['GPU_run_authorized'] is False and cfg['output']==str(ROOT/'outputs/current104_physical_FD_v2') and not Path(cfg['output']).exists() and not Path(cfg['output']+'_watchdog').exists()
        result=run_cpu([sys.executable,ENTRY,'--config',CONFIG],OUT,'entry2_default')
        checks['entry2_actual_default_CPU']=result['Torch_imported'] is False and result['Warp_imported'] is False and result['GPU_started'] is False and result['physics_steps']==0
        result=run_cpu([sys.executable,SUP,'--config',CONFIG],OUT,'supervisor2_default')
        checks['supervisor2_actual_default_CPU_no_watch']=result['Torch_imported'] is False and result['Warp_imported'] is False and result['GPU_started'] is False and result['physics_steps']==0
        entry=module(ENTRY,'newhelper3_physical_entry_CPU_metadata')
        def reject(name,edit):
            c=copy.deepcopy(cfg);edit(c);path=OUT/(name+'.json');path.write_text(json.dumps(c,indent=2)+'\n',encoding='utf8')
            try:entry.preflight(path)
            except (ValueError,KeyError,TypeError,AssertionError,FileNotFoundError) as e:
                checks[name]=True;(OUT/(name+'_rejection.txt')).write_text(repr(e),encoding='utf8')
            else:checks[name]=False
        def protocol2(c):
            p=pin(ROOT/'protocols/current104-physical-fd-execution-v2.json');c['protocol']=p;c['source_pins']['protocol']=p
        reject('reject_execution_revision2',protocol2)
        reject('reject_old_output1',lambda c:c.update(output=str(ROOT/'outputs/current104_physical_FD_v1')))
        reject('reject_helper_v2_closure',lambda c:c['source_pins'].update(transaction_helper=pin(ROOT/'src/rough_prefix/probe_transaction_v2.py')))
        reject('reject_worker_v1_closure',lambda c:c['source_pins'].update(worker=pin(ROOT/'src/rough_prefix/physical_fd.py')))
        reject('reject_entry_v1_closure',lambda c:c['source_pins'].update(entry=pin(ROOT/'scripts/probe_current104_physical_fd.py')))
        reject('reject_supervisor_v1_closure',lambda c:c['source_pins'].update(supervisor=pin(ROOT/'scripts/run_current104_physical_fd.py')))
        old=json.loads((ROOT/'outputs/current104_physical_FD_v1/config.json').read_text())
        def authorization1(c):
            c['GPU_run_authorized']=True;c['source_pins']['Root_authorization']=old['source_pins']['Root_authorization']
        reject('reject_old_attempt_authorization',authorization1)
        tree=ast.parse(SUP.read_text());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reject_other_workers')
        pattern=next(n.value.value for n in ast.walk(func) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pattern' for t in n.targets))
        names=['probe_current104_physical_fd.py','probe_current104_physical_fd_v2.py','probe_current104_physical_fd_v99.py',
               'evaluate_rough_prefix_pen_v2.py','train_rough_prefix_pen_v5.py','track_prefix_warp.py','train_controlled_pen.py',
               'evaluate_pen_checkpoint.py','evaluate_confirmation_checkpoint.py','evaluate_c2_portfolio.py','probe_world_model_gradients.py','rest_hold_warp.py']
        checks['new_regex_keeps_old_and_versioned_other_workers']=all(re.search(pattern,'D:\\workspace\\x\\'+name+' --config p') for name in names)
        checks['new_regex_does_not_match_suffix_debug']=re.search(pattern,'D:\\workspace\\x\\probe_current104_physical_fd_v2_debug.py --config p') is None
        checks['helper3_exact21proof_source']=cfg['source_pins']['transaction_helper']['sha256']=='43378a259abf3a38c9f06bca98cf50e755de1f99089ddfe28229d86ac1012851'
        checks['all_new_snapshot_sources_unchanged']=all(pin(p)==v for p,v in before.items())
        checks['all_old411pins_unchanged']=all(pin(v['path'])['sha256']==v['sha256'] for v in old['source_pins'].values())
        checks['no_framework_in_checker']=not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp'))
    except BaseException as exc:
        import traceback
        failure={'error':repr(exc),'traceback':traceback.format_exc()}
    report={'schema':'current104_physical_v2_routing_CPU_contract:1','status':'PASS' if checks and all(checks.values()) and failure is None else 'FAIL',
      'checks':checks,'passed':sum(checks.values()),'total':len(checks),'failure':failure,'config':pin(CONFIG) if CONFIG.exists() else None,
      'source_snapshots':before,'GPU_used':False,'Env_created':False,'WM_forwards':0,'physics_steps':0,'directions_regenerated':False,
      'scope':'New routing/default/closure/revision checks only. Old40/21/12 and numericaldiagnostics not rerun; no native/GPU qualification.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'report':pin(OUT/'report.json')}))
    if report['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
