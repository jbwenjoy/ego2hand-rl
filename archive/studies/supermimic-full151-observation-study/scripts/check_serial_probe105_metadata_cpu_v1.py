"""Pure planning/metadata source contracts; never call run_probe or object_terms."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
SOURCE=ROOT/'src/full151_gradient/serial_probe105_v1.py'

def pin(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);snapshot=output/'source_before';snapshot.mkdir()
    for path in (SOURCE,Path(__file__)):(snapshot/path.name).write_bytes(path.read_bytes())
    sys.path.insert(0,str(ROOT/'src'))
    module=importlib.import_module('full151_gradient.serial_probe105_v1')
    from full151_native.metadata import write
    checks={};details={}
    def check(name,value):
        if not value:raise AssertionError(name)
        checks[name]=True
    def rejects(name,fn):
        try:fn()
        except (ValueError,KeyError,FileNotFoundError) as exc:checks[name]=True;details[name]=repr(exc);return
        raise AssertionError('Expected rejection: '+name)
    for mode in ('serial_full','engineering_restore_smoke'):
        plan=module.execution_plan(mode);rows=plan['branch_slots'];snapshots=plan['snapshot_slots'];full=mode=='serial_full'
        check(mode+'_all152slots',len(rows)==152 and [r['row_id'] for r in rows]==list(range(152)))
        check(mode+'_all8snapshots',len(snapshots)==8 and all(s['status']=='NOT_RUN' for s in snapshots))
        check(mode+'_scheduled_coverage',sum(r['scheduled'] for r in rows)==(152 if full else 3) and sum(s['scheduled'] for s in snapshots)==(8 if full else 1))
        check(mode+'_finite_budget_math',plan['episode_substeps']==len(plan['donors'])*plan['donor_intervals']*4+plan['branch_count']*4)
        check(mode+'_drift_index_contract',max(plan['next_ticks'])<plan['donor_state_rows'] and max(plan['snapshot_ticks'])<plan['donor_state_rows'])
        check(mode+'_terminal150_only_response',plan['next_ticks'][-1]==(600 if full else 4) and plan['donor_intervals']==(150 if full else 1))
        check(mode+'_NOT_RUN_fields',all(r['status']=='NOT_RUN' and r['actualu'] is None and r['r_k'] is None and r['r_kplus1'] is None and not r['restorequalified'] for r in rows))
        for sid in range(8):
            group=rows[sid*19:(sid+1)*19]
            check(mode+'_snapshot_order_'+str(sid),all(r['snapshot_id']==sid and r['path']==module.PATHS[sid//4] and r['k']==module.PHASES[sid%4] for r in group)
            check(mode+'_branch_order_'+str(sid),[r['baseline_repeat'] for r in group[:3]]==[0,1,2] and [(r['epsilon_index'],r['joint_index'],r['sign']) for r in group[3:]]==[(ei,j,s) for ei in range(2) for j in (0,4,8,12) for s in (-1,1)])
    rejects('unknown_mode',lambda:module.execution_plan('old104'))
    arrays,directions=module.allocate_artifacts()
    check('directions_exactkeys',set(directions)=={'plus','minus','d_eff','center_offset','centers','clipped'})
    check('direction_shapes',all(directions[k].shape==(8,4,2,20) for k in ('plus','minus','d_eff','center_offset')))
    check('effective_direction_F64',directions['d_eff'].dtype==directions['center_offset'].dtype==np.float64)
    check('physical_endpoints_F32',directions['plus'].dtype==directions['minus'].dtype==np.float32)
    check('unrun_NaNs_and_masks',all(np.isnan(v).all() for k,v in arrays.items() if v.dtype==np.float32) and not arrays['completed'].any() and not arrays['snapshot_completed'].any())
    check('array_shapes',arrays['snapshot_x'].shape==(8,58) and arrays['branch_x_next'].shape==(152,58) and arrays['branch_self15'].shape==(152,15) and arrays['branch_object_terms_k'].shape==(152,2))
    x=np.zeros(58,np.float32)
    check('baseline_zero_accepts',module.baseline_errors(x,x,1.,1.)['passed'])
    for name,index,excess in (('q',0,2.1e-5),('qd',20,2.1e-4),('p',40,2.1e-5),('R',43,2.1e-5),('v',52,2.1e-4),('omega',55,2.1e-4)):
        y=x.copy();y[index]=excess;check('baseline_reject_'+name,not module.baseline_errors(y,x,1.,1.)['passed'])
    check('baseline_reject_score',not module.baseline_errors(x,x,1.000021,1.)['passed'])
    base={'schema':'full151105_same_warp_physical_probe_run:1','role':'full151105_serial_physical_probe','probe_mode':'engineering_restore_smoke',
        'GPU_run_authorized':False,'training_authorized':False,'automatic_retry':False,'compile_fallback':False,'evaluation_seed':0,'soft_worker_seconds':180,'hard_process_seconds':240}
    for name,key,value in (('old_schema','schema','current104_probe'),('wrong_role','role','native_a0'),('unknown_mode','probe_mode','full'),
        ('GPU_not_bool','GPU_run_authorized',1),('training_enabled','training_authorized',True),('retry_enabled','automatic_retry',True),('fallback_enabled','compile_fallback',True),
        ('wrong_seed','evaluation_seed',1),('seed_alias_only','evaluation_seed',None),('unbounded_soft','soft_worker_seconds',181),('wrong_hard','hard_process_seconds',45)):
        config=dict(base);config[key]=value;rejects('metadata_'+name,lambda c=config:module.validate_config(c))
    tree=ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    top_imports=[ast.unparse(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    check('no_top_framework_import',not any(any(p in text for p in ('import torch','import warp','import mujoco','import mjlab')) for text in top_imports))
    probe=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_probe')
    names={n.id for n in ast.walk(probe) if isinstance(n,ast.Name)}
    check('real_runtime_controller_transaction_paths_present',{'Transaction105','full151_runtime','full151_controller','PenSimReader','ManagerBasedRlEnv','CapacityObserver'}<=names)
    check('no_Actor_WM_optimizer_construction',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('Actor','WorldModel','Adam','DexSuperMimicRunner') for n in ast.walk(probe)))
    check('no_direct_model_deserialization',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and (n.func.attr=='load_state_dict' or (n.func.attr=='load' and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch')) for n in ast.walk(probe)))
    check('source_owns_re_import',any(isinstance(n,ast.Import) and any(a.name=='re' for a in n.names) for n in tree.body))
    proposal_path=ROOT/'protocols/full151_same_warp_control_gradient_proposal_v2.json'
    proposal=json.loads(proposal_path.read_text(encoding='utf-8-sig'))
    check('proposal2_exact_post20_scalar_contract',proposal['schema']=='full151105_same_Warp_control_gradient_diagnostic_proposal:2' and proposal['primary_scalar_kind']=='matched_post20_gpu_score' and proposal['scalar_contract']==module.SCALAR_CONTRACT)
    from full151_native.io import pin as inputpin
    old=ROOT/'protocols/full151_same_warp_control_gradient_proposal_v1.json'
    old_config={**base,'proposal':inputpin(old),'protocol':inputpin(old),'input_bundle':inputpin(ROOT/'outputs/full151_native_input_v1/bundle.json')}
    rejects('old_proposal1_scalar_semantics_rejected',lambda:module.validate_config(old_config))
    score=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='controlled_object_score_FP32')
    check('matched_function_retains_input_AD_no_detach',not any(isinstance(n,ast.Attribute) and n.attr in ('detach','numpy') for n in ast.walk(score)))
    actual_preflight=None
    if args.config:
        config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        check('actual_review_config_GPUfalse',config['GPU_run_authorized'] is False)
        actual_preflight=module.validate_config(config)
        check('actual_metadata_join_passed',actual_preflight['GPU_started'] is False)
    check('frameworks_absent_after_CPUchecks',not any(n in sys.modules for n in ('torch','warp','mujoco','mjlab')))
    source_pins={'serial_probe105':pin(SOURCE),'checker':pin(__file__),'transaction105':pin(ROOT/'src/full151_gradient/transaction105.py'),
        'metadata105':pin(ROOT/'src/full151_native/metadata.py'),'io105':pin(ROOT/'src/full151_native/io.py'),
        'proposal2':pin(proposal_path),'interface_document':pin(ROOT/'docs/serial_probe105_worker_interface_v1.md')}
    report={'schema':'full151105_serial_probe_CPU_planning_metadata_contract:1','status':'PASS_CPU_PLANNING_METADATA_ONLY','passed':len(checks),'total':len(checks),'checks':checks,'details':details,
        'source_pins':source_pins,'actual_metadata_preflight':actual_preflight,'config':pin(args.config) if args.config else None,
        'physics_steps':0,'Env_instances':0,'Actor_WM_forward':0,'GPU_calls':0,'native_inventory_or_restore':'PENDING_NOT_EXECUTED',
        'scope':'Mode indices, fixedslots, host-array ABI, baseline thresholds and fail-closed metadata only. No run_probe/object_terms/backend execution.'}
    write(output/'report.json',report)
    ready={'status':'CPU_PREPARATION_ONLY_NOT_GPU_QUALIFIED','source_pins':source_pins,'cpu_report':pin(output/'report.json'),'cpu_checks':{'passed':len(checks),'total':len(checks)},
        'api':{'entry':'full151_gradient.serial_probe105_v1.run_probe(bundle,config,out,emit)','metadata':'validate_config(config,bundle=None)','planning':'execution_plan(mode), allocate_artifacts()',
        'status':['COMPLETED_ENGINEERING_RESTORE_SMOKE','COMPLETED_PHYSICAL_PROBE','INVENTORY_REVIEW_REQUIRED','FAILED_ENGINEERING_PARTIAL']},
        'donor_input_basenames':list(module.DONOR_FILES),'foreign_inventory_None':'Discovery artifacts retained then raise InventoryReviewRequired; no branch/native restore qualification'}
    write(output/'READY.json',ready)
    print(json.dumps({'status':report['status'],'passed':len(checks),'total':len(checks),'report':pin(output/'report.json'),'READY':pin(output/'READY.json')}))

if __name__=='__main__':main()
