"""New-schema prior health fixtures and inherited-AST checks; no actual run."""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/audit_full151_checkpoint_cpu_v2.py'
OLD=ROOT/'scripts/audit_full151_checkpoint_cpu_v1.py'


def main(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();checks={};notes={}
    sp=importlib.util.spec_from_file_location('_full151_three_arm_health_v2_CPU_fixture',SOURCE)
    a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
    source_pins={'auditor_v2':a.pin(SOURCE),'checker':a.pin(__file__),'original_health_v1':a.pin(OLD)}
    for filename in ('learning_v2.py','prior_hook_v1.py','training_capacity.py','controller.py','runtime.py','native.py'):
        path=ROOT/'src/full151_native'/filename
        if path.exists():source_pins[filename]=a.pin(path)
    def check(name,value):
        checks[name]=bool(value)
        if not checks[name]:raise AssertionError(name)
    def reject(name,call):
        try:value=call()
        except (ValueError,TypeError,KeyError,IndexError):checks[name]=True;return
        if isinstance(value,tuple):value=value[0]
        check(name,value is False or isinstance(value,dict) and not all(value.values()))
    try:
        check('original_1fb9_source_unchanged',a.digest(OLD)=='1fb9da1e721fc3fb01559a7a526eb67efaa9dbff5d9753e6b97546a383213fa9')
        trees=[]
        for p in (OLD,SOURCE):trees.append({node.name:ast.dump(node,include_attributes=False) for node in ast.parse(p.read_text()).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))})
        unchanged=('effective_config_equal','optimizer_counts_match','expected_collect_steps','budget_counts','is_final_artifact','qualified_final_scope','expected_shapes','capacity_checks','policy_checks','actualu_checks','initial_checks')
        for name in unchanged:check('inherited_'+name+'_AST_exact',trees[0][name]==trees[1][name])
        check('CPU_no_native_or_GPU',not torch.cuda.is_available() and not torch.cuda.is_initialized() and not any(n.split('.')[0] in ('mujoco','warp','mujoco_warp','mjlab','dex_supermimic') for n in sys.modules))
        rng=torch.get_rng_state().clone()
        effective={'wm_only_iters':20,'horizon':8,'pi_batch':128}
        spec={'path':str(out/'synthetic_spec.json'),'sha256':'1'*64}
        cfg={'schema':a.CONFIG_SCHEMA,'role':'engineering_smoke','method':'A2','contact_binding':spec,'input_bundle':{'path':'SYNTHETIC_INPUT','sha256':'2'*64}}
        hdr={'format_version':105,**{k:{'fixture':True} for k in ('full151_controller','full151_runtime_evidence','full151_capacity_evidence','full151_initial_parameter_fingerprints','full151_prior_evidence')}}
        for method in ('A1L','P_ONLY','A2'):
            check('new105_schema2_method_'+method,a.new105_header_ok(hdr,cfg|{'method':method,'contact_binding':None if method=='A1L' else spec}))
        check('reject_schema1_config',not a.new105_header_ok(hdr,cfg|{'schema':'full151_knownrough_native_training_run:1'}))
        check('reject_old104_with_relabelled105_payload',not a.new105_header_ok(hdr|{'rough_controller':{}},cfg))
        check('reject_unknown_method',not a.new105_header_ok(hdr,cfg|{'method':'A0'}))
        check('reject_A1L_prior_spec_attachment',not a.new105_header_ok(hdr,cfg|{'method':'A1L'}))
        check('reject_A2_missing_spec',not a.new105_header_ok(hdr,cfg|{'contact_binding':None}))
        def fixture(method='A2',policy_calls=20):
            counts={k:0 for k in a.PRIOR_COUNTERS};logs=[];latest=None
            base={'schema':'full151_imagination_prior_checkpoint_evidence:1','method':method,'status':'IMAGINATION_ONLY','spec_pin':spec,'objective_metadata':{'fixture':True},
                'dt_s':.02,'imagined_soft_required':True,'real_collect_shaped':False,'WMfit_shaped':False,'logging_errors':[],
                'dead_end_rule':'unchanged native full151 phase<150 hardmask after reward'}
            for i in range(policy_calls):
                starts=np.resize(np.array([0,142,143,148,149],int)+i%2,128).clip(max=149)
                mask=starts[None]+np.arange(8)[:,None]<150
                rows=int(mask.sum());counts['policy_update_calls']+=1;counts['objective_forward_calls']+=8
                counts['active_rows']+=rows;counts['inactive_rows']+=1024-rows;counts['dt_applications']+=8
                logs.append({'schema':'full151_imagination_prior_policy_log:1','method':method,'dt_s':.02,'policy_update_call':i+1,'iteration':20+i,
                    'objective_forward_calls':8,'active_rows':rows,'inactive_rows':1024-rows,'dt_applications':8,'complete':True,
                    'cumulative':{**base,**dict(counts)},'active_means':{'prior/bonus_unscaled':0.,'prior/support_match':.25}})
                latest={'active_mask_H_B':mask.tolist()}
            ev={**base,**counts}
            return ev,logs,latest
        for method in ('P_ONLY','A2'):
            ev,logs,latest=fixture(method)
            check('positive_'+method+'_H8_log_counts',all(a.prior_counter_checks(ev,cfg|{'method':method},effective,40,logs,latest)[0].values()))
            prefixev,prefixlogs,prefixlatest=fixture(method,0)
            check('positive_'+method+'_WM_only_prefix_no_policy',all(a.prior_counter_checks(prefixev,cfg|{'method':method},effective,20,[],{'status':'NOT_RUN'})[0].values()))
        ev,logs,latest=fixture()
        extra=copy.deepcopy(logs[-1]);extra['policy_update_call']=21
        pc,pd=a.prior_counter_checks(ev,cfg,effective,40,logs+[extra],latest)
        check('live_later_policy_append_ignored',all(pc.values()) and pd['later_complete_rows_ignored']==1)
        for name,edit in (
            ('bool_counter',lambda v:v.update(objective_forward_calls=True)),
            ('double_dt',lambda v:v.update(dt_applications=320)),
            ('one_forward_per_policy',lambda v:v.update(objective_forward_calls=20)),
            ('wrong_method',lambda v:v.update(method='P_ONLY')),
            ('wrong_spec',lambda v:v.update(spec_pin={'path':'wrong','sha256':'0'*64})),
            ('negative_rows',lambda v:v.update(active_rows=-1)),
            ('missing_metadata',lambda v:v.update(objective_metadata=None)),
            ('noninteger_counter',lambda v:v.update(policy_update_calls=20.0))):
            changed=copy.deepcopy(ev);edit(changed);reject('reject_'+name,lambda changed=changed:a.prior_counter_checks(changed,cfg,effective,40,logs,latest))
        for name,edit in (
            ('incomplete_policy',lambda v:v[2].update(complete=False)),
            ('bad_cumulative',lambda v:v[2]['cumulative'].update(active_rows=0)),
            ('wrong_iteration',lambda v:v[2].update(iteration=99)),
            ('wrong_call_order',lambda v:v[2].update(policy_update_call=9)),
            ('double_dt_log',lambda v:v[2].update(dt_applications=16)),
            ('nonfinite_means',lambda v:v[2]['active_means'].update(**{'prior/bonus_unscaled':float('nan')}))):
            changed=copy.deepcopy(logs);edit(changed);reject('reject_log_'+name,lambda changed=changed:a.prior_counter_checks(ev,cfg,effective,40,changed,latest))
        reject('reject_missing_prefix_log',lambda:a.prior_counter_checks(ev,cfg,effective,40,logs[:-1],latest))
        wrongmask=copy.deepcopy(latest);wrongmask['active_mask_H_B'][0][0]=False
        reject('reject_latest_active_mask_mismatch',lambda:a.prior_counter_checks(ev,cfg,effective,40,logs,wrongmask))
        none={'schema':'full151_imagination_prior_checkpoint_evidence:1','method':'A1L','status':'NONE','spec_pin':None,'objective_metadata':None,
            'dt_s':.02,'imagined_soft_required':True,'real_collect_shaped':False,'WMfit_shaped':False,'logging_errors':[],
            'dead_end_rule':'unchanged native full151 phase<150 hardmask after reward',**{k:0 for k in a.PRIOR_COUNTERS}}
        nonecfg=cfg|{'method':'A1L','contact_binding':None}
        check('A1L_NONE_no_prior_even_after20policy',all(a.prior_counter_checks(none,nonecfg,effective,40,[])[0].values()))
        reject('A1L_reject_any_prior_log',lambda:a.prior_counter_checks(none,nonecfg,effective,40,logs))
        reject('A1L_reject_prior_call_count',lambda:a.prior_counter_checks(none|{'policy_update_calls':20},nonecfg,effective,40,[]))
        manifest={'target':{'path':'SYNTHETIC_TARGET','sha256':'3'*64}}
        mspec={'schema':'full151_declared_H123_objective_spec:2','GPU_authorized':False,'binding_manifest':{'path':'BINDING','sha256':'4'*64},'target_receipt':{'path':'TARGET_RECEIPT','sha256':'5'*64},'independent_target_review':{'path':'REVIEW','sha256':'6'*64}}
        metadata={'input_bundle':cfg['input_bundle'],'binding_manifest':mspec['binding_manifest'],'target':manifest['target'],'target_receipt':mspec['target_receipt'],'independent_target_review':mspec['independent_target_review'],
            'retarget_variant_id':a.VARIANT,'contact_intent_variant_id':'source_rigid_H123_surface_projection_all151_v1','source_window_half_open':[200,351],
            'frame_count':151,'active_real_imagination_intervals':[0,150],
            **{k:False for k in ('human_contact_observation_known','source_confidence_known','target_defined_from_policy_contacts','robotq_used_to_define_targets','qualified_contact_predictor','forceclosure_qualified','dynamics_gradient_qualified','physical_reference_qualified')},
            'local_proxy_qualification':'FAILED_RETAINED_EXPLORATORY_ONLY','compiled_precision_qualification':'FAILED_RETAINED_NOT_REVALIDATED',
            'dt_applied_inside_module':False,'caller_dt_s':.02,'penetration_allowance_m':.003,'penetration_normalization_m':.001,
            'formula':'phase_weight*official_support_match-.1*penetration_prior','scoring':{'support_coefficient':1.,'no_finger_identity_or_extra_contact_penalty':True},
            'basis':{'training_directions':64,'evaluation_directions':512,'training_indices':list(range(0,512,8))},'unknown_masks_attraction_only':True,'known_frame_count':151,'active_slot_count':453}
        metadata.update(objective_spec_schema=mspec['schema'],objective_spec_sha256=hashlib.sha256(a.canonical(mspec).encode()).hexdigest(),
            execution_scope='EXPLORATORY_MATCHED_TRAINING',CPU_authorized=True,training_authorized=True,GPU_authorized=False,
            representation_vs_native={'target_support_mu':.1,'target_unitforce_edges':8,'target_normal_ray':True,'target_direct_torsion':False,'native_condim':4,'same_representation':False,'forward_and_targets_changed':False},
            matched_native_parameter_cone={'condition':'H123','feasible_frames':151,'frame_count':151,'normalforce_uppercap':None,'columns_per_contact':6,'condim':4},
            legacy_proxy_cone_status={'feasible':23,'solver_infeasible':128,'frame0_feasible':False,'retained':True})
        check('A2_metadata_identity_and_failed_qualification_retained',all(a.objective_metadata_checks(metadata,'A2',cfg,mspec,manifest).values()))
        penalty=copy.deepcopy(metadata);penalty.update(formula='-.1*penetration_prior',support_coefficient=0.,penetration_coefficient=.1,support_score_is_diagnostic_only=True);penalty['scoring']['support_coefficient']=0.
        check('P_ONLY_direct_negative_point1P_metadata',all(a.objective_metadata_checks(penalty,'P_ONLY',cfg,mspec,manifest).values()))
        for name,edit in (
            ('physical_promotion',lambda v:v.update(physical_reference_qualified=True)),
            ('human_label',lambda v:v.update(human_contact_observation_known=True)),
            ('policy_leak',lambda v:v.update(target_defined_from_policy_contacts=True)),
            ('wrong_window',lambda v:v.update(source_window_half_open=[0,151])),
            ('double_internal_dt',lambda v:v.update(dt_applied_inside_module=True)),
            ('wrong_basis',lambda v:v['basis'].update(training_directions=512)),
            ('known_as_release',lambda v:v.update(unknown_masks_attraction_only=False))):
            changed=copy.deepcopy(metadata);edit(changed);reject('reject_metadata_'+name,lambda changed=changed:a.objective_metadata_checks(changed,'A2',cfg,mspec,manifest))
        hashes={'python':'1'*64,'numpy':'2'*64,'torch_cpu':'3'*64,'torch_cuda':['4'*64],'cuda_initialized':True}
        rngreport={'schema':'full151_objective_build_rng:1','status':'PRESERVED','method':'A2','spec_pin':spec,'factory_error':None,
            'checks':{k:True for k in ('Python_exact','NumPy_exact','TorchCPU_exact','CUDA_initialization_unchanged','already_initialized_CUDA_exact')},'before':hashes,'after':copy.deepcopy(hashes)}
        check('objective_import_factory_rng_preserved_saved_hash_fixture',all(a.objective_build_rng_checks(rngreport,'A2',spec).values()))
        changed=copy.deepcopy(rngreport);changed['after']['torch_cuda']=['5'*64]
        reject('reject_saved_RNG_changed_despite_alltrue_flags',lambda:a.objective_build_rng_checks(changed,'A2',spec))
        changed=copy.deepcopy(rngreport);changed['checks']['Python_exact']=False
        reject('reject_rng_failure_record',lambda:a.objective_build_rng_checks(changed,'A2',spec))
        check('A1L_factory_not_constructed_RNG_scope',all(a.objective_build_rng_checks({'schema':'full151_objective_build_rng:1','status':'NOT_CONSTRUCTED','method':'A1L','objective':'NONE','RNG_capture_required':False},'A1L',None).values()))
        check('torch_RNG_and_sources_unchanged',torch.equal(rng,torch.get_rng_state()) and all(a.pin(v['path'])==v for v in source_pins.values()))
        status='PASS_NEW_THREE_ARM_HEALTH_CPU_CONTRACTS_ONLY'
    except Exception as exc:
        status='CPU_CHECKER_FAILURE_RETAINED';notes={'exception':repr(exc),'traceback':traceback.format_exc()}
    report={'schema':'full151105_three_arm_health_auditor_CPU_contract:1','status':status,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'source_pins':source_pins,'notes':notes,
        'actual_new_schema2_checkpoint_audit':'NOT_RUN_NO_ACTUAL_NEW_TRAINING_ARTIFACT','scope':'Synthetic prior counter/log/metadata fixtures; inherited base AST checked instead of repeating64 old fixtures. No objective construction, WM/Actor forward, optimizer, Env, geometry, physics or GPU.',
        'GPU_used':False,'created_environment':False,'new_training':False,'wall_seconds':time.perf_counter()-start}
    a.json_write(out/'report.json',report)
    print(json.dumps({'status':status,'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':a.pin(out/'report.json')},indent=2))
    return 0 if status.startswith('PASS_') else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();raise SystemExit(main(args.output))
