"""SOURCE DRAFT: one new native-only B152 FW; reuse qualified old three roles.

No backend/model import on declaration or metadata. Actual execution is a fresh
Root-authorized CPU inference job, not training, AD, auxiliary-head evaluation,
old-role replay, physics, a checkpoint selector or a held-out experiment.
"""
from __future__ import annotations
import csv
import importlib.util
import os
from pathlib import Path
import sys
import time
import zipfile

from . import native_public_only_training105_v1 as nt
from . import boundary_public_self15_final_evaluation105_v1 as ev

c=nt.c
ROOT=c.ROOT;read=c.read;pin=c.pin;checked=c.checked;same=c.same;require=c.require
SCHEMA='full151105_native_public_only_common_evaluation_CPU_protocol:1'
ROLE='native52_self15_only_final1000_vs_existing_three_roles_same152_common_metrics'
PREFIX='full151105_native_public_only_common_evaluation_CPU_'
NEW_ROLE='native52_self15_only'
OLD_ROLES=('unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency')
INPUTS={'native_training_protocol','native_training_result','native_final_manifest','native_final_saved_health',
        'reference_eval_protocol','reference_eval_report','reference_eval_saved12','reference_description_report'}
HEALTH_SHA='ba7d6cae7cbcf451ff7a4b480ab44329f69f1993e759ac7ec0d2f4fa791a7e58'
CHECKPOINT_SHA='0301dceef5da06d7350931a8ad6718a76de1e517fffe7c9bbb126b7d382f9d6d'
PROTOCOL_SHA='0334a711d4db636e2ffda038819a7a4905c7fba4179ec83f51e8f127671b3220'
OWN={'consumer':ROOT/'src/full151_gradient/native_public_only_common_evaluation105_v1.py',
     'CLI':ROOT/'scripts/evaluate_native_public_only_common_cpu_v1.py'}
METRIC_SOURCE=ROOT/'scripts/describe_boundary_component_predictions_saved_cpu_v1.py'
SENSOR_SOURCE=ROOT/'src/full151_gradient/boundary_self15_cached_saved105_v1.py'
METRIC_SHA='7487e16b819271fd3b6d3d6562d4802fd97216afa4e3f666f705948f88cce036'
SENSOR_SHA='97b74e0173bb6d95b55c59d924db7c5502c83cf610907c6c3aa357d54808da5e'
BUDGET={'trusted_CPU_torch_loads':2,'native_WM_constructors':1,'native_FW_attempted':1,'native_FW_returned':1,
        'predicted_rows':152,'aux_head_FW':0,'raw_deepcopies':0,'native_difference_calls':1,
        'AD_optimizer_Actor_MJB_Env_physics_GPU':0}
PREPARED_KEYS=('x58_FP32','u20_FP32','target58_FP32','physical_xi46_FP32','effective_s52_FP32',
               'cached_self15_target_FP32','source_group_id152','source_substep_rows152x4')
COMMON_KEYS=('groups','native52_scaled_MSE_F64_of_saved_FP32_delta','state_native_delta','raw_R9_difference',
             'raw58_difference','native52_per_feature','cached_self15','native_pair_F64_rescore',
             'captured_native_pair_row_FP32_mean_F64')
ROW_COLUMNS=('role','row_id','path','k','mode','group_id','native52_scaled_MSE_F64','cached15_BCE_mean_F64',
             'cached15_Brier_F64','cached15_pred_intensity','cached15_true_intensity','captured_native_pair_row_FP32',
             'cached15_TP','cached15_FP','cached15_FN','cached15_TN','native_pair_F64_rescore',
             'q_native_delta_vector_norm','qd_native_delta_vector_norm','p_native_delta_vector_norm',
             'native_LogR_native_delta_vector_norm','v_native_delta_vector_norm','omega_native_delta_vector_norm')


def validate_protocol(p,execute=False):
    require(p.get('schema')==SCHEMA and p.get('role')==ROLE and p.get('output_prefix')==PREFIX,'New native-only common evaluation namespace')
    require(type(p.get('CPU_run_authorized')) is bool and p.get('GPU_run_authorized') is p.get('training_authorized') is p.get('automatic_retry') is False
            and type(p.get('cpu_threads')) is int and p['cpu_threads']==4,'CPU4 inference only; no retry, training or GPU')
    require(set(p.get('inputs',{}))==INPUTS and p.get('source_pins'),'Exact eight metadata inputs and current source closure')
    if execute:require(p['CPU_run_authorized'],'Root explicit CPU inference authorization')
    return p


def required_sources(p):
    train=read(checked(p['inputs']['native_training_protocol']))
    direct=[*train['source_pins'].values(),*[pin(v) for v in OWN.values()],pin(METRIC_SOURCE),pin(SENSOR_SOURCE)]
    return c.canonical_sources(direct)


def metadata_preflight(protocol_path,output=None):
    require(not any(k in sys.modules for k in ('torch','mujoco','warp')),'Metadata must not import model backends')
    p=validate_protocol(read(protocol_path));i=p['inputs']
    for v in [*i.values(),*p['source_pins'].values()]:checked(v)
    require(i['native_training_protocol']['sha256']==PROTOCOL_SHA and i['native_final_saved_health']['sha256']==HEALTH_SHA,'Exact current native-only training and final serialized health')
    prior=nt.metadata_preflight(checked(i['native_training_protocol']))
    training=prior['protocol'];result=read(checked(i['native_training_result']));health=read(checked(i['native_final_saved_health']));c.allchecks(health)
    require(health['schema']=='full151105_native_public_only_checkpoint_saved_review:1' and health['status']=='PASS_SAVED_NATIVE52_SELF15_ONLY_COMPONENT_HEALTH_NOT_ACCURACY'
            and health['engineering_qualified'] is health['final_iteration_scope'] is True and health['iteration']==1000 and health['own_CPU_checkpoint_loads']==3,'Actual final1000 saved engineering health only')
    manifest=read(checked(i['native_final_manifest']))
    require(same(health['protocol'],i['native_training_protocol']) and same(health['producer_manifest'],i['native_final_manifest'])
            and manifest['schema']==nt.MANIFEST_SCHEMA and manifest['arm']==nt.ARM and manifest['iteration']==1000
            and same(manifest['single_checkpoint'],health['single_checkpoint']) and manifest['single_checkpoint']['sha256']==CHECKPOINT_SHA,'Exact AFTER1000 single checkpoint, never100 or PRE1000 operands')
    require(result['schema']=='full151105_native_public_only_training_completion:1' and result['status']=='COMPLETED_FIXED1000_NATIVE_PUBLIC_ONLY_TRAINING'
            and result['iteration']==1000 and same(result['protocol'],i['native_training_protocol']) and same(result['recipe'],training['recipe']),'Actual completed native-only fixed1000')
    for key in ('public15_eval_protocol','public15_eval_report','public15_eval_saved12'):
        alias={'public15_eval_protocol':'reference_eval_protocol','public15_eval_report':'reference_eval_report','public15_eval_saved12':'reference_eval_saved12'}[key]
        require(same(training['inputs'][key],i[alias]),'Same actual qualified old152 evaluation parents')
    reference=read(checked(i['reference_eval_report']));review=read(checked(i['reference_eval_saved12']));c.allchecks(review)
    require(review['schema']=='full151105_boundary_public_self15_final_prediction_saved_independent:1'
            and review['status']=='PASS_SAVED_ALL3_ALL152_PUBLIC_SELF15_FIXED_FINAL_PREDICTIONS' and review['engineering_qualified'] is True
            and same(review['producer_report'],i['reference_eval_report']) and same(review['protocol'],i['reference_eval_protocol']),'Qualified existing three-role predictions reused')
    description=read(checked(i['reference_description_report']))
    require(description['schema']=='full151105_boundary_public_self15_saved_prediction_description:1'
            and same(description['producer'],i['reference_eval_report']) and tuple(description['roles'])==OLD_ROLES,'Exact saved39/456 reference metrics')
    refp=read(checked(i['reference_eval_protocol']))
    for k in nt.FIT_DIGEST_KEYS:require(result[k]==reference[k]==prior['expected_fit_digests'][k],'Same fixed original152 fit provenance')
    for v in reference['outputs'].values():checked(v)
    for v in description['outputs'].values():checked(v)
    expected_cohorts=['ALL','A0','A2']+[f'{path}/k{k}' for path in ('A0','A2') for k in (0,49,99,149)]+['baseline','direction']
    require(description['cohorts']==expected_cohorts and all(set(description['statistics'][role])==set(expected_cohorts)
            and all(set(COMMON_KEYS)<=set(s) for s in description['statistics'][role].values()) for role in OLD_ROLES),'Existing exact39 common-statistics storage ABI')
    with checked(description['outputs']['row_CSV']).open(encoding='utf-8',newline='') as f:
        fields=csv.DictReader(f).fieldnames
    require(fields is not None and set(ROW_COLUMNS)<=set(fields),'Saved456 common-row CSV header ABI before any model FW')
    require(pin(METRIC_SOURCE)['sha256']==METRIC_SHA and pin(SENSOR_SOURCE)['sha256']==SENSOR_SHA,'Frozen cohort/state and cached15 arithmetic')
    required=required_sources(p)
    for v in required.values():require(any(same(v,w) for w in p['source_pins'].values()),'Current runtime source membership by samefile+SHA')
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent==ROOT/'outputs' and out.name.startswith(PREFIX) and not out.exists()
            and ('output' not in p or out==Path(p['output']).resolve()),'Fresh exact evaluation output')
    return {'protocol':p,'protocol_pin':pin(protocol_path),'training':training,'result':result,'health':health,'manifest':manifest,
            'reference':reference,'reference_protocol':refp,'description':description,'output':out,'sources':required,'actual_model_loads_FW':0}


def declaration():
    return {'schema':'native_public_only_common_eval_declaration:1','protocol_schema':SCHEMA,'role':ROLE,'input_aliases':sorted(INPUTS),
            'budget':BUDGET,'source_pins':{k:pin(v) for k,v in OWN.items()},'old_three_role_FW_repeated':False}


def pure_module(path,expected,name):
    require(pin(path)['sha256']==expected,'Frozen metric source')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def cached_summary(sensor,scores,indices,np):
    s=sensor.summary(scores,indices,np);f=s['confusion_per_feature'];q=s['confusion_total']
    s['positive_recall']=q['TP']/(q['TP']+q['FN']) if q['TP']+q['FN'] else None
    s['positive_precision']=q['TP']/(q['TP']+q['FP']) if q['TP']+q['FP'] else None
    s['negative_specificity']=q['TN']/(q['TN']+q['FP']) if q['TN']+q['FP'] else None
    s['per_feature_positive_recall']=[tp/(tp+fn) if tp+fn else None for tp,fn in zip(f['TP'],f['FN'])]
    s['per_feature_positive_precision']=[tp/(tp+fp) if tp+fp else None for tp,fp in zip(f['TP'],f['FP'])]
    return s


def saved_common_statistics(pred,delta,scale,logits,target,target15,pair_rows,identities,np):
    common=pure_module(METRIC_SOURCE,METRIC_SHA,'native_only_cohort_math')
    sensor=pure_module(SENSOR_SOURCE,SENSOR_SHA,'native_only_cached15_math')
    d=delta.astype('f8');error=pred.astype('f8')-target.astype('f8');s=scale.astype('f8')
    scores=sensor.score_matrix(logits,target15,np);statistics={};row_metrics=[]
    rawblocks={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),'R_raw9':(43,52,'raw matrix entry'),'v':(52,55,'m/s'),'omega':(55,58,'rad/s')}
    for name,mask in common.cohorts(identities):
        cs=cached_summary(sensor,scores,np.flatnonzero(mask),np)
        perfeature=common.vector_stats(d[mask],'Mixed units: per-feature values only')
        state=float(np.mean((d[mask]/s)**2))
        statistics[name]={'groups':int(mask.sum()),'native52_scaled_MSE_F64_of_saved_FP32_delta':state,
            'state_native_delta':{k:common.vector_stats(d[mask,a:b],unit) for k,(a,b,unit) in common.BLOCKS.items()},
            'raw58_difference':{k:common.vector_stats(error[mask,a:b],unit) for k,(a,b,unit) in rawblocks.items()},
            'raw_R9_difference':common.vector_stats(error[mask,43:52],'raw matrix entry; vector_RMS is Frobenius_RMS'),
            'native52_per_feature':{k:v for k,v in perfeature.items() if k.startswith('per_feature_') or k=='unit'},
            'cached_self15':cs,'native_pair_F64_rescore':state+.1*cs['BCE_sum15_mean_rows_F64'],
            'captured_native_pair_row_FP32_mean_F64':float(pair_rows[mask].astype('f8').mean())}
    for j,identity in enumerate(identities):
        cs=cached_summary(sensor,scores,[j],np);conf=cs['confusion_total']
        row={'role':NEW_ROLE,'row_id':j,'path':identity['path'],'k':identity['k'],'mode':identity['mode'],'group_id':identity['group_id'],
             'native52_scaled_MSE_F64':float(np.mean((d[j]/s)**2)),'cached15_BCE_mean_F64':cs['BCE_mean_entry_F64'],
             'cached15_Brier_F64':cs['Brier_mean_entry_F64'],'cached15_pred_intensity':cs['pred_intensity_mean'],
             'cached15_true_intensity':cs['target_intensity_mean'],'captured_native_pair_row_FP32':float(pair_rows[j]),
             **{'cached15_'+k:conf[k] for k in ('TP','FP','FN','TN')}}
        row['native_pair_F64_rescore']=row['native52_scaled_MSE_F64']+.1*cs['BCE_sum15_mean_rows_F64']
        row.update({k+'_native_delta_vector_norm':float(np.linalg.norm(d[j,a:b])) for k,(a,b,_) in common.BLOCKS.items()})
        row_metrics.append(row)
    return statistics,row_metrics


def write_csv(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run_evaluation(protocol_path,output):
    started=time.perf_counter();m=metadata_preflight(protocol_path,output);p=validate_protocol(m['protocol'],True);out=m['output'];out.mkdir()
    c._write_json(out/'protocol.json',p);c._write_json(out/'source_inventory.json',m['sources'])
    with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name,v in m['sources'].items():z.write(checked(v),name)
    counts={k:0 for k in BUDGET};phase='setup'
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import numpy as np
        import torch
        from . import component_H8_donor105_v1 as h8
        from . import boundary_impulse_component105_v1 as component
        from dex_supermimic.rl.dex_supermimic.state import difference
        torch.set_num_threads(4);before=c._rng(torch,np);h8.cf.save_raw_rng(out,'before',before,np)
        refi=m['reference_protocol']['inputs'];rp=m['reference']['outputs']
        initial=torch.load(checked(refi['original_initial_component']),map_location='cpu',weights_only=False);counts['trusted_CPU_torch_loads']+=1
        payload=torch.load(checked(m['manifest']['single_checkpoint']),map_location='cpu',weights_only=False);counts['trusted_CPU_torch_loads']+=1
        require(payload['schema']==nt.ARTIFACT_SCHEMA and payload['manifest_schema']==nt.MANIFEST_SCHEMA and payload['arm']==nt.ARM
                and payload['iteration']==1000 and payload['checkpoint_tag']=='AFTER_OPTIMIZER_I','New native-only AFTER1000 artifact')
        require(same(payload['protocol'],p['inputs']['native_training_protocol']) and same(payload['recipe'],m['training']['recipe']),'Exact native-only artifact source')
        for k in nt.FIT_DIGEST_KEYS:require(payload[k]==m['result'][k]==m['reference'][k],'Fixed fit digest metadata, no new label cohort')
        model=h8.make_initial_eval_component(initial,read(checked(refi['initial_validation'])),torch);counts['native_WM_constructors']+=1
        weights=payload['WM_state'];expected=model.state_dict()
        require(set(weights)==set(expected) and all(isinstance(weights[k],torch.Tensor) and weights[k].device.type=='cpu' and weights[k].shape==v.shape and weights[k].dtype==v.dtype and bool(torch.isfinite(weights[k]).all()) for k,v in expected.items()),'Strict raw native state shape/dtype/finite')
        norm=c._norm_state(model)
        require(all(component.bit_equal(weights[k],v) for k,v in norm.items()) and payload['normalizer_host_configuration']==c._norm_config(model)==initial['normalizer_host_configuration'],'Frozen native normalizers incl F64 counts and host fields')
        model.load_state_dict(weights,strict=True);model.eval().requires_grad_(False)
        require(len(list(model.parameters()))==14 and all(not q.requires_grad and q.grad is None for q in model.parameters()),'Raw native14 frozen; no auxiliary construction')
        require(c._tensor_digest(model.state_dict(),torch)==c._tensor_digest(weights,torch),'Loaded actual0301 WM byte identity')
        with np.load(checked(rp['prepared_inputs_and_context.npz']),allow_pickle=False) as z:data={k:z[k].copy() for k in PREPARED_KEYS}
        ids=read(checked(rp['group152_identities.json']));joined=read(checked(refi['cached_sensor_labels']))
        labels=ev.trainer.cached_label_rows(ids,joined)
        ev.trainer.validate_target_identity_arrays(ids,data['source_group_id152'],data['source_substep_rows152x4'],np)
        x,u,target=(torch.from_numpy(data[k]) for k in ('x58_FP32','u20_FP32','target58_FP32'));xi=initial['physical_xi46'];scale=initial['frozen_s52']
        y15=torch.from_numpy(data['cached_self15_target_FP32'])
        require(component.bit_equal(xi,torch.from_numpy(data['physical_xi46_FP32'])) and component.bit_equal(scale,torch.from_numpy(data['effective_s52_FP32']))
                and component.bit_equal(y15,torch.tensor(labels,dtype=torch.float32)),'Exact qualified prepared xi/effective52/cached15 source')
        require(component.bit_equal(torch.maximum(model.delta_norm.std(),model.error_floor),scale),'Frozen effective native52 formula')
        dynamics=torch.cat((x,xi[None].expand(152,-1),torch.zeros((152,174),dtype=torch.float32)),-1)
        input_digest=c._tensor_digest({'x':x,'u':u,'dynamics':dynamics,'target_outside_forward':target},torch)
        require(input_digest==payload['fit_input_digest'] and c._tensor_digest({'cached_self15_target_FP32':y15},torch)==payload['fit_self15_label_digest'],'Independent actual x/u/dynamics/target and cached15 byte digests')
        state_before=c._tensor_digest(model.state_dict(),torch);host_before=c._norm_config(model)
        phase='one_native_B152_forward'
        with torch.inference_mode():
            counts['native_FW_attempted']+=1;pred,logits=model(x,u,dynamics);counts['native_FW_returned']+=1;counts['predicted_rows']+=152
            component.tensor_check(pred,(152,58),torch.float32,'native-only prediction58');component.tensor_check(logits,(152,15),torch.float32,'native-only logits15')
            delta=difference(pred,target);counts['native_difference_calls']+=1
            recipe=nt.fixed_recipe();chart=c.chart_guard(pred,target,delta,recipe['native_log_chart_margin_rad'],torch)
            losses,operands=nt.native_only_losses(delta,scale,logits,y15,recipe,torch)
            probability=torch.sigmoid(logits);rate=-probability.sum(-1);penalty=rate*torch.tensor(.02,dtype=torch.float32)
        np.savez_compressed(out/'native52_self15_only_forward152.npz',prediction58_FP32=pred.numpy(),self15_logits_FP32=logits.numpy(),attempted_mask=np.ones(152,bool),completed_mask=np.ones(152,bool))
        np.savez_compressed(out/'native52_self15_only_common_operands.npz',**{k:v.numpy() for k,v in operands.items()},**{k:v.numpy() for k,v in losses.items()},self15_probability15_FP32=probability.numpy(),self15_penalty_rate_FP32=rate.numpy(),self15_penalty_component_FP32=penalty.numpy())
        np.savez_compressed(out/'prepared_public_inputs152.npz',**data)
        (out/'group152_identities.json').write_bytes(checked(rp['group152_identities.json']).read_bytes())
        statistics,newrows=saved_common_statistics(pred.numpy(),delta.numpy(),scale.numpy(),logits.numpy(),target.numpy(),y15.numpy(),operands['native_pair_per_row_FP32'].numpy(),ids,np)
        combined={role:{name:{k:v for k,v in s.items() if k in COMMON_KEYS} for name,s in m['description']['statistics'][role].items()} for role in OLD_ROLES}
        combined[NEW_ROLE]=statistics
        with checked(m['description']['outputs']['row_CSV']).open(encoding='utf-8',newline='') as f:
            original_rows=list(csv.DictReader(f))
        require(len(original_rows)==456,'Reuse all original456 row metrics, no old-model FW')
        columns=list(newrows[0]);require(set(columns)==set(ROW_COLUMNS),'Fixed native-only common row ABI');oldrows=[{k:r[k] for k in columns} for r in original_rows]
        require(len(combined)==4 and all(len(v)==13 for v in combined.values()),'Allfour roles/all13 overlapping cohorts')
        write_csv(out/'all608_common_model_row_metrics.csv',oldrows+newrows)
        c._write_json(out/'all52_common_model_cohort_statistics.json',combined)
        flat=[]
        for role in (*OLD_ROLES,NEW_ROLE):
            for name,s in combined[role].items():
                cs=s['cached_self15']
                flat.append({'role':role,'cohort':name,'groups':s['groups'],'native52_scaled_MSE_F64':s['native52_scaled_MSE_F64_of_saved_FP32_delta'],
                    'cached15_BCE_mean_F64':cs['BCE_mean_entry_F64'],'cached15_Brier_F64':cs['Brier_mean_entry_F64'],
                    'native_pair_F64_rescore':s['native_pair_F64_rescore'],'cached15_recall':cs['positive_recall'],'cached15_precision':cs['positive_precision'],
                    'Pen_proxy_error_mean_F64':cs['Pen_proxy_penalty_error_F64']['mean']})
        write_csv(out/'all52_common_model_cohort_metrics.csv',flat)
        after=c._rng(torch,np);h8.cf.save_raw_rng(out,'after',after,np)
        checks={'exact_one_new_native_FW_two_CPU_loads':counts==BUDGET,
            'raw_WM_norm_host_parameters_unchanged':state_before==c._tensor_digest(model.state_dict(),torch) and host_before==c._norm_config(model),
            'raw_input_bytes_unchanged':input_digest==c._tensor_digest({'x':x,'u':u,'dynamics':dynamics,'target_outside_forward':target},torch),
            'raw_rng_unchanged_without_manual_restore':all(ev.parity.raw_rng_equality(before,after,np,torch).values()),
            'no_gradients_or_GPU':not torch.cuda.is_initialized() and all(q.grad is None and not q.requires_grad for q in model.parameters())}
        require(all(checks.values()),'One-FW budget and immutable inference evidence')
        captured={role:{k:m['description']['captured_global_losses'][role][k] for k in ('L_state_FP32','L_self15_sum_FP32','L_native_pair_FP32')} for role in OLD_ROLES}
        captured[NEW_ROLE]={k:float(v) for k,v in losses.items()}
        report={'schema':'full151105_native_public_only_common_evaluation_CPU_analysis:1','status':'COMPLETED_ONE_NEW_NATIVE_ONLY_FW_ALL4_SAME152_COMMON_METRICS_NOT_GENERALIZATION',
            'protocol':m['protocol_pin'],'input_pins':p['inputs'],'counts':counts,'checks':checks,'passed':len(checks),'total':len(checks),
            'native_only_checkpoint':m['manifest']['single_checkpoint'],'native_only_health':p['inputs']['native_final_saved_health'],
            'old_role_reference_description':p['inputs']['reference_description_report'],'all_roles':[*OLD_ROLES,NEW_ROLE],'cohort_count_each':13,
            'captured_common_three_losses_FP32':captured,'native_only_chart':chart,'actual_fit_input_digest':input_digest,
            'comparison_scope':'Only same state/self15/native_pair plus separate per-unit errors/cache15 metrics. No auxiliary/public/total, winner gate or lambda selection. Existing three roles are read-only saved reference metrics; no repeated old FW.',
            'phase_scope':'Known fitted old152 at phases0/49/99/149; target state POST20ms versus cached self15 lastsolver~15ms. No fresh24/74/124/148, held-out/ego generalization, fullhidden/Jacobian or Actor/task qualification.',
            'old5_label_digest_scope':'Only x/u/dynamics/target and cached15 digests reconstructed; old5/combined strings delegated to final health and pinned parent provenance. No J/K/r_saved values read or losses computed.',
            'precision_scope':'Raw CPU FP32 ampFalse inference; native D52 and native row reductions captured. F64 stable cached scoring and physical-unit errors named separately; Pen proxy uses rewarddt .02, not full reward or exact native4h clock.',
            'elapsed_seconds':time.perf_counter()-started,'source_pins':m['sources'],'accuracy_ego_policy_holding_physical_gradient_qualified':False}
        report['outputs']={f.name:pin(f) for f in out.iterdir() if f.is_file() and f.name!='report.json'}
        c._write_json(out/'report.json',report)
        print(__import__('json').dumps({'status':report['status'],'report':pin(out/'report.json'),'counts':counts,'elapsed_seconds':report['elapsed_seconds']}),flush=True)
        return report
    except BaseException as error:
        if not (out/'failure.json').exists():c._write_json(out/'failure.json',{'phase':phase,'error':repr(error),'counts':counts,'automatic_retry':False})
        raise
