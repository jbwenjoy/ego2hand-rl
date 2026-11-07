"""SOURCE DRAFT: fixed public15 models, common-visible fresh endpoints/uVJP.

No native/model backend is imported by metadata/default. Actual Root execution
is separate. Fresh snapshot8 repeat19 is an input DESIGN, not measured PRE152.
Primary controlled POST20 object score excludes self15 and every other reward.
"""
from __future__ import annotations
import csv,json,os,sys,time,zipfile
from pathlib import Path
from . import boundary_public_self15_final_evaluation105_v1 as ev
from . import component_task_gradient105_v1 as tg

c=ev.c;cf=tg.cf;ROOT=c.ROOT
require=c.require;read=c.read;pin=c.pin;checked=c.checked;same=c.same
SCHEMA='full151105_public_self15_fresh_gradient_CPU_protocol:1'
ROLE='exploratory_common_visible_fresh_endpoints_and_object_score_uVJP'
PREFIX='full151105_public_self15_fresh_gradient_CPU_'
ROLES=ev.ROLES;ARMS=ev.ARMS;PHASES=(24,74,124,148)
CONTRACT=ROOT/'docs/public_self15_fresh_gradient_contract_v1.md'
CONTRACT_SHA='dbdc02123ed5c7cf814d96da04605d7f2a6979ac1adf3b5aeafd24c9a840be15'
PREP=ROOT/'outputs/full151105_public_self15_fresh_gradient_preparation_CPU_v1'
INPUTS={'native_evaluation_protocol','native_evaluation_report','native_evaluation_saved12','physical_context_buffers',
        'fresh_config','fresh_report','fresh_arrays','fresh_branch_index','fresh_snapshot_records','fresh_engineering_saved540',
        'prior_task_gradient_protocol','prior_task_gradient_report','prior_task_gradient_saved27'}
OWN={'consumer':ROOT/'src/full151_gradient/public_self15_fresh_gradient105_v1.py',
     'CLI':ROOT/'scripts/diagnose_full151_public_self15_fresh_gradient_cpu_v1.py',
     'checker':ROOT/'scripts/check_full151_public_self15_fresh_gradient_cpu_v1.py'}
DEPS={'new_family_eval':ROOT/'src/full151_gradient/boundary_public_self15_final_evaluation105_v1.py',
      'old_task_math':ROOT/'src/full151_gradient/component_task_gradient105_v1.py',
      'score_source':ROOT/'src/full151_gradient/serial_probe105_calibration_v1.py',
      'old_confirmation':ROOT/'src/full151_gradient/component_confirmation105_v1.py',
      'old_common_pre':ROOT/'src/full151_gradient/response_calibration105_v1.py',
      'component':ROOT/'src/full151_gradient/boundary_impulse_component105_v1.py',
      'initial_constructor':ROOT/'src/full151_gradient/component_H8_donor105_v1.py',
      'parity':ROOT/'src/full151_gradient/boundary_component_initial_parity105_v1.py'}
FIXED_CHECKPOINTS={'unwrapped_baseline':'ea051225b7279fc69fbd97725eb15462faaf4b3a6706881f8fe4a05b658a14bf',
 'shared_public_losses':'6cac9913bc943113b862c0b5cd92e884e904821269a5f4193a37e829324fc547',
 'shared_public_plus_cross_target_consistency':'78cd4584adc456dd63cd1bb7013e2f931802b85c1efb29462320f67bf38620bd'}
SCORE_CONTRACT={'function':'serial_probe105_calibration_v1.controlled_object_score_FP32','terms':'2_position_plus_8_shaft_endpoints',
 'state_phase':'matched_POST20','goal_phase':'k','VJP_reduction':'score.sum()','includes_self15':False,'includes_other_reward':False}
BUDGET={'own_trusted_torch_loads':3,'native_WM_constructors':1,'raw_deepcopies':2,'native_FW_attempted':6,'native_FW_returned':6,
        'endpoint_predictions':456,'center_predictions':24,'aux_head_FW':4,'uVJP_calls':3,'native_difference_calls':7,
        'optimizer_Actor_119f_MJB_Env_physics_GPU':0}

def validate_protocol(p,authorized=False):
    require(type(p) is dict and p.get('schema')==SCHEMA and p.get('role')==ROLE,'New common-visible exploratory namespace')
    for key in ('CPU_run_authorized','GPU_run_authorized','training_authorized','Env_or_physics_authorized'):
        require(type(p.get(key)) is bool,'Explicit authorization '+key)
    require(p['GPU_run_authorized'] is p['training_authorized'] is p['Env_or_physics_authorized'] is False,'No GPU/training/environment')
    require(type(p.get('cpu_threads')) is int and p['cpu_threads']==4 and p.get('output_prefix')==PREFIX,'CPU4/fresh output route')
    require(p.get('model_roles')==list(ROLES) and p.get('checkpoint_iteration')==1000 and p.get('test_phases')==list(PHASES),'Fixed initial/two AFTER1000/new phases')
    require(p.get('score_contract')==SCORE_CONTRACT and p.get('expected_counts')==BUDGET,'Fixed object-only score and six-forward/three-uVJP budget')
    require(p.get('comparison_modes')==tg.MODES and p.get('physical_floor_rule')==tg.FLOOR_RULE,'Unchanged original physical floor/two conventions')
    require(p.get('endpoint_input_scope')=='SNAPSHOT8_REPEAT19_COMMON_VISIBLE_DESIGN_NOT_ACTUAL_PRE152'
        and p.get('fresh_impulse_or_consistency_loss') is False,'No fresh PRE58 or J/K/r_saved qualification')
    require(type(p.get('inputs')) is dict and set(p['inputs'])==INPUTS and type(p.get('source_pins')) is dict and p['source_pins'],'Exact13 aliases/current source closure')
    require(CONTRACT_SHA is not None and pin(CONTRACT)['sha256']==CONTRACT_SHA and same(p['source_contract'],pin(CONTRACT)),'Frozen independent scientific contract required')
    if authorized:require(p['CPU_run_authorized'],'Root separate actual execution authorization required')
    return p

def declaration():
    return {'schema':SCHEMA,'role':ROLE,'input_aliases':sorted(INPUTS),'model_roles':list(ROLES),'test_phases':list(PHASES),
        'output_prefix':PREFIX,'cpu_threads':4,'score_contract':SCORE_CONTRACT,'expected_counts':BUDGET,'physical_floor_rule':tg.FLOOR_RULE,
        'comparison_modes':tg.MODES,'endpoint_input_scope':'SNAPSHOT8_REPEAT19_COMMON_VISIBLE_DESIGN_NOT_ACTUAL_PRE152',
        'fresh_impulse_or_consistency_loss':False,'actual_NPZ_model_FW_AD_reads':0}

def metadata_preflight(protocol_path,output=None):
    cf.no_backend();p=validate_protocol(read(protocol_path));i=p['inputs']
    for value in i.values():checked(value)
    parent=ev.metadata_preflight(checked(i['native_evaluation_protocol']))
    report=read(checked(i['native_evaluation_report']));review=read(checked(i['native_evaluation_saved12']));c.allchecks(report);c.allchecks(review)
    require(report['schema']=='full151105_boundary_public_self15_final_evaluation_CPU_analysis:1' and report['status']=='COMPLETED_FIXED_INITIAL_AND_TWO_PUBLIC_SELF15_AFTER1000_EVAL_NOT_CONFIRMATION','Completed new family fixed final evaluation')
    require(review['schema']=='full151105_boundary_public_self15_final_prediction_saved_independent:1' and review['status']=='PASS_SAVED_ALL3_ALL152_PUBLIC_SELF15_FIXED_FINAL_PREDICTIONS'
        and review['engineering_qualified'] is True and same(review['producer_report'],i['native_evaluation_report']) and same(review['protocol'],i['native_evaluation_protocol']),'Positive actual parent saved12')
    require(same(report['protocol'],i['native_evaluation_protocol']) and checked(i['native_evaluation_report']).parent.samefile(Path(review['run_dir'])),'Parent report/protocol/run exact')
    require(same(report['outputs']['physical_context_buffers.npz'],i['physical_context_buffers']) and checked(i['physical_context_buffers']).samefile(checked(i['native_evaluation_report']).parent/'physical_context_buffers.npz'),'Only pinned original context buffers, no impulse targets')
    pi=parent['protocol']['inputs'];initial_pin=pi['original_initial_component'];checkpoint_pins={ROLES[0]:initial_pin,**parent['pair']['checkpoints']}
    require(all(v['sha256']==FIXED_CHECKPOINTS[r] for r,v in checkpoint_pins.items()),'Exactly ea051 initial and new public15 6cac/78cd finals')
    fresh=read(checked(i['fresh_engineering_saved540']));c.allchecks(fresh)
    require(fresh['schema']=='full151105_fresh_response_engineering_saved_independent:1' and fresh['status']=='PASS_FRESH_RESPONSE_ENGINEERING_SEALED_NO_MODEL_METRICS'
        and same(fresh['producer_report'],i['fresh_report']) and same(fresh['config'],i['fresh_config']),'Actual fresh540 producer/config binding')
    require(fresh['completed_branches']==152 and fresh['completed_inventory_snapshots']==8 and fresh['episode_step_returned']==1808
        and fresh['baseline_all_passed'] is True and fresh['capacity_qualified'] is True,'All fixed fresh branches/baselines/clock/capacity')
    require(fresh['branch_PRE_full58_direct_capture'] is fresh['branch_PRE_full58_recomputed'] is fresh['branch_PRE_full58_qualified'] is False
        and fresh['whole_hidden_state_qualified'] is fresh['physical_gradient_qualified'] is False,'Do not upgrade source-backed restored primitives to PRE58/hidden/Jac')
    oldp=read(checked(i['prior_task_gradient_protocol']));oldr=read(checked(i['prior_task_gradient_report']));olds=read(checked(i['prior_task_gradient_saved27']));c.allchecks(olds)
    require(oldp['schema']==tg.SCHEMA,'Pinned original task protocol namespace')
    require(oldr['schema']=='full151105_component_task_gradient_CPU_analysis:1' and oldr['status']=='COMPLETED_EXPLORATORY_TWO_COMPONENT_TASK_SCORE_GRADIENT_NOT_CONFIRMATION'
        and olds['schema']=='full151105_component_task_gradient_saved_independent:1' and olds['status']=='PASS_SAVED_ALL32_ALL64_TASK_GRADIENT_ARITHMETIC'
        and olds['engineering_qualified'] is True and same(oldr['protocol'],i['prior_task_gradient_protocol']) and same(olds['producer_report'],i['prior_task_gradient_report'])
        and same(olds['protocol'],i['prior_task_gradient_protocol']),'Original fixed physical-floor/task arithmetic condition report')
    for alias,old_alias in (('fresh_config','test_full_config'),('fresh_report','test_full_report'),('fresh_arrays','test_full_arrays'),('fresh_branch_index','test_branch_index'),('fresh_snapshot_records','test_snapshot_records'),('fresh_engineering_saved540','test_full_engineering_review')):
        require(same(i[alias],oldp['inputs'][old_alias]) and same(i[alias],oldr['input_pins'][old_alias]),'Original fresh physical input identity '+alias)
    require(oldp['test_phases']==list(PHASES) and oldp['physical_floor_rule']==tg.FLOOR_RULE and oldp['comparison_modes']==tg.MODES
        and olds['physical_group_counts']=={'STABLE_NONFLAT':6,'FLAT_OR_BELOW_NOISE':25,'NONFLAT_UNSTABLE':1},'Fixed all32 shared physical groups; no model selection')
    for path in [*OWN.values(),*DEPS.values(),CONTRACT]:require(any(same(pin(path),v) for v in p['source_pins'].values()),'Current critical source membership '+str(path))
    sources=c.canonical_sources([*parent['sources'].values(),*p['source_pins'].values(),*i.values(),*checkpoint_pins.values(),pin(CONTRACT)])
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent.samefile(ROOT/'outputs') and out.name.startswith(PREFIX) and not out.exists(),'Fresh output only')
    cf.no_backend();return {'protocol':p,'protocol_pin':pin(Path(protocol_path).resolve()),'output':out,'parent':parent,'parent_report':report,'sources':sources,
        'checkpoint_pins':checkpoint_pins,'physical_group_counts':olds['physical_group_counts'],'actual_NPZ_model_FW_AD_reads':0}

def ready_for_run():
    ready=read(PREP/'READY.json');proof=read(checked(ready['cpu_report']));c.allchecks(proof)
    require(ready['schema']=='full151105_public_self15_fresh_gradient_CPU_READY:1' and proof['schema']=='full151105_public_self15_fresh_gradient_CPU_contract:1'
        and proof['status']=='PASS_SYNTHETIC_ONLY_FRESH_DESIGN_OBJECT_SCORE_AND_BUDGET','Executed focused proof only')
    require(ready['cpu_checks']=={'passed':proof['passed'],'total':proof['total']} and proof['actual_component_loads_fresh_NPZ_physics']==0,'Focused synthetic scope')
    for key,path in OWN.items():require(same(ready['source_pins'][key],pin(path)) and same(proof['source_pins'][key],pin(path)),'Current source/proof binding')
    require(same(ready['source_contract'],pin(CONTRACT)),'Current contract/proof binding');return pin(PREP/'READY.json'),ready['cpu_report']

def context_from_saved(values,initial,component,torch):
    """One pinned old fixed context; never fresh per-row dynamics labels."""
    specs={'mass_actual_FP32':((),torch.float32),'I_body_F64':((3,3),torch.float64),'gravity_palm_F64':((3,),torch.float64),'DeltaT_F64':((),torch.float64),
        'palm_rot_FP32':((3,3),torch.float32),'effective_s52_FP32':((52,),torch.float32),'physical_xi46_FP32':((46,),torch.float32),'I_body_antisymmetry_maxabs_F64':((),torch.float64)}
    require(set(values)==set(specs),'Exact pinned original context buffer set')
    for name,(shape,dtype) in specs.items():component.tensor_check(values[name],shape,dtype,name)
    require(component.bit_equal(values['effective_s52_FP32'],initial['frozen_s52']) and component.bit_equal(values['physical_xi46_FP32'],initial['physical_xi46']),'Same initial frozen scales/xi')
    ctx=component.FixedPhysicalContext(mass=values['mass_actual_FP32'],Ib=values['I_body_F64'],g=values['gravity_palm_F64'],dt=values['DeltaT_F64'],P=values['palm_rot_FP32'],
        s52=values['effective_s52_FP32'],xi=values['physical_xi46_FP32'],row_count=1,row_guards={'pinned_original_parent_context_only_not_fresh_row_enrollment':True})
    require(all(component.bit_equal(v,ctx.state_dict()[k]) for k,v in values.items()),'Reconstructed context byte identity; no averaging/repair');return ctx

def call_model(model,x,u,dynamics,role,counts,torch,kind):
    n=len(x);require(n in (8,152) and kind in ('endpoint','center'),'Only B152 and B8 calls')
    counts['native_FW_attempted']+=1
    out=model(x,u,dynamics);counts['native_FW_returned']+=1
    if role==ROLES[0]:require(len(out)==2,'Unwrapped native output pair');pred,self15=out
    else:
        require(len(out)==4,'New own wrapper output native+aux');pred,self15,z,mode=out;counts['aux_head_FW']+=1
        require(z.shape==(n,2,6) and mode.shape==(n,2) and torch.isfinite(z).all() and torch.isfinite(mode).all(),'Finite same-call unused auxiliary outputs')
    require(pred.shape==(n,58) and self15.shape==(n,15) and pred.dtype==self15.dtype==torch.float32 and torch.isfinite(pred).all() and torch.isfinite(self15).all(),'Finite native prediction/logits')
    counts[kind+'_predictions']+=n;return pred,self15

def endpoint_cohorts(identities,np):
    result={'ALL':np.ones(152,bool),'A0':np.asarray([r['path']=='A0' for r in identities]),'A2':np.asarray([r['path']=='A2' for r in identities])}
    result.update({'snapshot_'+str(s):np.asarray([r['snapshot_id']==s for r in identities]) for s in range(8)})
    result.update({mode:np.asarray([r['type']==mode for r in identities]) for mode in ('baseline','direction')});return result

def cached15_statistics(logits,target,identities,np):
    require(logits.dtype==target.dtype==np.float32 and logits.shape==target.shape==(152,15) and np.isfinite(logits).all() and np.isin(target,[0,1]).all(),'Finite cached15 ABI')
    z=logits.astype(np.float64);y=target.astype(np.float64);bce=np.maximum(z,0)-z*y+np.log1p(np.exp(-abs(z)));prob=np.empty(z.shape,np.float64);pos=z>=0
    prob[pos]=1/(1+np.exp(-z[pos]));e=np.exp(z[~pos]);prob[~pos]=e/(1+e);prediction=z>0;truth=y>0
    rows={}
    for name,mask in endpoint_cohorts(identities,np).items():
        pp=prediction[mask];tt=truth[mask];tp=int((pp&tt).sum());fp=int((pp&~tt).sum());fn=int((~pp&tt).sum());tn=int((~pp&~tt).sum())
        perfeature=[]
        for feature in range(15):
            ppf=pp[:,feature];ttf=tt[:,feature];tpf=int((ppf&ttf).sum());fpf=int((ppf&~ttf).sum());fnf=int((~ppf&ttf).sum());tnf=int((~ppf&~ttf).sum())
            perfeature.append({'feature':feature,'rows':int(mask.sum()),'BCE_mean_F64':float(bce[mask,feature].mean()),'Brier_F64':float(np.mean((prob[mask,feature]-y[mask,feature])**2)),
                'TP':tpf,'FP':fpf,'FN':fnf,'TN':tnf,'tie_zero_negative':int((z[mask,feature]==0).sum()),
                'positive_recall':None if tpf+fnf==0 else tpf/(tpf+fnf),'positive_precision':None if tpf+fpf==0 else tpf/(tpf+fpf)})
        rows[name]={'rows':int(mask.sum()),'cached_sensor_labels':int(15*mask.sum()),'BCE_mean_elements_F64':float(bce[mask].mean()),'Brier_F64':float(np.mean((prob[mask]-y[mask])**2)),
            'TP':tp,'FP':fp,'FN':fn,'TN':tn,'tie_zero_negative':int((z[mask]==0).sum()),'positive_recall':None if tp+fn==0 else tp/(tp+fn),
            'positive_precision':None if tp+fp==0 else tp/(tp+fp),'perfeature15':perfeature,
            'predicted_soft_count_mean_F64':float(prob[mask].sum(-1).mean()),'cached_positive_count_mean_F64':float(y[mask].sum(-1).mean()),
            'predicted_penalty_component_mean_F64_reward_dt_0_02':float((-.02*prob[mask].sum(-1)).mean()),
            'cached_penalty_component_mean_F64_reward_dt_0_02':float((-.02*y[mask].sum(-1)).mean()),
            'self15_reward_proxy_error_mean_F64_dt_0_02':float((.02*(y[mask].sum(-1)-prob[mask].sum(-1))).mean())}
    return rows,{'probability15_F64':prob,'BCE_elements_F64':bce}

def native52_statistics(delta,np):
    """Native D(pred,target) coordinates, separate from raw58/R9 differences."""
    a=delta.astype(np.float64);require(a.shape==(152,52) and np.isfinite(a).all(),'Full finite native point52')
    blocks={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),'native_LogR':(43,46,'rad'),'v':(46,49,'m/s'),'omega':(49,52,'rad/s')}
    return {'perfeature_RMS':np.sqrt(np.mean(a*a,axis=0)).tolist(),'perfeature_MAE':np.mean(abs(a),axis=0).tolist(),
        'perfeature_maxabs':np.max(abs(a),axis=0).tolist(),'blocks':{name:{'unit':unit,'component_RMS':float(np.sqrt(np.mean(a[:,lo:hi]**2))),
        'MAE':float(np.mean(abs(a[:,lo:hi]))),'maxabs':float(np.max(abs(a[:,lo:hi])))} for name,(lo,hi,unit) in blocks.items()}}

def run_analysis(protocol_path,output):
    started=time.perf_counter();m=metadata_preflight(protocol_path,output);p=validate_protocol(m['protocol'],True);ready,proof=ready_for_run();i=p['inputs'];out=m['output'];out.mkdir()
    c._write_json(out/'protocol.json',p);c._write_json(out/'source_inventory.json',m['sources'])
    with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name,value in m['sources'].items():z.write(checked(value),name+'/'+checked(value).name)
    phase='imports';counts={key:0 for key in BUDGET};modules={}
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import numpy as np
        import torch
        torch.set_num_threads(4);require(not torch.cuda.is_initialized() and not torch.is_inference_mode_enabled(),'Fresh CPU4 process; input AD outside inference_mode')
        from . import boundary_impulse_component105_v1 as component
        from . import component_H8_donor105_v1 as h8
        from .serial_probe105_calibration_v1 import controlled_object_score_FP32
        from dex_supermimic.rl.dex_supermimic.state import difference
        parity=ev.parity;rng0=c._rng(torch,np);cf.save_raw_rng(out,'before',rng0,np)
        phase='fresh_saved_design'
        with np.load(checked(i['fresh_arrays']),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        data,identities,pairs=cf.prepare_test(a,read(checked(i['fresh_branch_index'])),read(checked(i['fresh_snapshot_records'])),np)
        baseline,groups,physical,arithmetic=tg.physical_descriptors(a,identities,pairs,np)
        c._write_json(out/'all152_design_identities.json',identities);tg.save_rows(out/'physical_groups32',groups);tg.save_rows(out/'physical_rows64',physical);c._write_json(out/'baseline8.json',baseline)
        require(a['branch_self15'].dtype==np.float32 and a['branch_self15'].shape==(152,15) and np.isin(a['branch_self15'],[0,1]).all(),'Source-delegated cached lastsolver15 labels, not POST20 geometry')
        phase='three_own_component_reads';initial=torch.load(checked(m['checkpoint_pins'][ROLES[0]]),map_location='cpu',weights_only=False);counts['own_trusted_torch_loads']+=1
        payloads={}
        for arm in ARMS:payloads[arm]=torch.load(checked(m['checkpoint_pins'][arm]),map_location='cpu',weights_only=False);counts['own_trusted_torch_loads']+=1
        scale=initial['frozen_s52'];xi=initial['physical_xi46']
        with np.load(checked(i['physical_context_buffers']),allow_pickle=False) as z:context={k:torch.from_numpy(np.array(z[k],copy=True)) for k in z.files}
        ctx=context_from_saved(context,initial,component,torch)
        raw=h8.make_initial_eval_component(initial,read(checked(m['parent']['protocol']['inputs']['initial_validation'])),torch);counts['native_WM_constructors']+=1
        modules,construction=component.make_matched_components(raw,ctx);counts['raw_deepcopies']=2
        expected_digests={key:m['parent']['training_result'][key] for key in ev.FIT_DIGEST_KEYS}
        for arm in ARMS:ev.load_final_into(modules[arm],payloads[arm],initial,m['parent']['protocol']['inputs']['training_protocol'],m['parent']['training'],expected_digests,torch)
        models={ROLES[0]:raw,**modules};states={};hosts={}
        for role,model in models.items():
            model.eval().requires_grad_(False);require(len(list(model.parameters()))==(14 if role==ROLES[0] else 16) and all(not q.requires_grad and q.grad is None for q in model.parameters()),'Actual native14/final16 frozen, no parameter AD')
            states[role]=component.state_digest(model);hosts[role]=c._norm_config(raw if role==ROLES[0] else model.wm)
        storage,disjoint=parity.named_storage_inventory(models);require(disjoint,'Independent initial/final model parameter and buffer storage');c._write_json(out/'named_storage_inventory.json',storage)
        x,target,u=(torch.from_numpy(data[k]) for k in ('x152_FP32','target152_FP32','actual_u152_FP32'))
        mi,pl=(torch.from_numpy(data[k]) for k in ('minus_indices','plus_indices'));du=torch.from_numpy(data['actual_du_rad_F64'])
        dynamics=torch.cat((x,xi[None].expand(152,-1),x.new_zeros((152,174))),-1)
        x8=torch.from_numpy(a['snapshot_x'].copy());u8=torch.from_numpy(a['snapshot_u'].copy());gp=torch.from_numpy(a['goal_p'].copy());gR=torch.from_numpy(a['goal_R'].copy())
        memory={'x_design':x,'u_actual':u,'target':target,'xi':xi,'scale':scale,'x8':x8,'u8':u8,'goal_p':gp,'goal_R':gR};memory_before=c._tensor_digest(memory,torch)
        with torch.no_grad():target_common=difference(x,target);counts['native_difference_calls']+=1;target_chart=c.chart_guard(x,target,target_common,.01,torch)
        np.savez_compressed(out/'fresh_design_and_physical_inputs.npz',**data,snapshot_x8_FP32=a['snapshot_x'],snapshot_u8_FP32=a['snapshot_u'],goal_p8_FP32=a['goal_p'],goal_R8_FP32=a['goal_R'],
            target_cached_self15_FP32=a['branch_self15'],physical_xi46_FP32=xi.numpy(),effective_s52_FP32=scale.numpy(),target_common52_FP32=target_common.numpy(),**arithmetic)
        summaries={};model_checks={}
        for role,model in models.items():
            phase='endpoint_'+role;diagnostics={}
            with torch.no_grad():
                pred,logits=call_model(model,x,u,dynamics,role,counts,torch,'endpoint')
                point,pair,charts=c.common_pre_losses(pred,target,x,scale,target_common,mi,pl,du,difference,torch,diagnostics=diagnostics);counts['native_difference_calls']+=2
                point_rows=(diagnostics['point_delta52_FP32']/scale).square().mean(-1);rowmean=point_rows.mean()
            raw_error,state_stats=cf.state_statistics(pred.numpy(),data['target152_FP32'],np);self_stats,self_arrays=cached15_statistics(logits.numpy(),a['branch_self15'],identities,np)
            np.savez_compressed(out/(role+'_endpoint152.npz'),prediction58_FP32=pred.numpy(),self15_logits_FP32=logits.numpy(),native_point_D52_FP32=diagnostics['point_delta52_FP32'].numpy(),pred_common52_FP32=diagnostics['pred_common52_FP32'].numpy(),
                point_rows_FP32=point_rows.numpy(),raw58_signed_error_F64=raw_error,**self_arrays)
            phase='center_object_uVJP_'+role;uv=u8.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                require(not torch.is_inference_mode_enabled(),'Real uVJP may not execute in inference_mode')
                d8=torch.cat((x8,xi[None].expand(8,-1),x8.new_zeros((8,174))),-1)
                center,center_logits=call_model(model,x8,uv,d8,role,counts,torch,'center')
                terms,score=controlled_object_score_FP32(center,gp,gR)
                gradient,=torch.autograd.grad(score.sum(),uv);counts['uVJP_calls']+=1
            require(gradient.dtype==torch.float32 and gradient.shape==(8,20) and torch.isfinite(gradient).all() and torch.isfinite(terms).all() and torch.isfinite(score).all()
                and all(q.grad is None and not q.requires_grad for q in model.parameters()),'Finite object-only input gradient, no model grad')
            rows=tg.projected_rows(gradient.detach().numpy().copy(),physical,np);tg.save_rows(out/(role+'_object_gradient_comparison64'),rows)
            np.savez_compressed(out/(role+'_center8_object_uVJP.npz'),prediction58_FP32=center.detach().numpy(),self15_logits_FP32=center_logits.detach().numpy(),object_terms_FP32=terms.detach().numpy(),object_score_FP32=score.detach().numpy(),gradient_u8x20_FP32=gradient.detach().numpy())
            summaries[role]={'point52_old_confirmation_flatmean_FP32':float(point),'point52_newfamily_rowmean_FP32':float(rowmean),'common_pre_response64_MSE_F64':float(pair),
                'native52_point_statistics_all152':native52_statistics(diagnostics['point_delta52_FP32'].numpy(),np),
                'raw58_state_statistics_all152':state_stats,'cached_sensor15_all13_cohorts':self_stats,'task_object_gradient':tg.comparison_summary(rows,np),'charts':charts,
                'primary_score_excludes_self15_and_other_reward':True}
            model_checks[role]=states[role]==component.state_digest(model) and hosts[role]==c._norm_config(raw if role==ROLES[0] else model.wm)
            require(model_checks[role],'Native weights/aux/context/norm/host changed')
        after=c._rng(torch,np);cf.save_raw_rng(out,'after',after,np);rng_equal=parity.raw_rng_equality(rng0,after,np,torch)
        require(all(rng_equal.values()) and counts==BUDGET and c._tensor_digest(memory,torch)==memory_before,'Budget/raw RNG/input immutability; caller does not repair')
        require(all(module._cache is None and module.counts['forward_returned']==2 and module.counts['hook_calls']==2 for module in modules.values()),'Two completed head captures per final, no residual graph cache')
        for value in [*i.values(),*m['sources'].values(),*m['checkpoint_pins'].values()]:checked(value)
        require(same(pin(Path(protocol_path).resolve()),m['protocol_pin']),'Immutable protocol')
        report={'schema':'full151105_public_self15_fresh_gradient_CPU_analysis:1','status':'COMPLETED_COMMON_VISIBLE_FRESH_ENDPOINTS_AND_OBJECT_uVJP_NOT_PHYSICAL_JACOBIAN',
            'protocol':m['protocol_pin'],'source_contract':pin(CONTRACT),'input_pins':i,'checkpoint_pins':m['checkpoint_pins'],'cpu_ready':ready,'cpu_report':proof,'counts':counts,'summary':summaries,
            'raw_RNG_value_and_byte_equality':rng_equal,'model_state_and_norm_checks':model_checks,'construction_receipt':construction,'target_chart':target_chart,
            'physical_group_counts':{label:sum(row['label']==label for row in groups) for label in ('STABLE_NONFLAT','FLAT_OR_BELOW_NOISE','NONFLAT_UNSTABLE')},
            'input_scope':'fresh snapshot8 repeat19 common-visible input design; physical targets/actual controls retained. Not measured/qualified branch PRE152.',
            'self15_scope':'branch_self15 labels delegated to pinned fresh540 and worker raw_found->15 source; lastsolver~15ms, not POST20 geometry. No self15/penalty VJP.',
            'xi_scope':'Model xi46 is the original initial/context pinned buffer. Its identity to fresh physical provenance is delegated to pinned540/27/source; no fresh raw xi reread or new independent per-row context measurement in this job.',
            'primary_score_contract':SCORE_CONTRACT,'fresh_J_K_r_saved_targets_or_consistency_loss':False,'old_and_fresh_cohorts_merged':False,
            'whole_hidden_state_qualified':False,'physical_gradient_qualified':False,'Actor_or_CHORD_benefit_qualified':False,'new_blind_confirmation':False,
            'elapsed_seconds':time.perf_counter()-started}
        c._write_json(out/'report.json',report);c._write_json(out/'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        c._write_json(out/'failure.json',{'phase':phase,'error':repr(error),'counts':counts,'automatic_retry':False});raise
    finally:
        for module in modules.values():module.close()
