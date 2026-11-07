"""SOURCE DRAFT: exploratory final-component task-score u-gradient check.

Metadata only by default. Explicit Root authorization is required before three
component loads, two B8 forwards and two uVJPs. No training, new physics, task
confirmation upgrade, Actor gradient or new blind evidence is produced.
"""
from __future__ import annotations
import csv,json,os,sys,time,zipfile
from pathlib import Path
from . import component_confirmation105_v1 as cf
from . import response_calibration105_v1 as c

ROOT=c.ROOT;ARMS=c.ARMS;PHASES=(24,74,124,148);JOINTS=(0,4,8,12);EPS=(.002,.001)
SCHEMA='full151105_component_task_gradient_CPU_protocol:1'
ROLE='exploratory_not_new_blind_confirmation'
OUTPUT_PREFIX='full151105_component_task_gradient_CPU_'
PARENT_REPORT_SHA='82b88d40d26aa0a940b09777a40e741c62dfecf2c2e14ba5bbbe51040b100908'
INPUTS=cf.INPUTS|{'parent_confirmation_protocol','parent_confirmation_report','parent_confirmation_saved_review'}
FLOOR_RULE={'minimum_score':1e-5,'empirical_multiplier':10.,'two_eps_relative_disagreement_max':.5,'relative_slope_denominator_floor':1e-12,
 'classification_slope_denominator':'2*requested_epsilon_rad','classification_scope':'Shared physical32groups, both epsilon rows retained for both models; descriptive only'}
MODES=['actual_coordinate_secant','legacy_nominal_epsilon_effective_direction']
OWN={'consumer':ROOT/'src/full151_gradient/component_task_gradient105_v1.py','CLI':ROOT/'scripts/diagnose_full151_component_task_gradient_cpu_v1.py'}
DEPS={'component_constructor':ROOT/'src/full151_gradient/component_confirmation105_v1.py','calibration_source':ROOT/'src/full151_gradient/response_calibration105_v1.py',
 'score_source':ROOT/'src/full151_gradient/serial_probe105_calibration_v1.py','physical_floor_source':ROOT/'src/full151_gradient/analysis105_v1.py'}
require=c.require;read=c.read;pin=c.pin;checked=c.checked;same=c.same

def validate_protocol(p,require_authorized=False):
    require(type(p) is dict and p.get('schema')==SCHEMA and p.get('role')==ROLE,'Explicit exploratory task-gradient namespace')
    for key in ('execution_authorized','CPU_run_authorized','GPU_run_authorized','training_authorized','Env_or_physics_authorized'):
        require(type(p.get(key)) is bool,'Explicit boolean '+key)
    require(p['GPU_run_authorized'] is p['training_authorized'] is p['Env_or_physics_authorized'] is False,'CPU only; no training or physical environment')
    require(type(p.get('cpu_threads')) is int and p['cpu_threads']==4 and p.get('output_prefix')==OUTPUT_PREFIX,'Fixed CPU4/fresh namespace')
    require(p.get('arms')==list(ARMS) and type(p.get('checkpoint_iteration')) is int and p['checkpoint_iteration']==1000 and p.get('test_phases')==list(PHASES),'Same two final1000/eight fresh centers')
    require(p.get('architecture')==cf.ARCH and all(type(p['architecture'][k]) is type(v) for k,v in cf.ARCH.items()),'Unchanged native CPU FP32 architecture')
    require(p.get('comparison_modes')==MODES and p.get('physical_floor_rule')==FLOOR_RULE,'Frozen separate denominator modes/shared old physical floor rule')
    require(p.get('include_endpoint_model_secants') is False,'No additional endpoint forwards or optional endpoint-score study')
    require(type(p.get('inputs')) is dict and set(p['inputs'])==INPUTS,'Exact19 semantic inputs')
    require(type(p.get('source_pins')) is dict and p['source_pins'],'Actual current source closure')
    if require_authorized:require(p['execution_authorized'] and p['CPU_run_authorized'],'Root separate explicit exploratory CPU authorization required')
    return p

def declaration():
    cf.no_backend()
    return {'schema':'full151105_component_task_gradient_declaration:1','status':'SOURCE_DRAFT_NO_MODEL_OR_TEST_SCORE_EXECUTED','protocol_schema':SCHEMA,'role':ROLE,
     'input_aliases':sorted(INPUTS),'architecture':cf.ARCH,'arms':list(ARMS),'test_phases':list(PHASES),'cpu_threads':4,'comparison_modes':MODES,'physical_floor_rule':FLOOR_RULE,
     'expected_model_budget':{'trusted_component_loads':3,'native_WM_constructors':2,'WM_forwards':2,'center_predictions':16,'uVJP_calls':2},'source_pins':{**{k:pin(v) for k,v in OWN.items()},**{k:pin(v) for k,v in DEPS.items()}},'actual_model_FW_AD_physics':0}

def metadata_preflight(protocol_path,output=None):
    """Hash/JSON only; reuse all existing fixed-component release/load guards."""
    cf.no_backend();p=validate_protocol(read(protocol_path));i=p['inputs']
    for value in i.values():checked(value)
    parent=cf.metadata_preflight(checked(i['parent_confirmation_protocol']))
    for alias in cf.INPUTS:require(same(parent['protocol']['inputs'][alias],i[alias]),'Same fixed semantic parent input: '+alias)
    require(i['parent_confirmation_report']['sha256']==PARENT_REPORT_SHA,'Fixed prior tradeoff report; not new checkpoint/model selection')
    saved=read(checked(i['parent_confirmation_saved_review']));c.allchecks(saved)
    require(saved['schema']=='full151105_WM_component_confirmation_saved_independent:1' and saved['status']=='PASS_SAVED_ALL152_ALL64_COMPONENT_CONFIRMATION_ARITHMETIC' and saved['engineering_qualified'] is True,'Actual independent confirmation arithmetic review')
    require(same(saved['protocol'],i['parent_confirmation_protocol']) and same(saved['producer_report'],i['parent_confirmation_report']),'Parent saved-review route/source joins')
    report=read(checked(i['parent_confirmation_report']))
    require(report['schema']=='full151105_WM_component_confirmation_CPU_analysis:1' and report['status']=='COMPLETED_ALL152_ALL64_COMPONENT_CONFIRMATION_NOT_POLICY_BENEFIT' and same(report['protocol'],i['parent_confirmation_protocol']),'Completed original confirmation report')
    require(report['decision']==saved['decision'],'Original fixed ea22 decision unchanged')
    for flag in ('whole_hidden_state_qualified','physical_gradient_qualified'):require(report[flag] is saved[flag] is False,'No inherited hidden/physical-gradient upgrade')
    sources=c.canonical_sources(list(p['source_pins'].values())+list(parent['sources'].values()))
    for path in (*OWN.values(),*DEPS.values()):
        require(any(same(pin(path),v) for v in p['source_pins'].values()),'Current executed helper not in new protocol closure: '+str(path))
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent==ROOT/'outputs' and out.name.startswith(OUTPUT_PREFIX) and not out.exists(),'Fresh exact exploratory output')
    cf.no_backend()
    return {'status':'PASS_EXPLORATORY_TASK_GRADIENT_JSON_HASH_ONLY','protocol':p,'protocol_pin':pin(protocol_path),'sources':sources,'output':out,
      'parent_confirmation_decision':report['decision'],'test_array_values_decoded':0,'trusted_model_loads_FW_AD':0}

def physical_descriptors(a,identities,pairs,np):
    """All32 physical group classifications and all64 rows, shared by models.

    Only saved primary GPU scores form physical finite differences/floors. The
    saved CPU secondary terms stay descriptive and never replace the primary.
    """
    from . import analysis105_v1 as old
    for key,shape in (('goal_p',(8,3)),('goal_R',(8,3,3)),('branch_object_terms_k',(152,2)),('branch_object_terms_k_CPU_secondary',(152,2)),('branch_object_score_k',(152,)),('donor_object_score_k',(8,))):
        require(a[key].dtype==np.float32 and a[key].shape==shape and np.isfinite(a[key]).all(),'Original finite saved score/goal ABI '+key)
    sums=a['branch_object_terms_k'].sum(axis=-1,dtype=np.float32)
    require(np.array_equal(sums.view(np.uint32),a['branch_object_score_k'].view(np.uint32)),'Original GPU primary FP32 two-term subset sum')
    # Original same-x20 arithmetic bound is retained, not retuned from agreement.
    expected=np.stack([old.object_terms_f64(a['branch_x_next'][bid:bid+1],a['goal_p'][bid//19:bid//19+1],a['goal_R'][bid//19:bid//19+1])[0] for bid in range(152)])
    bounds=8*np.finfo(np.float32).eps*np.maximum(1.,np.abs(expected));term_delta=a['branch_object_terms_k'].astype(np.float64)-expected
    require(np.isfinite(expected).all() and (np.abs(term_delta)<=bounds).all(),'Original per-term same-post20 F64 arithmetic gate')
    baseline=old.baseline_summary(a);groups=[];rows=[]
    by={(p['snapshot_id'],p['j'],p['epsilon_index']):p for p in pairs};require(len(by)==64,'Exact full64 pair identities')
    for s in range(8):
        for j,q in enumerate(JOINTS):
            ids=[(s*19+3+8*ei+2*j,s*19+4+8*ei+2*j) for ei in range(2)]
            minus=np.asarray([a['branch_object_score_k'][m] for m,_ in ids],np.float32);plus=np.asarray([a['branch_object_score_k'][pl] for _,pl in ids],np.float32)
            cl=old.classify_pair(minus,plus,baseline[s]['baseline_range'],baseline[s]['baseline_donor_error'])
            group_id=s*4+j;groups.append({'group_id':group_id,'snapshot_id':s,'path':('A0','A2')[s//4],'k':PHASES[s%4],'j':j,'joint_index':q,**cl,'classification_scope':'Physical-only shared exploratory descriptor, not a Jacobian/noise certificate'})
            for ei,eps in enumerate(EPS):
                mi,pl=ids[ei];pair=by[(s,j,ei)];require(pair['minus_branch_id']==mi and pair['plus_branch_id']==pl,'Pair/source score order')
                du=a['branch_u'][pl].astype(np.float64)-a['branch_u'][mi].astype(np.float64);den=float(du[q]);require(np.isfinite(du).all() and den>=0,'Actual plus/minus order; no denominator repair')
                delta=float(np.float64(a['branch_object_score_k'][pl])-np.float64(a['branch_object_score_k'][mi]));defined=den!=0.
                rows.append({**pair,'group_id':group_id,'primary_GPU_score_minus_FP32':float(a['branch_object_score_k'][mi]),'primary_GPU_score_plus_FP32':float(a['branch_object_score_k'][pl]),
                  'delta_score_F64':delta,'actual_delta_u20_rad_F64':du.tolist(),'actual_coordinate_denominator_rad':den,'actual_coordinate_defined':defined,
                  'physical_actual_coordinate_secant':delta/den if defined else None,'physical_legacy_nominal_epsilon_slope':delta/(2*eps),
                  'legacy_d_eff_dimensionless':a['direction__d_eff'][s,j,ei].tolist(),'center_offset_rad':a['direction__center_offset'][s,j,ei].tolist(),
                  'clipped':bool(a['direction__clipped'][s,j,ei]),'asymmetric_about_baseline':bool(np.any(a['direction__center_offset'][s,j,ei]!=0)),
                  'physical_group_label':cl['label'],'stable_nonflat_shared_physical':cl['stable_nonflat'],'signal_floor_score':cl['signal_floor'],'relative_epsilon_disagreement':cl['relative_disagreement'],
                  'primary_GPU_terms_minus_FP32':a['branch_object_terms_k'][mi].tolist(),'primary_GPU_terms_plus_FP32':a['branch_object_terms_k'][pl].tolist(),
                  'CPU_secondary_terms_minus_FP32':a['branch_object_terms_k_CPU_secondary'][mi].tolist(),'CPU_secondary_terms_plus_FP32':a['branch_object_terms_k_CPU_secondary'][pl].tolist(),
                  'true_Jacobian_qualified':False})
    require(len(groups)==32 and len(rows)==64,'No outcome subset')
    return baseline,groups,rows,{'physical_terms_expected_F64':expected,'physical_terms_delta_F64':term_delta,'physical_terms_original_bound_F64':bounds,
      'CPU_secondary_score_sum_FP32':a['branch_object_terms_k_CPU_secondary'].sum(-1,dtype=np.float32)}

def projected_rows(gradient,physical,np):
    require(gradient.dtype==np.float32 and gradient.shape==(8,20) and np.isfinite(gradient).all(),'Finite original FP32 input-gradient8x20')
    rows=[]
    for r in physical:
        s=r['snapshot_id'];q=r['joint_index'];g=gradient[s].astype(np.float64);du=np.asarray(r['actual_delta_u20_rad_F64'],np.float64);den=r['actual_coordinate_denominator_rad']
        effective=float(g@np.asarray(r['legacy_d_eff_dimensionless'],np.float64));actual=float(g@du/den) if r['actual_coordinate_defined'] else None
        rows.append({**r,'WM_AD_coordinate_at_baseline_u0':float(g[q]),'WM_AD_actual_delta_projected_per_coordinate':actual,'WM_AD_legacy_effective_direction':effective,
          'actual_secant_minus_WM_AD':None if actual is None else r['physical_actual_coordinate_secant']-actual,
          'legacy_slope_minus_WM_AD_effective':r['physical_legacy_nominal_epsilon_slope']-effective,
          'linearization_scope':'Baseline-center local u derivative vs finite possibly asymmetric chord; no true physical Jacobian or Actor gradient'})
    return rows

def comparison_summary(rows,np):
    actual=[r for r in rows if r['actual_coordinate_defined']]
    def describe(selected,physical_key,model_key):
        if not selected:return {'reported_rows':0,'MAE':None,'RMSE':None,'sign_matches':None}
        p=np.asarray([r[physical_key] for r in selected],np.float64);m=np.asarray([r[model_key] for r in selected],np.float64)
        return {'reported_rows':len(selected),'MAE':float(np.abs(p-m).mean()),'RMSE':float(np.sqrt(np.mean((p-m)**2))),'sign_matches':int((np.sign(p)==np.sign(m)).sum()),'scope':'Descriptive score/rad; includes weak/zero responses unless explicitly marked stable diagnostic'}
    result={'all64_rows_retained':len(rows),'actual_defined_rows':len(actual),'actual_undefined_rows':len(rows)-len(actual),
      'all_defined_actual_coordinate':describe(actual,'physical_actual_coordinate_secant','WM_AD_actual_delta_projected_per_coordinate'),
      'all64_legacy':describe(rows,'physical_legacy_nominal_epsilon_slope','WM_AD_legacy_effective_direction'),'stable_physical_groups_per_epsilon':{}}
    for ei,eps in enumerate(EPS):
        selected=[r for r in rows if r['epsilon_index']==ei and r['stable_nonflat_shared_physical']]
        result['stable_physical_groups_per_epsilon'][str(eps)]=describe(selected,'physical_legacy_nominal_epsilon_slope','WM_AD_legacy_effective_direction')
    return result

def save_rows(path,rows):
    c._write_json(path.with_suffix('.json'),rows)
    with path.with_suffix('.csv').open('x',encoding='utf-8',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def run_analysis(protocol_path,output):
    started=time.perf_counter();m=metadata_preflight(protocol_path,output);p=validate_protocol(m['protocol'],True);i=p['inputs'];out=m['output'];out.mkdir()
    c._write_json(out/'protocol.json',p);c._write_json(out/'source_inventory.json',m['sources'])
    with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for key,value in m['sources'].items():z.write(checked(value),key)
    phase='released_saved_inputs';counts={'trusted_component_torch_loads':0,'native_WM_constructors':0,'WM_forwards':0,'center_predictions':0,'uVJP_calls':0,'original119f_loads':0,'optimizer_Actor_MJB_Env_physics_GPU':0}
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import numpy as np
        import torch
        torch.set_num_threads(4);require(not torch.cuda.is_initialized() and not torch.is_inference_mode_enabled(),'Fresh CPU process outside inference_mode')
        from .serial_probe105_calibration_v1 import controlled_object_score_FP32
        rng=c._rng(torch,np);rng_sha=c._rng_digest(rng);cf.save_raw_rng(out,'before',rng,np)
        with np.load(checked(i['test_full_arrays']),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        _,identities,pairs=cf.prepare_test(a,read(checked(i['test_branch_index'])),read(checked(i['test_snapshot_records'])),np)
        baseline,groups,physical,arithmetic=physical_descriptors(a,identities,pairs,np)
        c._write_json(out/'all152_identities.json',identities);c._write_json(out/'baseline8.json',baseline);save_rows(out/'physical_groups32',groups);save_rows(out/'physical_rows64',physical)
        phase='trusted_component_files';initial=torch.load(checked(i['initial_component']),map_location='cpu',weights_only=False);counts['trusted_component_torch_loads']+=1
        require(initial['schema']==c.ARTIFACT_SCHEMA,'Trusted isolated initial schema');receipt=read(checked(i['initial_validation']))
        require(c._tensor_digest(initial['WM'],torch)==receipt['initial_WM_sha256'] and initial['parent_checkpoint105']['sha256']==c.FROZEN_CHECKPOINT_SHA and same(initial['parent_checkpoint105'],receipt['strict_loader']['checkpoint']),'Initial validated parent state')
        norm={k:v for k,v in initial['WM'].items() if k.startswith(('norm.','delta_norm.')) or k=='error_floor'}
        require(c._tensor_digest(norm,torch)==receipt['frozen_norm_sha256'] and c._tensor_digest({'xi':initial['physical_xi46']},torch)==receipt['xi46_sha256'] and initial['normalizer_host_configuration']==receipt['normalizer_host_configuration'],'Original frozen norm/xi receipt')
        payloads={}
        for arm,alias in zip(ARMS,('pointwise_component','response_component')):
            payloads[arm]=torch.load(checked(i[alias]),map_location='cpu',weights_only=False);counts['trusted_component_torch_loads']+=1
            require(same(payloads[arm]['protocol'],i['training_protocol']),'Actual final component training run')
        xi=initial['physical_xi46'];scale=initial['frozen_s52'];require(xi.dtype==scale.dtype==torch.float32 and xi.shape==(46,) and scale.shape==(52,) and torch.isfinite(xi).all() and torch.isfinite(scale).all() and (scale>0).all(),'Frozen xi46/s52')
        x=torch.from_numpy(a['snapshot_x'].copy());u0=torch.from_numpy(a['snapshot_u'].copy());gp=torch.from_numpy(a['goal_p'].copy());gR=torch.from_numpy(a['goal_R'].copy())
        memory={'x':x,'u0':u0,'goal_p':gp,'goal_R':gR,'xi':xi,'scale':scale};memory_before=c._tensor_digest(memory,torch)
        np.savez_compressed(out/'physical_task_inputs.npz',snapshot_x=a['snapshot_x'],snapshot_u=a['snapshot_u'],goal_p=a['goal_p'],goal_R=a['goal_R'],
          branch_x_next=a['branch_x_next'],branch_u=a['branch_u'],branch_object_score_k=a['branch_object_score_k'],branch_object_terms_k=a['branch_object_terms_k'],
          branch_object_terms_k_CPU_secondary=a['branch_object_terms_k_CPU_secondary'],donor_object_score_k=a['donor_object_score_k'],
          direction_plus=a['direction__plus'],direction_minus=a['direction__minus'],legacy_d_eff=a['direction__d_eff'],center_offset_rad=a['direction__center_offset'],clipped=a['direction__clipped'],physical_xi46=xi.numpy(),frozen_s52=scale.numpy(),**arithmetic)
        metrics={};weight_hashes={};checks={}
        for arm in ARMS:
            phase='component_'+arm;model=cf.make_eval_component(payloads[arm],initial,arm,torch);counts['native_WM_constructors']+=1
            require(not model.training and all(not q.requires_grad for q in model.parameters()),'Freeze eval parameters; input AD only')
            before=c._tensor_digest(model.state_dict(),torch);norm_before=c._tensor_digest(c._norm_state(model),torch);cfg_before=c._norm_config(model)
            u=u0.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                require(not torch.is_inference_mode_enabled(),'inference_mode would invalidate requested input AD')
                dynamics=torch.cat((x,xi[None].expand(8,-1),x.new_zeros((8,174))),-1)
                pred,self15=model(x,u,dynamics);counts['WM_forwards']+=1;counts['center_predictions']+=8
                require(pred.shape==(8,58) and self15.shape==(8,15) and torch.isfinite(pred).all() and torch.isfinite(self15).all(),'Finite B8 native component output')
                terms,score=controlled_object_score_FP32(pred,gp,gR)
                gradient,=torch.autograd.grad(score.sum(),u);counts['uVJP_calls']+=1
            require(gradient.dtype==torch.float32 and gradient.shape==(8,20) and torch.isfinite(gradient).all() and all(q.grad is None for q in model.parameters()),'Finite frozen-model input derivative only')
            require(torch.isfinite(terms).all() and torch.isfinite(score).all() and torch.equal(u.detach().contiguous().view(torch.int32),u0.contiguous().view(torch.int32)),'Score and input bytes')
            grad=gradient.detach().numpy().copy();rows=projected_rows(grad,physical,np);save_rows(out/(arm+'_comparison64'),rows)
            np.savez_compressed(out/(arm+'_center_gradients.npz'),predicted_x8_FP32=pred.detach().numpy(),self15_logits_FP32=self15.detach().numpy(),score_terms8_FP32=terms.detach().numpy(),score8_FP32=score.detach().numpy(),gradient_u8x20_FP32=grad,
              snapshot_x8_FP32=a['snapshot_x'],snapshot_u8_FP32=a['snapshot_u'],goal_p8_FP32=a['goal_p'],goal_R8_FP32=a['goal_R'],
              actual_defined_mask64=np.asarray([r['actual_coordinate_defined'] for r in rows],bool),stable_shared_physical_mask64=np.asarray([r['stable_nonflat_shared_physical'] for r in rows],bool),
              physical_actual_secant64_F64=np.asarray([np.nan if r['physical_actual_coordinate_secant'] is None else r['physical_actual_coordinate_secant'] for r in rows],np.float64),
              WM_AD_actual_projected64_F64=np.asarray([np.nan if r['WM_AD_actual_delta_projected_per_coordinate'] is None else r['WM_AD_actual_delta_projected_per_coordinate'] for r in rows],np.float64),
              physical_legacy_slope64_F64=np.asarray([r['physical_legacy_nominal_epsilon_slope'] for r in rows],np.float64),WM_AD_legacy_effective64_F64=np.asarray([r['WM_AD_legacy_effective_direction'] for r in rows],np.float64))
            metrics[arm]=comparison_summary(rows,np);after=c._tensor_digest(model.state_dict(),torch);weight_hashes[arm]={'before':before,'after':after}
            checks[arm]=before==after and norm_before==c._tensor_digest(c._norm_state(model),torch) and cfg_before==c._norm_config(model)
            require(checks[arm],'Weights/normalizers/config mutated');del model
        after_rng=c._rng(torch,np);cf.save_raw_rng(out,'after',after_rng,np);require(c._rng_digest(after_rng)==rng_sha,'RNG changed; caller never repairs/restores it')
        require(c._tensor_digest(memory,torch)==memory_before,'Center/goals/xi/scale bytes changed')
        require(counts['trusted_component_torch_loads']==3 and counts['native_WM_constructors']==2 and counts['WM_forwards']==2 and counts['center_predictions']==16 and counts['uVJP_calls']==2,'Exact two-arm model/AD budget')
        for value in i.values():checked(value)
        for value in m['sources'].values():checked(value)
        require(same(pin(protocol_path),m['protocol_pin']),'Exploratory protocol changed')
        report={'schema':'full151105_component_task_gradient_CPU_analysis:1','status':'COMPLETED_EXPLORATORY_TWO_COMPONENT_TASK_SCORE_GRADIENT_NOT_CONFIRMATION','role':ROLE,
          'protocol':m['protocol_pin'],'input_pins':i,'counts':counts,'coverage':{'snapshots':8,'shared_physical_direction_groups':32,'epsilon_rows_per_arm':64},
          'physical_group_counts':dict(__import__('collections').Counter(r['label'] for r in groups)),'comparison_summaries':metrics,
          'parent_ea22_decision_unchanged':m['parent_confirmation_decision'],'physical_floor_rule':FLOOR_RULE,'invariants':{'weights_norm_config_unchanged':checks,'raw_RNG_before_after_unchanged_without_caller_repair':True,'center_goal_xi_scale_input_bytes_unchanged':True,'all_source_input_files_unchanged':True},
          'weight_digests':weight_hashes,'input_memory_digests':{'before':memory_before,'after':c._tensor_digest(memory,torch)},'RNG_digests':{'before':rng_sha,'after':c._rng_digest(after_rng)},
          'raw_RNG_saved_before_after':True,'branch_PRE_full58_directly_observed':False,'whole_hidden_state_qualified':False,'physical_gradient_qualified':False,'HO_force_gradient_qualified':False,'Actor_or_CHORD_benefit_qualified':False,'new_blind_confirmation':False,
          'scope':'Exploratory CPU FP32 absolute-u score derivative versus saved GPU finite chords, all32/64 retained. No old stable9 pooling, new data, true Jacobian, historical GPUAMP policy gradient, Actor parameter gradient, independent trajectory or ego claim.',
          'elapsed_seconds':time.perf_counter()-started}
        c._write_json(out/'report.json',report);c._write_json(out/'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        try:c._write_json(out/'failure.json',{'schema':'full151105_component_task_gradient_failure:1','status':'BLOCKED_EXPLORATORY_TASK_GRADIENT_NO_ROW_REPAIR_OR_RETRY','phase':phase,'error':repr(error),'counts':counts,'protocol':m['protocol_pin'],'no_confirmation_upgrade':True})
        except BaseException as saving:
            try:error.add_note('Failure evidence save also failed: '+repr(saving))
            except BaseException:pass
        raise
