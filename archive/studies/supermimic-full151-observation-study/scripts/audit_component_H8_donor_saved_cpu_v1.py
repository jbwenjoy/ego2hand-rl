"""Independent saved H8 donor audit; no model/checkpoint/Torch/FW execution.

Default validates prereg/protocol/source JSON and hashes only. Actual --run must
wait for Root's terminal authorization. Raw call traces, original donor source
bytes and six-unit arithmetic are checked; NN execution remains source-backed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
UTIL_PATH=ROOT/'scripts/audit_component_task_gradient_saved_cpu_v1.py'
UTIL_SHA='472c72117dde51f7edd7c28d34301cef88939ebf6f5a64cb9bc6849dab42d0c5'
MATH_PATH=ROOT/'src/full151_gradient/component_H8_donor_independent_math_v1.py'
MATH_SHA='4158040c650986e8f65979debbe79648e0f2c5806f1006c289cdb99d6464ea55'
CORE=ROOT/'src/full151_gradient/component_H8_donor105_v1.py'
CORE_SHA='0c77dc5255ffd2d7294dabdc7bbaf8ed09e80f3f2bc0eb89fee3954d268a6b13'
READY=ROOT/'outputs/full151105_component_H8_donor_preparation_CPU_v1/READY.json'
READY_SHA='47772391f9ebdb7f62746b7ec10d4a147e2d6660eeb213c286fd6d90df053700'
PREREG=ROOT/'protocols/full151105_three_component_H8_donor_CPU_prereg_v1.json'
PREREG_SHA='21781c6174c899df88f55b1457f8c771d30c5007809397cc5f949f1a892c63a6'
ROLES=('initial_unmodified_A2','pointwise_final1000','pointwise_plus_response_final1000')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(v,s):
    if not v:raise ValueError(s)


def module(p,expected):
    require(sha(p)==expected,'Frozen independent helper '+str(p))
    s=importlib.util.spec_from_file_location('H8_saved_'+Path(p).stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


UTIL=module(UTIL_PATH,UTIL_SHA)
pin=UTIL.pin;read=UTIL.read;checked=UTIL.checked;same=UTIL.same;load=UTIL.load;bits=UTIL.bits;dump=UTIL.dump;subset=UTIL.subset


def metadata(protocol_path):
    require(sha(PREREG)==PREREG_SHA and sha(CORE)==CORE_SHA and sha(READY)==READY_SHA,'Frozen H8 source/method/focused proof')
    p,design=read(protocol_path),read(PREREG)
    require(p['schema']==design['execution_protocol_schema'] and p['role']==design['role'],'Explicit exploratory H8 namespace')
    for k in ('execution_authorized','CPU_run_authorized','GPU_run_authorized'):require(type(p[k]) is bool,'Explicit authorization bool')
    require(p['GPU_run_authorized'] is False,'CPU only')
    for k in ('model_roles','cpu_threads','horizon','batch_windows','starts_per_path','accuracy_gate','output_prefix'):
        require(UTIL.raw_tree_equal(p[k],design[k]),'Frozen H8 method '+k)
    require(set(p['inputs'])==set(design['inputs']) and len(p['inputs'])==14,'Fixed14 semantic inputs')
    for k,v in p['inputs'].items():require(same(v,design['inputs'][k]),'No source population/model selection '+k)
    require(same(p['proposal'],design['proposal']),'Fixed proposal')
    for v in p['source_pins'].values():checked(v)
    require(any(same(v,pin(CORE)) for v in p['source_pins'].values()),'Actual core bound in protocol')
    i=p['inputs'];health=read(checked(i['final_checkpoint_saved_review']));UTIL.allchecks(health)
    require(health['engineering_qualified'] is True and health['iteration']==1000 and health['final_iteration_scope'] is True,'Actual final1000 saved component health')
    require(same(health['initial_component'],i['initial_component']) and same(health['protocol'],i['training_protocol']),'Initial/training source route')
    for arm,key in zip(('pointwise','pointwise_plus_common_pre_response'),('pointwise_component','response_component')):require(same(health['checkpoints'][arm],i[key]),'Same final component '+arm)
    review=read(checked(i['H1_saved_independent25']));UTIL.allchecks(review)
    require(review['schema']=='full151105_state_prediction452_saved_independent:1' and review['engineering_qualified'] is True and same(review['config'],i['H1_config']) and same(review['producer_report'],i['H1_report']),'Original matched native H1 saved authority')
    report=read(checked(i['H1_report']));require(same(report['config'],i['H1_config']) and report['prediction_rows']==452,'Original all452 route')
    h1run=checked(i['H1_report']).parent;manifest=read(h1run/'output_pins.json')
    for key,name in [('H1_matched_inputs','matched_input_arrays.npz'),('H1_prediction_arrays','prediction_arrays.npz'),('H1_identities','group_identities452.json')]:require(same(i[key],manifest[name]) and checked(i[key]).samefile(h1run/name),'Original native saved array alias '+key)
    receipt=read(checked(i['initial_validation']));require(same(receipt['strict_loader']['checkpoint'],report['input_pins']['fixed_A2_checkpoint']),'Same original unmodified A2 parent')
    ready=read(READY);proof=read(checked(ready['cpu_report']));UTIL.allchecks(proof)
    require(ready['source_pins']==proof['source_pins'] and ready['source_pins']['consumer']['sha256']==CORE_SHA,'Actual current-source focused CPU proof')
    for v in ready['source_pins'].values():checked(v)
    return p


def window_identities(mapping):
    return [{'window_id':w,'path':('A0','A2')[int(pi)],'start_k':int(k),'action_phases':list(range(int(k),int(k)+8)),
      'target_observations':list(range(int(k)+1,int(k)+9)),'end_observation':int(k)+8,'no_u150':True}
      for w,(pi,k) in enumerate(zip(mapping['window_path_index286'],mapping['window_start_k286']))]


def producer_style_statistics(stats,role):
    rows=[]
    for h in range(1,9):
        for path in ('ALL','A0','A2'):
            view=stats['per_horizon'][str(h)] if path=='ALL' else stats['by_path_per_horizon'][path][str(h)]
            for name,b in view['blocks'].items():
                row={'horizon':h,'path':path,'block':name,'unit':b['units'],'window_count':view['overlapping_windows'],
                  'component_RMSE':b['component_RMSE'],'MAE':b['mean_absolute'],'maximum_absolute':b['maximum_absolute'],
                  'vector_or_Frobenius_RMS':b.get('vector_RMS',b.get('Frobenius_RMS','')),'model_role':role}
                rows.append(row)
    return rows


def audit(protocol_path,run,out):
    out.mkdir(exist_ok=False);auditor=pin(__file__);(out/'executed_auditor.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'equations_before_actual_H8_read.json',{'auditor':auditor,'math':pin(MATH_PATH),'generic_saved_helpers':pin(UTIL_PATH),'actual_H8_prediction_values_read':False,
      'scope':'All2x143x8; original donor keys and 298/300 byte checks; actual call traces consume previous prediction in both inputs. F64 original six units, no angle/accuracy gate.'})
    p=metadata(protocol_path);i=p['inputs'];math=module(MATH_PATH,MATH_SHA);r=read(run/'report.json');sources=read(run/'source_inventory.json');checks={};details={}
    def gate(k,v):checks[k]=bool(v)
    gate('authorized_protocol_fixed14inputs_proposal_roles_horizon_no_accuracy_gate',p['execution_authorized'] is p['CPU_run_authorized'] is True and same(r['protocol'],pin(protocol_path)) and UTIL.raw_tree_equal(p,read(run/'protocol.json')) and UTIL.raw_tree_equal(i,r['input_pins']))
    for v in sources.values():checked(v)
    gate('current_unique_source_inventory_frozen_core',any(same(v,pin(CORE)) for v in sources.values()) and len({(checked(v).stat().st_dev,checked(v).stat().st_ino) for v in sources.values()})==len(sources))
    with zipfile.ZipFile(run/'execution_sources.zip') as z:archive_ok=len(z.namelist())==len(sources) and all(hashlib.sha256(z.read(k)).hexdigest()==v['sha256'] for k,v in sources.items())
    gate('all_execution_ZIP_source_bytes_exact',archive_ok)
    outputpins=read(run/'output_pins.json')
    for v in outputpins.values():checked(v)
    gate('all_immutable_output_pins_current_exact',True)
    a=load(checked(i['H1_matched_inputs']));identities=read(checked(i['H1_identities']));mapping=math.donor_window_mapping(identities)
    original=math.source_byte_guards(a['pre_x452_FP32'],a['target_x452_FP32'],a['actual_u452_FP32'],a['all_substep_u452x4x20_FP32'],mapping)
    rows=mapping['window_transition_row_indices286x8'];starts=window_identities(mapping)
    wanted={'initial_x286_FP32':a['pre_x452_FP32'][mapping['window_initial_row_indices286']],
      'target286x8_FP32':a['target_x452_FP32'][rows],'actual_u286x8_FP32':a['actual_u452_FP32'][rows],
      'source_group_ids286x8':rows,'donor_indices2x150':mapping['donor_row_indices2x150'],
      'all298_links58_bitmask':np.ones((2,149,58),bool),'all300_control4_bitmask':np.ones((2,150,4,20),bool)}
    data=load(run/'matched_donor_windows286.npz')
    gate('source_full298_native58_links_and300_fourheld_controls_recomputed',all(original[k] is True for k in ('full58_links_bitexact','four_substep_controls_bitexact')))
    gate('all286_windows_targets_controls_indices_and_terminal149_to_obs150',set(data)==set(wanted) and all(bits(v,data[k]) for k,v in wanted.items()) and UTIL.raw_tree_equal(starts,read(run/'all286_window_identities.json')))
    xi=load(checked(i['H1_prediction_arrays']))['physical_xi46_FP32'];UTIL.typed(xi,(46,),'f4','old physical xi46')
    receipt=read(checked(i['initial_validation']))
    gate('old_H1_xi46_rawbytes_match_fixed_component_initial_receipt',UTIL.tensor_digest_arrays({'xi':xi})==receipt['xi46_sha256'])
    memory=UTIL.tensor_digest_arrays({'initial_x':wanted['initial_x286_FP32'],'controls':wanted['actual_u286x8_FP32'],'xi':xi})
    gate('recomputed_captured_input_memory_digest_matches_before_after',memory==r['input_memory_digests']['before']==r['input_memory_digests']['after'])
    all_rows=[];stats_by_role={};trace_by_role={};completed_count=0
    for role in ROLES:
        e=load(run/(role+'_rollout_final.npz'));partial=load(run/(role+'_rollout_partial.npz'))
        require(set(e)==set(partial)|{'error58_F64'},'Actual final/last-partial ABI')
        gate(role+'_final_equals_latest_partial_all_saved_raw_arrays',all(bits(v,e[k]) for k,v in partial.items()) and same(r['outputs'][role],pin(run/(role+'_rollout_final.npz'))))
        UTIL.typed(e['prediction58_FP32'],(286,8,58),'f4','all H8 predictions');UTIL.typed(e['self15_logits_FP32'],(286,8,15),'f4','raw auxiliary self15')
        for k in ('attempted_mask','completed_mask'):require(e[k].dtype==bool and e[k].shape==(286,8) and e[k].all(),'Full nonvacuous H8 mask '+k)
        completed_count+=int(e['completed_mask'].sum())
        trace_by_role[role]=math.trace_byte_guards(data['initial_x286_FP32'],e['prediction58_FP32'],e['state_input58_FP32'],e['dynamics_state58_FP32'],xi)
        gate(role+'_all8_real_call_traces_previous_prediction_no_teacher_or_stale_dynamics',all(trace_by_role[role][k] is True for k in ('initial_native_PRE_only','previous_predicted_state_each_h','dynamics_state_refreshed_each_h')) and bits(e['actual_call_u20_FP32'],data['actual_u286x8_FP32']))
        math_result=math.raw58_error_statistics(e['prediction58_FP32'],data['target286x8_FP32'],mapping['window_path_index286'])
        gate(role+'_all286x8x58_F64_signed_errors_original_units',bits(e['error58_F64'],math_result['signed_error286x8x58_F64']))
        stats_by_role[role]=math_result['statistics'];all_rows.extend(producer_style_statistics(math_result['statistics'],role))
        np.savez_compressed(out/(role+'_independent_error286x8x58.npz'),signed_error286x8x58_F64=math_result['signed_error286x8x58_F64'],absolute_error286x8x58_F64=math_result['absolute_error286x8x58_F64'])
    gate('all_three_roles_432_horizon_path_unit_stats_and_CSV_no_58mixed_norm',UTIL.raw_tree_equal(all_rows,r['statistics']) and UTIL.csv_matches(run/'all_models_horizons_units.csv',all_rows))
    rng=UTIL.raw_rng(run);gate('raw_Python_NumPy_TorchCPU_RNG_values_headers_bytes_equal',all(rng.values()) and r['raw_RNG_saved'] is r['RNG_unchanged_without_repair'] is True)
    inv=r['model_invariants']
    gate('initial_unmodified_WM_digest_and_three_frozen_norm_digest_relations',set(inv)==set(ROLES) and inv[ROLES[0]]['WM_sha256_before']==receipt['initial_WM_sha256'] and all(v['WM_sha256_before']==v['WM_sha256_after'] and v['norm_sha256_before']==v['norm_sha256_after']==receipt['frozen_norm_sha256'] and v['normalizer_host_configuration_unchanged'] is True and v['artifact_role']==k and v['not_native105_resume_qualification'] is True for k,v in inv.items()))
    counts={'component_file_reads':3,'native_WM_constructors':3,'WM_forward_attempted':24,'WM_forward_returned':24,'predicted_rows':6864,'AD_optimizer_Actor_119f_MJB_Env_physics_GPU':0}
    gate('all6864_dependent_outputs_source_3loads_3ctors_24B286_FW_zeroAD',completed_count==6864 and r['counts']==counts)
    gate('runtime_original_source_gate_flags_crosschecked_by_raw_arrays',r['runtime_source_gates']=={'all298_links_full58_bitexact':True,'all300_actual_u_four_substeps_bitexact':True,'component_oldH1_xi46_bitexact':True})
    gate('exploratory_status_no_teacher_clamp_Rprojection_accuracy_hidden_or_physics_upgrade',r['schema']=='full151105_three_component_H8_donor_CPU_analysis:1' and r['status']=='COMPLETED_EXPLORATORY_FIXED_CONTROL_H8_NOT_POLICY_IMAGINATION'
      and r['all286_windows_all8_horizons_used'] is True and all(r[k] is False for k in ('teacher_forcing','output_clamping_or_R_projection','accuracy_qualified','physical_gradient_qualified','physical_reference_qualified','whole_hidden_state_qualified')) and r['source_lineage_N']==1 and r['independent_ego_N']==0)
    for v in [*sources.values(),*i.values(),*outputpins.values()]:checked(v)
    gate('reviewer_NumPy_only_no_model_checkpoint_Torch_nativeLog_FW_AD_or_physics',not any(k.split('.')[0] in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules) and pin(__file__)==auditor and sha(MATH_PATH)==MATH_SHA)
    details.update(original_source_byte_guards=original,actual_recurrence_trace_guards=trace_by_role,raw_RNG=rng,source_unique_files=len(sources),output_pins=len(outputpins),
      model_scope='Initial digest/final weight/norm/host invariance are source-backed saved digest relations; no reviewer component load/constructor or forward.',
      input_scope='Actual saved x58 and dynamics[:58] call traces are independently byte checked. Fixed xi46 input digest bound to old H1; zero174 construction is source-backed, not a separately captured full278 trace.',
      population_scope='Three fixed models, two known donor paths, 286 overlapping windows, 6864 dependent predictions. H1 here is only286 starts, not the old452 mixture. No independent-N, CI, model winner, policy imagination, reference or physics success claim.')
    result={'schema':'full151105_three_component_H8_donor_saved_independent:1','status':'PASS_SAVED_ALL286_H8_THREE_COMPONENT_RAWSTATE_ARITHMETIC' if all(checks.values()) else 'FAILED_SAVED_H8_REVIEW',
      'engineering_qualified':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':auditor,'math':pin(MATH_PATH),'producer_READY':pin(READY),
      'protocol':pin(protocol_path),'producer_report':pin(run/'report.json'),'run_dir':str(run.resolve()),'input_pins':i,'window_count':286,'horizon':8,'dependent_prediction_rows':completed_count,
      'raw58_statistics':stats_by_role,'details':details,'accuracy_qualified':False,'whole_hidden_state_qualified':False,'physical_gradient_qualified':False,'policy_benefit_qualified':False,
      'independent_ego_N':0,'reviewer_component_loads_FW_AD_MJB_Env_physics_GPU':0}
    dump(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total'],'dependent_prediction_rows':completed_count},indent=2));return 0 if all(checks.values()) else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run-dir',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args();metadata(a.protocol)
    if not a.run:
        print(json.dumps({'status':'PASS_SOURCE_JSON_HASH_ONLY_NO_ACTUAL_H8_VALUES','auditor':pin(__file__),'math':pin(MATH_PATH),'actual_H8_prediction_values_read':0}));return 0
    require(a.run_dir is not None and a.output is not None,'Actual run/fresh review output required')
    try:return audit(a.protocol,a.run_dir.resolve(strict=True),a.output.resolve())
    except Exception as error:
        if a.output.exists() and not(a.output/'failure.json').exists():dump(a.output/'failure.json',{'error':repr(error),'auditor':pin(__file__),'saved_only':True,'producer_outputs_unchanged':True})
        raise


if __name__=='__main__':raise SystemExit(main())
