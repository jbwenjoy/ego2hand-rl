"""Independent fixed-two-arm confirmation saved audit; no model/native Log/FW.

Default is protocol/release/final-health JSON/hash metadata only. Actual NPZ
reads and pure-Torch arithmetic require explicit --run after Root terminal.
No missing branch PRE58 is reconstructed or upgraded by this adapter.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MATH=ROOT/'src/full151_gradient/calibration_confirmation_independent_math_v2.py'
MATH_SHA='cac8f8fa098aa2ccfae24d0d710613c5b272b442ca591c55a9531a8a877a23d2'
LOSS=ROOT/'scripts/audit_WM_component_checkpoint_cpu_v2.py'
LOSS_SHA='64c38dbed909b98381086cbd0c2cbfad89f8f790108fd07bbe4675ec23b2d47e'
PREREG=ROOT/'protocols/full151105_WM_component_confirmation_eval_prereg_v1.json'
PREREG_SHA='3162bb17a8c224f2bbb2e3292d3530ab1e407ad730a597a3294f271a71283e31'
RELEASE_SHA='ca1520d9f94336b67e89e618ce57d346c043f421f1cfdc42a238eacc0270fe1f'
CORE_SHA='f75302fa70a894300ceb3a71492b9cffd973244990f13be26cee5b7e5e112659'
PRODUCER_READY=ROOT/'outputs/full151105_WM_component_confirmation_preparation_CPU_v1/READY.json'
ARMS=('pointwise','pointwise_plus_common_pre_response');PHASES=(24,74,124,148)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}


def checked(p):
    if type(p) is not dict or set(p)!={'path','sha256'} or not Path(p['path']).is_absolute():raise ValueError('Exact absolute pin')
    q=Path(p['path']).resolve(strict=True)
    if sha(q)!=p['sha256']:raise ValueError('Pinned bytes changed '+str(q))
    return q


def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)


def module(path,expected):
    if sha(path)!=expected:raise ValueError('Independent math/helper changed')
    s=importlib.util.spec_from_file_location('independent_'+path.stem,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def strict_json(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(strict_json(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(strict_json(x,y) for x,y in zip(a,b))
    return a==b


def metadata(protocol_path):
    if sha(PREREG)!=PREREG_SHA:raise ValueError('Fixed confirmation prereg')
    p=read(protocol_path);design=read(PREREG)
    if p['schema']!='full151105_WM_component_confirmation_CPU_protocol:1' or p['role']!=design['role'] or p['criterion']!=design['criterion'] or p['test_phases']!=list(PHASES) or p['arms']!=list(ARMS):raise ValueError('Frozen confirmation role/counts/criterion')
    if type(p['checkpoint_iteration']) is not int or p['checkpoint_iteration']!=1000 or type(p['cpu_threads']) is not int or p['cpu_threads']!=4 or p['GPU_run_authorized'] is not False:raise ValueError('Fixed final1000 FP32 CPU4 only')
    if set(p['inputs'])!=set(design['inputs']) or not all(same(v,design['inputs'][k]) for k,v in p['inputs'].items()):raise ValueError('Exact fixed16 semantic inputs')
    for q in p['source_pins'].values():checked(q)
    i=p['inputs'];release=read(checked(i['release']));health=read(checked(i['final_checkpoint_saved_review']))
    if i['release']['sha256']!=RELEASE_SHA or release['scientific_release_authorized'] is not True or release['evaluation_only'] is not True:raise ValueError('Root explicit fixed scientific release')
    if health['schema']!='full151105_WM_component_saved_checkpoint_CPU_review:1' or health['engineering_qualified'] is not True or type(health['passed']) is not int or type(health['total']) is not int or health['passed']!=29 or health['total']!=29 or len(health['checks'])!=29:raise ValueError('Actual final component health')
    if health['iteration']!=1000 or health['final_iteration_scope'] is not True or not all(v is True for v in health['checks'].values()):raise ValueError('True final1000 health scope')
    for arm,alias in zip(ARMS,('pointwise_component','response_component')):
        if not(same(health['checkpoints'][arm],i[alias]) and same(release['checkpoint_pins'][arm],i[alias])):raise ValueError('Both final models fixed before release')
    if not same(release['final_checkpoint_saved_review'],i['final_checkpoint_saved_review']) or not same(health['protocol'],i['training_protocol']) or not same(health['initial_component'],i['initial_component']):raise ValueError('Final health/initial/train provenance')
    seal=read(checked(i['sealed_manifest']));eng=read(checked(i['test_full_engineering_review']))
    if seal['scientific_release_authorized'] is not False or seal['branch_pre58_qualified'] is not False or seal['whole_hidden_state_qualified'] is not False:raise ValueError('Original seal qualifications retained')
    for q in seal['all_output_files'].values():checked(q)
    if eng['engineering_qualified'] is not True or eng['completed_branches']!=152 or eng['baseline_count']!=24 or eng['episode_step_returned']!=1808 or eng['branch_PRE_full58_qualified'] is not False or eng['WM_accuracy_or_response_benefit_evaluated'] is not False:raise ValueError('Engineering is not PRE58 or model qualification')
    for alias,name in [('test_full_arrays','physical_arrays.npz'),('test_branch_index','branch_index.json'),('test_snapshot_records','snapshot_records.json')]:
        if not same(i[alias],seal['all_output_files'][name]):raise ValueError('Original exact sealed input alias')
    ready=read(PRODUCER_READY);proof=read(checked(ready['cpu_report']))
    if ready['source_pins']!=proof['source_pins'] or ready['source_pins']['consumer']['sha256']!=CORE_SHA or proof['passed']!=proof['total'] or not all(v is True for v in proof['checks'].values()):raise ValueError('Current focused eval CPU proof')
    for q in ready['source_pins'].values():checked(q)
    return p


def symbolic_order():
    rows=[];pairs=[]
    for s in range(8):
        path=('A0','A2')[s//4];k=PHASES[s%4]
        for off in range(19):
            base=off<3;ei=None if base else (off-3)//8;j=None if base else (off-3)%8//2
            rows.append({'row_id':s*19+off,'snapshot_id':s,'path':path,'k':k,'type':'baseline' if base else 'direction',
               'baseline_repeat':off if base else None,'j':j,'epsilon_index':ei,'sign':0 if base else (-1,1)[(off-3)%2],
               'epsilon_rad':None if base else (.002,.001)[ei],'joint_index':None if base else (0,4,8,12)[j]})
        for j,q in enumerate((0,4,8,12)):
            for ei,eps in enumerate((.002,.001)):
                mi=s*19+3+8*ei+2*j
                pairs.append({'pair_id':len(pairs),'snapshot_id':s,'path':path,'k':k,'j':j,'joint_index':q,'epsilon_index':ei,
                    'requested_epsilon_rad':eps,'minus_branch_id':mi,'plus_branch_id':mi+1})
    return rows,pairs


def expected_input(a,rows,snapshots):
    wanted,pairs=symbolic_order()
    if len(rows)!=152 or len(snapshots)!=8:raise ValueError('All152/8 source rows')
    for row,w in zip(rows,wanted):
        if any(type(row.get(k)) is not type(v) or row[k]!=v for k,v in w.items()) or row['status']!='COMPLETED' or row['restorequalified'] is not True:raise ValueError('Original EPS-major branch order/status')
    shape={'snapshot_x':(8,58),'snapshot_u':(8,20),'branch_x_next':(152,58),'branch_u':(152,20),'direction__centers':(8,20),
        'direction__minus':(8,4,2,20),'direction__plus':(8,4,2,20)}
    for k,s in shape.items():
        if a[k].shape!=s or a[k].dtype!=np.float32 or not np.isfinite(a[k]).all():raise ValueError('Original finite FP32 '+k)
    for k in ('direction__d_eff','direction__center_offset'):
        if a[k].dtype!=np.float64 or a[k].shape!=(8,4,2,20) or not np.isfinite(a[k]).all():raise ValueError('Original F64 direction descriptor')
    if a['direction__clipped'].dtype!=bool or a['direction__clipped'].shape!=(8,4,2) or not bits(a['snapshot_u'],a['direction__centers']):raise ValueError('Original typed clipping and centers')
    for k,s in [('completed',(152,)),('snapshot_completed',(8,)),('restore_inventory_bitexact',(152,)),('restore_qualified',(152,))]:
        if a[k].dtype!=bool or a[k].shape!=s or not a[k].all():raise ValueError('All completed empirical masks')
    pair_out=[]
    for source in pairs:
        s,j,ei=source['snapshot_id'],source['j'],source['epsilon_index'];q=source['joint_index'];mi,pl=source['minus_branch_id'],source['plus_branch_id'];eps=source['requested_epsilon_rad']
        minus,plus=a['branch_u'][mi],a['branch_u'][pl];center=a['snapshot_u'][s];delta=plus.astype('f8')-minus.astype('f8')
        if delta[q]==0 or not np.isfinite(delta[q]):raise ValueError('Any zero denominator blocks all64')
        eff=delta/(2*eps);offset=(plus.astype('f8')+minus.astype('f8'))*.5-center.astype('f8');unit=np.eye(20,dtype='f4')[q]
        clip=not(np.array_equal(plus,center+np.float32(eps)*unit) and np.array_equal(minus,center-np.float32(eps)*unit))
        if not(bits(minus,a['direction__minus'][s,j,ei]) and bits(plus,a['direction__plus'][s,j,ei]) and bits(eff,a['direction__d_eff'][s,j,ei]) and bits(offset,a['direction__center_offset'][s,j,ei]) and bool(a['direction__clipped'][s,j,ei])==clip):raise ValueError('All original actual endpoints/legacy descriptors')
        pair_out.append({**source,'actual_du_rad':float(delta[q]),'clipped':clip,'center_offset_rad':offset.tolist(),'legacy_d_eff_dimensionless':eff.tolist()})
    for s,record in enumerate(snapshots):
        if record['snapshot_id']!=s or record['path']!=('A0','A2')[s//4] or record['k']!=PHASES[s%4] or record['baseline3_qualified'] is not True:raise ValueError('Fresh snapshot primitive identity')
    for bid,row in enumerate(rows):
        s,off=divmod(bid,19);source=np.asarray(row['actualu']);u=a['branch_u'][bid]
        if source.shape!=(20,) or not np.array_equal(source.astype('f8'),source.astype('f4').astype('f8')) or not bits(source.astype('f4'),u):raise ValueError('Source JSON actualu exact')
        keep=np.ones(20,dtype=bool) if off<3 else np.arange(20)!=row['joint_index']
        if not bits(u[keep],a['snapshot_u'][s,keep]):raise ValueError('Baseline/all non-j unchanged bytes')
    return {'x152_FP32':np.repeat(a['snapshot_x'],19,axis=0),'target152_FP32':a['branch_x_next'].copy(),'actual_u152_FP32':a['branch_u'].copy(),
       'minus_indices':np.asarray([r['minus_branch_id'] for r in pairs],dtype='i8'),'plus_indices':np.asarray([r['plus_branch_id'] for r in pairs],dtype='i8'),
       'actual_du_rad_F64':np.asarray([r['actual_du_rad'] for r in pair_out],dtype='f8')},wanted,pair_out


def raw58_schema_stats(value):
    out={}
    for k,v in value['blocks'].items():
        r={'unit':v['units'],'component_RMSE':v['component_RMSE'],'MAE':v['MAE'],'maximum_absolute':v['maximum_absolute'],'rows':value['rows']}
        if 'vector_RMS' in v:r['vector_RMS']=v['vector_RMS']
        if 'Frobenius_RMS' in v:r['Frobenius_RMS']=v['Frobenius_RMS']
        out[k]=r
    return out


def raw_RNG_before_after(run,helper):
    a,b=load(run/'before_RNG_arrays.npz'),load(run/'after_RNG_arrays.npz')
    ha,hb=read(run/'before_RNG_python_numpy.json'),read(run/'after_RNG_python_numpy.json')
    shape=set(a)==set(b)=={'numpy_MT','torch_CPU'} and a['numpy_MT'].dtype==b['numpy_MT'].dtype==np.uint32 and a['numpy_MT'].shape==b['numpy_MT'].shape==(624,) and a['torch_CPU'].dtype==b['torch_CPU'].dtype==np.uint8 and a['torch_CPU'].ndim==b['torch_CPU'].ndim==1 and a['torch_CPU'].shape==b['torch_CPU'].shape and a['torch_CPU'].size>0
    return {'raw_vector_dtype_shape_valid':bool(shape),'Python_JSON_recursive_values_types_floatbits_equal':helper._raw_tree_equal(ha['python'],hb['python']),
       'NumPy_header_types_values_floatbits_equal':helper._raw_tree_equal(ha['numpy_header'],hb['numpy_header']),
       'NumPy_MT_uint32_bytes_equal':bool(shape and bits(a['numpy_MT'],b['numpy_MT'])),
       'Torch_CPU_uint8_bytes_equal':bool(shape and bits(a['torch_CPU'],b['torch_CPU'])),
       'CUDA_initialized_both_False':ha['CUDA_initialized'] is hb['CUDA_initialized'] is False}


def audit(protocol_path,run,out):
    out.mkdir(exist_ok=False);auditor=pin(__file__);(out/'executed_auditor_source.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'math_source_before_actual_eval_read.json',{'auditor':auditor,'math':pin(MATH),'native52_arithmetic_helper':pin(LOSS),'actual_predictions_read':False,
       'rules':'All152/64, captured nativeFP32 D52 then F64 response/s52/actualdu. FP32 point native arithmetic. Raw58 units separate. No reconstruction of missing branchPRE58.'})
    p=metadata(protocol_path);i=p['inputs'];r=read(run/'report.json');sources=read(run/'source_inventory.json');math=module(MATH,MATH_SHA);helper=module(LOSS,LOSS_SHA);checks={};details={}
    def gate(k,v):checks[k]=bool(v)
    gate('actual_authorized_protocol_fixed_role_release_and_inputaliases',p['execution_authorized'] is p['CPU_run_authorized'] is True and p['GPU_run_authorized'] is False and p['cpu_threads']==4
       and same(r['protocol'],pin(protocol_path)) and strict_json(p,read(run/'protocol.json')) and strict_json(i,r['input_pins']))
    for q in sources.values():checked(q)
    gate('all_current_unique_source_pins_frozen_consumer',any(q['sha256']==CORE_SHA for q in sources.values()) and len({(checked(q).stat().st_dev,checked(q).stat().st_ino,q['sha256']) for q in sources.values()})==len(sources))
    with zipfile.ZipFile(run/'execution_sources.zip') as z:zipok=len(z.namelist())==len(sources) and all(hashlib.sha256(z.read(k)).hexdigest()==q['sha256'] for k,q in sources.items())
    gate('all_immutable_source_archive_bytes_exact',zipok)
    outputpins=read(run/'output_pins.json')
    for q in outputpins.values():checked(q)
    gate('all_original_output_pins_current_exact',True)
    a=load(checked(i['test_full_arrays']));source_rows=read(checked(i['test_branch_index']));snaps=read(checked(i['test_snapshot_records']))
    expected,identities,pairs=expected_input(a,source_rows,snaps);data=load(run/'matched_test_inputs152.npz')
    gate('all152_snapshot_repeat_design_target20_actualu_and64indices_exact',set(expected)==set(data) and all(bits(v,data[k]) for k,v in expected.items())
       and strict_json(identities,read(run/'all152_identities.json')) and strict_json(pairs,read(run/'all64_pairs.json')))
    import torch
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():raise ValueError('Saved arithmetic only, never CUDA')
    metrics={};all_stats={};arithmetic={};targets={};scales={};pair_details={}
    for arm in ARMS:
        e=load(run/(arm+'_evaluation_arrays.npz'))
        gate(arm+'_all_captured_native52_and_raw58_target_control_index_binding',bits(e['target152_FP32'],data['target152_FP32']) and bits(e['actual_du_rad_F64'],data['actual_du_rad_F64']) and bits(e['minus_indices'],data['minus_indices']) and bits(e['plus_indices'],data['plus_indices']))
        calculated=math.all_confirmation_arithmetic(e['point_delta52_FP32'],e['pred_common52_FP32'],e['target_common52_FP32'],e['scale52_FP32'],e['minus_indices'],e['plus_indices'],e['actual_du_rad_F64'],e['pred152_FP32'],e['target152_FP32'])
        native=helper.postprocess_saved_losses(e,torch);metrics[arm]={'pointwise_loss':native['pointwise_loss'],'common_pre_pair_loss':native['common_pre_pair_loss']}
        gate(arm+'_native_FP32_point_and_F64_pair_scalar_exact',all(metrics[arm][k]==r['native52_frozen_criterion_metrics'][arm][k] for k in metrics[arm])
           and r['native52_frozen_criterion_metrics'][arm]['point_rows']==152 and r['native52_frozen_criterion_metrics'][arm]['pairs']==64 and native['native_rotvec_norm_max']<np.pi-.01)
        gate(arm+'_all152_nonrotation_native_difference_capture_exact',helper.nonrotation_delta_evidence(e,{'target':data['target152_FP32'],'common_pre':data['x152_FP32']}))
        raw_stats=raw58_schema_stats(calculated['raw58_state_error_original_units']);all_stats[arm]=calculated['raw58_state_error_original_units']
        gate(arm+'_raw58_F64_signed_error_and_six_units_exact_no_angle',bits(e['state_error58_F64'],calculated['arrays']['raw_state_error152x58_F64']) and strict_json(raw_stats,r['raw58_state_error_original_units'][arm]))
        aux=e['self15_logits_FP32'];gate(arm+'_self15_finite_raw_only_not_HO_label',aux.dtype==np.float32 and aux.shape==(152,15) and np.isfinite(aux).all())
        csvok=True;count=0
        with (run/(arm+'_all152_state_error58.csv')).open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                bid,feature=divmod(count,58);count+=1
                unit=next(u for lo,hi,u in math.BLOCKS58.values() if lo<=feature<hi);identity=identities[bid]
                csvok &= int(row['branch_id'])==bid and int(row['snapshot_id'])==identity['snapshot_id'] and row['path']==identity['path'] and int(row['k'])==identity['k'] and int(row['feature'])==feature and row['unit']==unit
                csvok &= float(row['predicted'])==float(e['pred152_FP32'][bid,feature]) and float(row['target'])==float(e['target152_FP32'][bid,feature]) and float(row['error_F64'])==calculated['arrays']['raw_state_error152x58_F64'][bid,feature]
        gate(arm+'_every152x58_CSV_row_identity_values_units_exact',csvok and count==152*58)
        # Native scalar criteria use Torch reduction order. Independent NumPy
        # per-pair/full-coordinate summaries are retained, never substituted.
        point_rows=(torch.from_numpy(e['point_delta52_FP32'])/torch.from_numpy(e['scale52_FP32'])).square().mean(-1).numpy()
        np.savez_compressed(out/(arm+'_independent_all152_all64_arithmetic.npz'),**calculated['arrays'],native_point_row_MSE152_FP32=point_rows)
        arithmetic[arm]={'native_scalar_replay':metrics[arm],'NumPy_F64_pair_MSE_shadow':calculated['all64_normalized_pair_MSE_F64'],
           'native52_point_coordinates':calculated['native52_point_coordinates'],'native52_response_coordinates':calculated['native52_response_coordinates'],
           'native52_secant_error_per_command_rad':calculated['native52_secant_error_per_command_rad']}
        targets[arm]=e['target_common52_FP32'];scales[arm]=e['scale52_FP32'];pair_details[arm]={'all64_pairs':64,'clipped_pairs':sum(x['clipped'] for x in pairs)}
    gate('two_arms_same_captured_target52_and_fixed_s52_bytes',bits(targets[ARMS[0]],targets[ARMS[1]]) and bits(scales[ARMS[0]],scales[ARMS[1]]))
    decision=math.confirmation_rule(metrics[ARMS[0]]['pointwise_loss'],metrics[ARMS[1]]['pointwise_loss'],metrics[ARMS[0]]['common_pre_pair_loss'],metrics[ARMS[1]]['common_pre_pair_loss'])
    wanted={'criterion':p['criterion'],'response_MSE_strictly_lower':decision['response_error_strictly_lower'],'pointwise_MSE_not_higher':decision['native_pointwise_not_increased'],
       'local_component_confirmation':decision['local_component_confirmation'],'all152_rows_all64_pairs_used':True,'policy_benefit_or_physical_gradient_qualified':False}
    gate('frozen_all64_strict_response_all152_nonincrease_decision_no_threshold',strict_json(wanted,r['decision']))
    rng=raw_RNG_before_after(run,helper);gate('three_raw_RNG_groups_before_after_header_dtype_bytes',all(rng.values()) and r['raw_RNG_saved_before_after'] is True)
    gate('source_weight_norm_input_hash_assertions_separate_not_raw_model_reload',all(r['invariants']['component_weights_normalizers_unchanged'].values())
       and all(v['before']==v['after'] for v in r['weight_digests'].values()) and r['input_memory_digests']['before']==r['input_memory_digests']['after'] and all(v is True for k,v in r['invariants'].items() if k!='component_weights_normalizers_unchanged'))
    counts={'trusted_component_torch_loads':3,'native_WM_constructors':2,'WM_forwards':2,'predicted_rows':304,'AD_calls':0,'original119f_loads':0,'MJB_Env_physics_GPU':0}
    gate('source_successful_path_counts_three_loads_two_ctors_two_B152_FW_zeroAD',r['counts']==counts)
    gate('no_branchPRE58_hidden_gradient_policy_or_ego_qualification',r['schema']=='full151105_WM_component_confirmation_CPU_analysis:1' and r['status']=='COMPLETED_ALL152_ALL64_COMPONENT_CONFIRMATION_NOT_POLICY_BENEFIT'
       and r['branch_PRE_full58_directly_observed'] is r['whole_hidden_state_qualified'] is r['physical_gradient_qualified'] is r['physical_reference_qualified'] is False and r['source_lineage_N']==1 and r['independent_ego_N']==0)
    for q in [*sources.values(),*i.values(),*outputpins.values()]:checked(q)
    gate('reviewer_no_model_or_checkpoint_load_nativeLog_AD_Env_physics',not torch.cuda.is_initialized() and not any(k in sys.modules for k in ('dex_supermimic','mujoco','warp','mujoco_warp','mjlab')) and pin(__file__)==auditor and sha(MATH)==MATH_SHA and sha(LOSS)==LOSS_SHA)
    details.update(raw_RNG=rng,pair_details=pair_details,weight_norm_input_invariance='Digest/source assertions bound; no reviewer model/weight reload. Native52 rotational Log capture remains source evidence, not recomputed.',
       input_scope='Direct snapshot_x8 repeated19 input design; no independent observation or reconstruction of branch PRE58. Conditional primitive/inventory/base gates only.',
       source_budget='Successful producer counters3own component loads/2new nativeWM/2B152 forwards; not independently re-instrumented by this NumPy/Torch arithmetic reader.')
    result={'schema':'full151105_WM_component_confirmation_saved_independent:1','status':'PASS_SAVED_ALL152_ALL64_COMPONENT_CONFIRMATION_ARITHMETIC' if all(checks.values()) else 'FAILED_SAVED_COMPONENT_CONFIRMATION_REVIEW',
       'engineering_qualified':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':auditor,'math':pin(MATH),'native52_arithmetic_helper':pin(LOSS),
       'protocol':pin(protocol_path),'producer_report':pin(run/'report.json'),'run_dir':str(run.resolve()),'native52_metrics':metrics,'decision':wanted,
       'raw58_state_error_original_units':all_stats,'native52_descriptive_arithmetic':arithmetic,'details':details,'branch_PRE_full58_qualified':False,'whole_hidden_state_qualified':False,
       'physical_gradient_qualified':False,'policy_benefit_qualified':False,'independent_ego_N':0,'reviewer_model_checkpoint_loads_FW_AD_MJB_Env_physics_GPU':0}
    dump(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total'],'native52_metrics':metrics,'decision':wanted},indent=2));return 0 if all(checks.values()) else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run-dir',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    p=metadata(args.protocol)
    if not args.run:
        print(json.dumps({'status':'PASS_BLIND_SOURCE_HASH_JSON_METADATA_ONLY','auditor':pin(__file__),'math':pin(MATH),'loss_helper':pin(LOSS),
           'protocol':pin(args.protocol),'actual_test_payload_or_predictions_read':0,'Torch_imported':'torch' in sys.modules},indent=2));return 0
    if args.run_dir is None or args.output is None:raise ValueError('Actual fixed source run and fresh output required')
    try:return audit(args.protocol,args.run_dir,args.output)
    except Exception as e:
        if args.output.exists() and not(args.output/'failure.json').exists():dump(args.output/'failure.json',{'error':repr(e),'auditor':pin(__file__),'saved_only':True,'source_outputs_unchanged':True})
        raise


if __name__=='__main__':raise SystemExit(main())
