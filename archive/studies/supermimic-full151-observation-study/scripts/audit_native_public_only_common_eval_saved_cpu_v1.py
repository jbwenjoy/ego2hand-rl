"""Separate scalar saved arithmetic for one native-only FW and reused old3.

No Torch/checkpoint/nativeLog/model/FW imports or calls. The reviewer authored
the new producer: this is an independent arithmetic implementation, not an
independent experimental author. Reuse frozen prior scalar helpers; no new gate.
"""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/full151105_native_public_only_common_evaluation_CPU_v1'
OUT=ROOT/'outputs/full151105_native_public_only_common_eval_saved_recheck_v1'
ROLE='native52_self15_only'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def check(v):assert pin(v['path'])['sha256']==v['sha256'];return Path(v['path'])
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main():
    OUT.mkdir(exist_ok=False)
    write(OUT/'equations_before_arrays.json',{'source':pin(__file__),'actual_array_reads':False,
       'F64_comparison':'Inherited frozen scalar/fsum helper equivalent: atol2e-14+rtol2e-12, no new scientific accuracy gate.',
       'FP32_BCE':'Inherited16eps32*(1+abs(logit)+abs(stableF64BCE)), not cross-stride bit replay.',
       'FP32_reductions':'Inherited conditional gamma_n nonnegative reduction envelope; no NumPy mean asserted as Torch bit truth.',
       'scope':'New13 scalar/perfeature/cache metrics; original39 copied reference statistics/456 rows exact-field projection; rawRNG and sourceZIP direct, model weights/norm invariants source-backed only.'})
    r=read(RUN/'report.json');p=read(check(r['protocol']));i=p['inputs'];reference=read(check(i['reference_description_report']))
    oldreview=read(check(i['reference_eval_saved12']));helper=check(oldreview['auditor'])
    spec=importlib.util.spec_from_file_location('qualified_scalar_common_math',helper);R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
    M,C,V=R.M,R.C,R.V
    checks={};details={}
    def gate(k,v):checks[k]=bool(v)
    assert r['schema']=='full151105_native_public_only_common_evaluation_CPU_analysis:1'
    for item in [*r['outputs'].values(),*r['source_pins'].values(),*i.values()]:check(item)
    with zipfile.ZipFile(RUN/'execution_sources.zip') as z:
        zipok=set(z.namelist())==set(r['source_pins']) and all(hashlib.sha256(z.read(k)).hexdigest()==v['sha256'] for k,v in r['source_pins'].items())
    gate('protocol_current_sources_ZIP_and_all_output_pins',zipok and read(RUN/'protocol.json')==p)
    d=arrays(RUN/'prepared_public_inputs152.npz');f=arrays(RUN/'native52_self15_only_forward152.npz');e=arrays(RUN/'native52_self15_only_common_operands.npz')
    ids=read(RUN/'group152_identities.json');refreport=read(check(i['reference_eval_report']));refproto=read(check(i['reference_eval_protocol']))
    with np.load(check(refreport['outputs']['prepared_inputs_and_context.npz']),allow_pickle=False) as z:
        equal=all(V.bits(v,z[k]) for k,v in d.items())
    expected=V.schedule();assert len(ids)==152 and all(all(row[k]==s[k] for k in s) for row,s in zip(ids,expected))
    dyn=np.concatenate([d['x58_FP32'],np.tile(d['physical_xi46_FP32'],(152,1)),np.zeros((152,174),np.float32)],axis=1)
    digest=R.A.tensor_digest_arrays({'x':d['x58_FP32'],'u':d['u20_FP32'],'dynamics':dyn,'target_outside_forward':d['target58_FP32']})
    gate('exact_old152_prepared_input_groups_and_actual_four_tensor_digest',equal and digest==r['actual_fit_input_digest']==refreport['fit_input_digest'] and V.bits(d['source_group_id152'],np.asarray([row['group_id'] for row in expected],np.int64)) and V.bits(d['source_substep_rows152x4'],4*d['source_group_id152'][:,None]+np.arange(4)))
    joined=read(check(refproto['inputs']['cached_sensor_labels']));truth=[]
    for row,identity in zip(joined,ids):
        assert all(row[k]==identity[k] for k in identity)
        assert row['label_time']=='lastsolver_preintegration~kdt+.015' and row['state_time']=='postintegration_kdt+.020'
        y=C.mapped15([row['raw_found25']]);assert V.bits(y,np.asarray(row['contact15'],np.float32));truth.append(y)
    gate('all152_cached_rawfound25_to15_labels_and_15ms_phase',len(joined)==152 and V.bits(np.asarray(truth,np.float32),d['cached_self15_target_FP32']))
    V.typed(f['prediction58_FP32'],(152,58),'f4','prediction');V.typed(f['self15_logits_FP32'],(152,15),'f4','logits')
    assert np.all(f['attempted_mask']) and np.all(f['completed_mask']) and set(f)=={'prediction58_FP32','self15_logits_FP32','attempted_mask','completed_mask'}
    D=e['native_D52_FP32'];scale=d['effective_s52_FP32'];V.typed(D,(152,52),'f4','capturedD52')
    nonrot=True
    for a,b,c,z in [(0,40,0,40),(40,43,40,43),(46,49,52,55),(49,52,55,58)]:
        nonrot &= V.bits(D[:,a:b],d['target58_FP32'][:,c:z]-f['prediction58_FP32'][:,c:z])
    gate('native_nonrot_sign_shape_finite_and_full_forward_masks',nonrot and V.bits(e['self15_logits_FP32'],f['self15_logits_FP32']) and V.bits(e['self15_target_FP32'],d['cached_self15_target_FP32']) and V.bits(e['effective_s52_FP32'],scale))
    u=2**-24;eps=2**-23;norm=(D.astype('f8')/scale.astype('f8'))**2;sq=e['point_squared52_FP32'];point=e['point_per_row_FP32'];sb=e['self15_BCE_elements_FP32'];ss=e['self15_sum_per_row_FP32'];pair=e['native_pair_per_row_FP32']
    fpok=bool(np.all(abs(sq.astype('f8')-norm)<=8*u*np.maximum(norm,np.finfo('f4').tiny)))
    pointok,maxpoint=R.reduction_rows(point,sq);fpok &= pointok
    for j in range(152):
        exact=math.fsum(float(v) for v in sb[j]);bound=(18*u/(1-18*u))*abs(exact)+18*float(np.nextafter(np.float32(0),np.float32(1)))
        fpok &= abs(float(ss[j])-exact)<=bound
    bce=M.bce_elements(f['self15_logits_FP32'],d['cached_self15_target_FP32'].astype('?'))
    bnd=16*eps*(1+abs(f['self15_logits_FP32'].astype('f8'))+abs(bce))
    fpok &= bool(np.all(abs(sb.astype('f8')-bce)<=bnd)) and V.bits(pair,point+np.float32(.1)*ss)
    reductions={key:M.native_fp32_reduction_evidence(e[key],value) for key,value in [('L_state_FP32',point),('L_self15_sum_FP32',ss),('L_native_pair_FP32',pair)]}
    fpok &= all(v['within_envelope'] for v in reductions.values())
    prob=M.probability(f['self15_logits_FP32']);fpok &= bool(np.all(abs(e['self15_probability15_FP32'].astype('f8')-prob)<=8*u))
    fpok &= bool(np.all(abs(e['self15_penalty_rate_FP32'].astype('f8')+e['self15_probability15_FP32'].astype('f8').sum(1))<=18*u*15)) and V.bits(e['self15_penalty_component_FP32'],e['self15_penalty_rate_FP32']*np.float32(.02))
    gate('saved_FP32_point_BCE_sum15_pair_reductions_and_proxy',fpok)
    details['FP32']={'point_row_max_difference':maxpoint,'BCE_max_fraction_of_fixed16eps':float((abs(sb.astype('f8')-bce)/bnd).max()),'reductions':reductions}
    scores=C.score(f['self15_logits_FP32'],d['cached_self15_target_FP32']);delta=D.astype('f8');raw=f['prediction58_FP32'].astype('f8')-d['target58_FP32'].astype('f8')
    blocks={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),'native_LogR':(43,46,'rad'),'v':(46,49,'m/s'),'omega':(49,52,'rad/s')}
    rawblocks={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),'R_raw9':(43,52,'raw matrix entry'),'v':(52,55,'m/s'),'omega':(55,58,'rad/s')}
    stats={};masks=V.cohort_masks(expected)
    ratio=lambda a,b:a/b if b else None
    def self_summary(indices):
        s=C.summary(scores,list(indices));q=s['confusion_total'];ff=s['confusion_per_feature']
        s.update(positive_recall=ratio(q['TP'],q['TP']+q['FN']),positive_precision=ratio(q['TP'],q['TP']+q['FP']),negative_specificity=ratio(q['TN'],q['TN']+q['FP']),per_feature_positive_recall=[ratio(a,a+b) for a,b in zip(ff['TP'],ff['FN'])],per_feature_positive_precision=[ratio(a,a+b) for a,b in zip(ff['TP'],ff['FP'])])
        return s
    for name,mask in masks:
        self15=self_summary(np.flatnonzero(mask));state=M.fmean((delta[mask]/scale.astype('f8'))**2);per=M.vector_statistics(delta[mask],'Mixed units: per-feature values only')
        stats[name]={'groups':int(mask.sum()),'native52_scaled_MSE_F64_of_saved_FP32_delta':state,
            'state_native_delta':{k:M.vector_statistics(delta[mask,a:b],unit) for k,(a,b,unit) in blocks.items()},
            'raw58_difference':{k:M.vector_statistics(raw[mask,a:b],unit) for k,(a,b,unit) in rawblocks.items()},
            'raw_R9_difference':M.vector_statistics(raw[mask,43:52],'raw matrix entry; vector_RMS is Frobenius_RMS'),
            'native52_per_feature':{k:v for k,v in per.items() if k.startswith('per_feature_') or k=='unit'},
            'cached_self15':self15,'native_pair_F64_rescore':state+.1*self15['BCE_sum15_mean_rows_F64'],
            'captured_native_pair_row_FP32_mean_F64':M.fmean(pair[mask])}
    combined=read(RUN/'all52_common_model_cohort_statistics.json');commonkeys=set(stats['ALL'])
    old_roles=('unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency')
    oldok=all(combined[role]=={name:{k:v for k,v in q.items() if k in commonkeys} for name,q in reference['statistics'][role].items()} for role in old_roles)
    gate('all_new13_full_F64_perfeature_metrics_and_old39_exact_projection',V.equivalent(combined[ROLE],stats) and oldok and set(combined)==set(old_roles+(ROLE,)) and all(len(v)==13 for v in combined.values()))
    rows=list(csv.DictReader((RUN/'all608_common_model_row_metrics.csv').open(encoding='utf-8')))
    oldrows=list(csv.DictReader(check(reference['outputs']['row_CSV']).open(encoding='utf-8')))
    csvok=len(rows)==608 and len(oldrows)==456 and all(row=={k:old[k] for k in row} for row,old in zip(rows[:456],oldrows))
    for j,row in enumerate(rows[456:]):
        s=self_summary([j]);conf=s['confusion_total'];v={'native52_scaled_MSE_F64':M.fmean((delta[j]/scale.astype('f8'))**2),'cached15_BCE_mean_F64':s['BCE_mean_entry_F64'],'cached15_Brier_F64':s['Brier_mean_entry_F64'],'cached15_pred_intensity':s['pred_intensity_mean'],'cached15_true_intensity':s['target_intensity_mean'],'captured_native_pair_row_FP32':float(pair[j]),**{'cached15_'+k:conf[k] for k in ('TP','FP','FN','TN')}}
        v['native_pair_F64_rescore']=v['native52_scaled_MSE_F64']+.1*s['BCE_sum15_mean_rows_F64']
        v.update({k+'_native_delta_vector_norm':math.sqrt(math.fsum(float(x*x) for x in delta[j,a:b])) for k,(a,b,_) in blocks.items()})
        csvok &= row['role']==ROLE and int(row['row_id'])==j and all(row[k]==str(ids[j][k]) for k in ('path','k','mode','group_id')) and all(V.equivalent(float(row[k]),float(val)) for k,val in v.items())
    cohortrows=list(csv.DictReader((RUN/'all52_common_model_cohort_metrics.csv').open(encoding='utf-8')))
    csvok &= len(cohortrows)==52
    for row in cohortrows:
        s=combined[row['role']][row['cohort']];q=s['cached_self15']
        values={'groups':s['groups'],'native52_scaled_MSE_F64':s['native52_scaled_MSE_F64_of_saved_FP32_delta'],'cached15_BCE_mean_F64':q['BCE_mean_entry_F64'],'cached15_Brier_F64':q['Brier_mean_entry_F64'],'native_pair_F64_rescore':s['native_pair_F64_rescore'],'cached15_recall':q['positive_recall'],'cached15_precision':q['positive_precision'],'Pen_proxy_error_mean_F64':q['Pen_proxy_penalty_error_F64']['mean']}
        csvok &= all((row[k]=='' if val is None else V.equivalent(float(row[k]),float(val))) for k,val in values.items())
    gate('all52_cohort_and608_row_CSV_complete_metrics_and_nulls',csvok)
    before,after=arrays(RUN/'before_RNG_arrays.npz'),arrays(RUN/'after_RNG_arrays.npz')
    rngok=set(before)==set(after) and all(V.bits(v,after[k]) for k,v in before.items()) and read(RUN/'before_RNG_python_numpy.json')==read(RUN/'after_RNG_python_numpy.json')
    gate('raw_RNG_bytes_headers_and_source_backed_inference_invariants',rngok and all(v is True for v in r['checks'].values()) and r['counts']=={'trusted_CPU_torch_loads':2,'native_WM_constructors':1,'native_FW_attempted':1,'native_FW_returned':1,'predicted_rows':152,'aux_head_FW':0,'raw_deepcopies':0,'native_difference_calls':1,'AD_optimizer_Actor_MJB_Env_physics_GPU':0})
    assert not any(k.split('.')[0] in ('torch','mujoco','warp','dex_supermimic','mjlab') for k in sys.modules)
    write(OUT/'independent_new13_statistics.json',stats);write(OUT/'FP32_arithmetic_evidence.json',details)
    result={'schema':'full151105_native_public_only_common_saved_recheck:1','status':'PASS_SAVED_COMMON4ROLE_152_13COHORT_ARITHMETIC' if all(checks.values()) else 'FAILED_SAVED_COMMON_METRICS_NO_RETRY','engineering_saved_arithmetic_qualified':all(checks.values()),'auditor':pin(__file__),'scalar_reference_auditor':pin(helper),'producer_report':pin(RUN/'report.json'),'protocol':r['protocol'],'passed':sum(checks.values()),'total':len(checks),'checks':checks,'new_statistics':pin(OUT/'independent_new13_statistics.json'),'FP32_arithmetic':pin(OUT/'FP32_arithmetic_evidence.json'),'source_files_ZIP':len(r['source_pins']),'counts_source_backed':r['counts'],'reviewer_Torch_checkpoints_nativeLog_NN_FW_AD_MJB_physics_GPU':0,
        'scope':['Reviewer authored new producer: separate frozen scalar/fsum arithmetic, not independent experimental author.','Raw saved inputs/labels/predictions/D52/operands/RNG and current sourceZIP rechecked. Native LogR/weights/ctor/FW/norm before-after facts remain source-backed; no model replay or unexported raw buffer claims.','Raw nonrot D52 target-pred sign rechecked; full52 FP32 arithmetic consistency is not a physical gradient or SO3 accuracy certificate.','All known old152 fitted training entries and all13 overlapping cohorts, not ego/held-out/task/Actor performance. Cache15~15ms distinct from geometryPOST20. No optimization-total or winner comparison.']}
    write(OUT/'report.json',result)
    print(json.dumps({'report':pin(OUT/'report.json'),'passed':result['passed'],'total':result['total'],'actual_models_FW':0}))


if __name__=='__main__':main()
