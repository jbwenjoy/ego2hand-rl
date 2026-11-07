"""Light saved-only integrity/science summary; no Torch/NN/native imports."""
from pathlib import Path
import json,hashlib
import numpy as np
H=Path(__file__).resolve().parent;R=H/'completed_v1';OUT=H/'saved_summary_v1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def check(v):
    if pin(v['path'])!=v:raise ValueError('File pin mismatch: '+v['path'])
def raw(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def classify(a,b):return np.where(np.all(a-b==0,axis=-1),0,np.where(np.all(a.astype(np.float32)==b.astype(np.float32),axis=-1),1,2)).astype(np.int8)
def load(v):
    with np.load(v['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def finite_stats(a):
    a=np.asarray(a);return dict(count=len(a),median=float(np.median(a)) if len(a) else None,max=float(np.max(a)) if len(a) else None)
def main():
    if OUT.exists():raise ValueError('Fresh review output')
    OUT.mkdir();r=read(R/'report.json');t=read(H/'process_run1/terminal.json');p=read(r['protocol']['path'])
    for v in [r['protocol'],r['checkpoint'],t['source'],t['protocol'],t['stdout'],t['stderr'],*r['outputs'].values()]:check(v)
    C=load(r['outputs']['CHECK_all1476_predictions_and_targets.npz']);S=load(r['outputs']['CHECK_all1440_side_response_arrays.npz']);V=load(r['outputs']['VAL114_predictions.npz'])
    end=read(r['outputs']['endpoint_statistics.json']['path']);sides=read(r['outputs']['side_response_statistics.json']['path']);signals=read(r['outputs']['all20x22_native_signal_strata.json']['path']);objects=read(r['outputs']['all20_object_native_signal_strata.json']['path']);central=read(r['outputs']['all720_central_pair_records.json']['path']);ids=read(r['outputs']['all1476_identities.json']['path'])
    known=C['known'];valid=S['defined'];si=S['side_row'];bi=S['baseline_row']
    checks={}
    checks['terminal_source_protocol_fixed5000_pins']=(t['returncode']==0 and t['automatic_retry'] is False and t['stderr']['bytes']==0 and t['protocol']==r['protocol'] and t['source']==p['sources']['evaluator'] and r['checkpoint']['sha256']=='344bac4568f57b8d97973bda06e5568b97fe10e13864f8b3d5a326bcf98dc9a7')
    checks['exact2FW1586rows_zero_update_or_native']=(r['counts']['own_checkpoint_CPU_loads']==r['counts']['NN_constructors']==1 and r['counts']['NN_forward_calls']==r['counts']['structured_decode_calls']==2 and r['counts']['NN_rows']==1586 and r['counts']['NN_AD_optimizer_training_native_GPU']==0 and r['counts']['TRAIN_FIT_TEST_NPZ_reads']==0 and all(v is True for v in r['immutability'].values()))
    checks['all1476endpoints1472known4unknown']=(known.dtype==np.bool_ and known.shape==(1476,) and int(known.sum())==1472 and len(ids)==1476 and len(np.unique(C['branch_id']))==1476 and all(np.isfinite(C[k][known]).all() and np.isnan(C[k][~known]).all() for k in ['raw222_FP32','head178_F64','prediction58_F64','meanFn22_N','coverage_probability22_F64','normalized_input366_FP32','model_physical_delta52','true_head178_F64']))
    checks['all1440sides1436known720central716defined']=(valid.dtype==np.bool_ and valid.shape==(1440,) and int(valid.sum())==1436 and raw(valid,known[si]&known[bi]) and len(central)==720 and sum(v['defined'] for v in central)==716 and all(np.isfinite(S[k][valid]).all() and np.isnan(S[k][~valid]).all() for k in ['native_slope178_actualdu','model_slope178_actualdu','native_J_response','model_J_response']))
    du=C['actual_u20'][si]-C['actual_u20'][bi];j=S['joint'];expected=du[np.arange(1440),j]
    nf=C['normalized_input366_FP32'];jcol=j+58;ndu=nf[si].astype(np.float64)[np.arange(1440),jcol]-nf[bi].astype(np.float64)[np.arange(1440),jcol]
    # Exact raw-control and stored response comparisons; NN-normalized physical
    # scaling is source/checkpoint evidence, with its raw encoded inputs retained.
    controls=(raw(expected,S['actual_du_F64']) and np.all(abs(expected[valid])>=1e-6) and np.array_equal(np.unique(j),np.arange(20)) and raw(S['model_response178'],C['head178_F64'][si]-C['head178_F64'][bi]) and raw(S['native_response178'],C['true_head178_F64'][si]-C['true_head178_F64'][bi]))
    controls=controls and raw(S['rawphysical_FP32_endpoint_du_F64'],C['actual_u20'][si].astype(np.float32).astype(np.float64)[np.arange(1440),j]-C['actual_u20'][bi].astype(np.float32).astype(np.float64)[np.arange(1440),j])
    controls=controls and np.all(S['actual_normalized_NN_FP32_du_physical_F64'][valid]!=0) and np.all(ndu[valid]!=0)
    checks['all20axes_actual_and_encoded_steps_retained']=controls
    expected_class=np.full((1440,22),-1,dtype=np.int8);expected_class[valid]=classify(C['true_J22x6_F64'][si[valid]],C['true_J22x6_F64'][bi[valid]])
    oc=np.full(1440,-1,dtype=np.int8);oc[valid]=classify(C['native_physical_delta52'][si[valid],40:52],C['native_physical_delta52'][bi[valid],40:52])
    signal_ok=raw(expected_class,S['perpair_native_signal_class']) and raw(oc,S['native_object_signal_class']) and len(signals)==20*22*3 and len(objects)==20*3
    for row in signals:
        cl=['EXACT_F64_ZERO','NONZERO_F64_FP32_ENDPOINT_COLLAPSED','FP32_ENDPOINT_RESOLVED'].index(row['signal_class']);mask=valid&(j==row['joint'])&(expected_class[:,row['pair_index']]==cl)
        signal_ok=signal_ok and row['count']==int(mask.sum())
    for row in objects:
        cl=['EXACT_F64_ZERO','NONZERO_F64_FP32_ENDPOINT_COLLAPSED','FP32_ENDPOINT_RESOLVED'].index(row['signal_class']);signal_ok=signal_ok and row['count']==int((valid&(j==row['joint'])&(oc==cl)).sum())
    checks['all_zero_precision_strata_and_all_pair_counts_retained']=signal_ok
    cohorts=sorted({q['cohort'] for q in end});checks['all13endpoint_cohorts39roles_plus_VAL114']=(len(cohorts)==13 and len(end)==39 and all({v['role'] for v in end if v['cohort']==c}=={'model5000','CV','ballistic'} for c in cohorts) and V['prediction58_F64'].shape==(114,58) and len(np.unique(V['row_id']))==114 and np.isfinite(V['prediction58_F64']).all())
    endpoint_summary=[dict(cohort=q['cohort'],role=q['role'],state=q['state'],head178_blocks=q['head178_blocks']) for q in end]
    sideall=next(x for x in sides if x['cohort']=='ALL_KNOWN_SIDES');blocks=['q_rad','qd_rad_s','COM_m','world_dR_rad','Jforce_Ns','Jmoment_Nms']
    ratios={name:(float(e['RMS']/n['RMS']) if n['RMS'] else None) for name,e,n in zip(blocks,sideall['model_response_error_blocks'],sideall['native_response_blocks'])}
    nativeCOM=S['native_delta52_response'][:,40:43];modelCOM=S['model_delta52_response'][:,40:43];zero=valid&np.all(nativeCOM==0,axis=1)
    magnitude=np.linalg.norm(modelCOM[zero]/expected[zero,None],axis=1)
    object_counts={name:int((valid&(oc==cl)).sum()) for cl,name in enumerate(['EXACT_F64_ZERO','NONZERO_F64_FP32_ENDPOINT_COLLAPSED','FP32_ENDPOINT_RESOLVED'])}
    checks['COM_science_anchor_matches_saved_endpoint']=(sum(object_counts.values())==1436 and int(zero.sum())==836)
    summary=dict(schema='phone_geometry_momentum_knownstate_CHECK_saved_summary:1',status='PASS_LIGHT_SAVED_REVIEW_NOT_NEW_INFERENCE' if all(checks.values()) else 'FAIL_LIGHT_SAVED_REVIEW_PRESERVED',
        reviewer_is_source_author=True,producer_report=pin(R/'report.json'),terminal=pin(H/'process_run1/terminal.json'),auditor=pin(__file__),checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),
        all_endpoint_cohort_states_and_head_errors=endpoint_summary,all_side_response_cohorts=sides,all20_object_signal_strata=objects,
        all20x22_pair_signal_counts=[{k:v[k] for k in ['joint','pair_index','signal_class','count']} for v in signals],full_perpair_statistics_pin=r['outputs']['all20x22_native_signal_strata.json'],
        response_error_RMS_divided_by_native_response_RMS=ratios,
        ratio_interpretation='Action-independent zero-response prior has ratio1. World dR finite-response error ratio>1 is worse than that prior despite small endpoint orientation error; this is not a trueJacobian or policy result.',
        native_COM_exactzero_sides=int(zero.sum()),model_COM_secant_norm_on_native_exactzero_m_per_rad=finite_stats(magnitude),
        all_object_delta52_signal_counts=object_counts,
        main_observation='Small known-state novel-action endpoint errors coexist with residual spurious object responses and weak/worse finite orientation response fitting. Both observations are retained; no winner or causal-zero certificate.',
        limitations=['CHECK source states were fitted originalTRAIN states; new actions do not create unseen-state or ego generalization evidence.',
            'Do not strictly compare the zero-response median with the earlier12-state raw80 study: states, conditioning set, model and recipe differ.',
            'Combined geometry/momentum/CF baseline has no matched ablation attribution, native/task-gradient or Actor/CHORD benefit claim.',
            'Immutability, actual inference call counts and NN-normalized step rescaling remain source/producer evidence; no weight/Torch/native reload here.'],
        review_counts=dict(saved_prediction_response_VAL_NPZ_reads=3,NN_Torchload_forward_AD_train_native=0,raw_target_NPZ_reads=0),
        retained_full_records=dict(endpoints=1476,known1472=1472,unknown4=4,sides=1440,known_sides=1436,central=720,known_central=716),full_producer_output_pins=r['outputs'])
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps(dict(report=pin(OUT/'report.json'),passed=summary['passed'],total=summary['total'],ratios=ratios,object_counts=object_counts,COMzero=int(zero.sum()),ghost=finite_stats(magnitude))))
    if not all(checks.values()):raise SystemExit(1)
if __name__=='__main__':main()
