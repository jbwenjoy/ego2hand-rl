"""Saved-only fixed AFTER1000 public-native15 metrics; no model construction."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
COMMON=ROOT/'scripts/describe_boundary_component_predictions_saved_cpu_v1.py'
COMMON_SHA='7487e16b819271fd3b6d3d6562d4802fd97216afa4e3f666f705948f88cce036'
SENSOR=ROOT/'src/full151_gradient/boundary_self15_cached_saved105_v1.py'
SENSOR_SHA='97b74e0173bb6d95b55c59d924db7c5502c83cf610907c6c3aa357d54808da5e'
ROLES=('unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency')
PRIMARY=('native52_scaled_MSE_F64_of_saved_FP32_delta','mode_BCE_F64_of_saved_FP32_logits','impulse_z_MSE_F64','saved_defect_consistency_MSE_F64')


def pin(p):return {'path':str(Path(p).absolute()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load_npz(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def module(path,expected,name):
    assert pin(path)['sha256']==expected
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvfile(p,rows):
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def add_native15(common,sensor_summary,loss,mask):
    """Only comparable F64 rescoring; captured FP32 scalars stay separate."""
    s=dict(sensor_summary);q=s['confusion_total']
    s['positive_recall']=q['TP']/(q['TP']+q['FN']) if q['TP']+q['FN'] else None
    s['positive_precision']=q['TP']/(q['TP']+q['FP']) if q['TP']+q['FP'] else None
    s['negative_specificity']=q['TN']/(q['TN']+q['FP']) if q['TN']+q['FP'] else None
    f=s['confusion_per_feature']
    s['per_feature_positive_recall']=[tp/(tp+fn) if tp+fn else None for tp,fn in zip(f['TP'],f['FN'])]
    s['per_feature_positive_precision']=[tp/(tp+fp) if tp+fp else None for tp,fp in zip(f['TP'],f['FP'])]
    common['cached_self15']=s
    self_sum=s['BCE_sum15_mean_rows_F64']
    common['native_pair_F64_rescore']=common[PRIMARY[0]]+.1*self_sum
    common['full_public_F64_rescore']=common['native_pair_F64_rescore']+common[PRIMARY[1]]+common[PRIMARY[2]]
    common['captured_native_pair_row_FP32_mean_F64']=float(loss['native_pair_per_row_FP32'][mask].astype('f8').mean())
    return common


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    common=module(COMMON,COMMON_SHA,'_public15_common_metrics');sensor=module(SENSOR,SENSOR_SHA,'_public15_cached_sensor_metrics')
    run=a.run_dir.resolve(strict=True);parent=read(run/'report.json')
    assert parent['schema']=='full151105_boundary_public_self15_final_evaluation_CPU_analysis:1'
    assert parent['status']=='COMPLETED_FIXED_INITIAL_AND_TWO_PUBLIC_SELF15_AFTER1000_EVAL_NOT_CONFIRMATION'
    assert parent['passed']==parent['total']==len(parent['checks']) and all(v is True for v in parent['checks'].values())
    for v in parent['outputs'].values():assert Path(v['path']).parent.samefile(run) and pin(v['path'])['sha256']==v['sha256']
    data=load_npz(run/'prepared_inputs_and_context.npz');ids=read(run/'group152_identities.json')
    assert [g['row_id'] for g in ids]==list(range(152)) and len({g['group_id'] for g in ids})==152
    assert np.array_equal(np.asarray([g['group_id'] for g in ids]),data['source_group_id152'])
    assert np.array_equal(np.asarray([g['physical_rows'] for g in ids]),data['source_substep_rows152x4'])
    y=data['cached_self15_target_FP32'];assert y.shape==(152,15) and y.dtype==np.float32 and np.isin(y,[0,1]).all()
    f={r:load_npz(run/(r+'_forward152.npz')) for r in ROLES};loss={r:load_npz(run/(r+'_descriptive_losses.npz')) for r in ROLES}
    for r in ROLES:
        for k,shape in [('prediction58_FP32',(152,58)),('self15_logits_FP32',(152,15)),('z12_FP32',(152,2,6)),('mode_logits2_FP32',(152,2))]:
            assert f[r][k].dtype==np.float32 and f[r][k].shape==shape and np.isfinite(f[r][k]).all()
        for k,shape in [('native_point_squared_scaled52_FP32',(152,52)),('mode_probability2_FP32',(152,2)),('self15_probability15_FP32',(152,15)),
                        ('self15_penalty_rate_FP32',(152,)),('self15_penalty_component_FP32',(152,)),('self15_BCE_elements_FP32',(152,15)),('native_pair_per_row_FP32',(152,))]:
            assert loss[r][k].dtype==np.float32 and loss[r][k].shape==shape and np.isfinite(loss[r][k]).all()
        assert np.array_equal(loss[r]['self15_target_FP32'].view('u4'),y.view('u4'))
    cohort=common.cohorts(ids);initial=f[ROLES[0]]['self15_logits_FP32'].astype('f8')
    stats={};rows=[];flat=[]
    for r in ROLES:
        scores=sensor.score_matrix(f[r]['self15_logits_FP32'],y,np)
        stats[r]={}
        for group,mask in cohort:
            s=common.summarize(f[r],loss[r],data,initial,mask)
            s=add_native15(s,sensor.summary(scores,np.flatnonzero(mask),np),loss[r],mask);stats[r][group]=s
            row={'role':r,'cohort':group,'groups':s['groups'],**{k:s[k] for k in PRIMARY},'native_pair_F64_rescore':s['native_pair_F64_rescore'],
                 'full_public_F64_rescore':s['full_public_F64_rescore'],'cached15_BCE_mean_F64':s['cached_self15']['BCE_mean_entry_F64'],
                 'cached15_Brier_F64':s['cached_self15']['Brier_mean_entry_F64'],'cached15_positive_recall':s['cached_self15']['positive_recall'],
                 'cached15_pred_intensity':s['cached_self15']['pred_intensity_mean'],'cached15_true_intensity':s['cached_self15']['target_intensity_mean'],
                 'Pen_proxy_error_mean_F64':s['cached_self15']['Pen_proxy_penalty_error_F64']['mean']}
            for name in common.BLOCKS:row[name+'_vector_RMS']=s['state_native_delta'][name]['vector_RMS']
            flat.append(row)
        for row in common.row_table(r,f[r],loss[r],data,ids,initial):
            j=row['row_id'];q=sensor.summary(scores,[j],np)
            row.update(cached15_BCE_mean_F64=q['BCE_mean_entry_F64'],cached15_Brier_F64=q['Brier_mean_entry_F64'],
                       cached15_pred_intensity=q['pred_intensity_mean'],cached15_true_intensity=q['target_intensity_mean'],
                       cached15_TP=q['confusion_total']['TP'],cached15_FP=q['confusion_total']['FP'],cached15_FN=q['confusion_total']['FN'],cached15_TN=q['confusion_total']['TN'])
            row['native_pair_F64_rescore']=row['native52_scaled_MSE_F64']+.1*q['BCE_sum15_mean_rows_F64']
            row['full_public_F64_rescore']=row['native_pair_F64_rescore']+row['mode_BCE_F64']+row['impulse_z_MSE_F64']
            row['captured_native_pair_row_FP32']=float(loss[r]['native_pair_per_row_FP32'][j])
            rows.append(row)
    assert len(flat)==39 and len(rows)==456
    out=a.output.absolute();out.mkdir(exist_ok=False)
    csvfile(out/'all39_public15_model_cohort_summary.csv',flat);csvfile(out/'all456_public15_model_row_metrics.csv',rows)
    scalar_keys=('L_state_FP32','L_self15_sum_FP32','L_native_pair_FP32','L_mode_FP32','L_imp_F64','L_cons_F64','L_public','L_total')
    report={'schema':'full151105_boundary_public_self15_saved_prediction_description:1','status':'COMPLETED_FIXED_PUBLIC15_AFTER1000_ALL152_NOT_GENERALIZATION',
            'producer':pin(run/'report.json'),'source':pin(__file__),'source_helpers':{'common':pin(COMMON),'sensor':pin(SENSOR)},
            'roles':list(ROLES),'cohorts':[g for g,_ in cohort],'statistics':stats,'captured_global_losses':{r:{k:float(loss[r][k]) for k in scalar_keys} for r in ROLES},
            'precision_scope':'F64 rescoring from captured FP32 nativeD/logits; native per-row FP32 loss scalars and captures are separately named. No native Log/model or original reward kernel replay.',
            'source_scope':'Known152 dependent training rows, one robot cached lineage, all13 overlapping cohorts. Cached15 lastsolver~15ms versus POST20 state, not post20 geometry or HO/human labels.',
            'comparison_scope':'Within this family compare identical common losses and individual physical/state/contact metrics. Consistency-weighted totals differ by arm; no automatic winner/checkpoint selection. Prior no-self15 public excludes15 and is a different objective, not directly comparable as Lpublic.',
            'classification':'Strict logits>0; zero-negative/ties counted. Empty-positive recall/precision denominators return null, not perfect scores.',
            'no_new_weights_models_NN_FW_AD_MJB_Env_physics_GPU':True,'task_CHORD_physicalgradient_ego_qualified':False,
            'outputs':{'cohort_CSV':pin(out/'all39_public15_model_cohort_summary.csv'),'row_CSV':pin(out/'all456_public15_model_row_metrics.csv')}}
    write(out/'report.json',report)
    preview={}
    for r in ROLES:
        s=stats[r]['ALL'];preview[r]={**{k:s[k] for k in PRIMARY},'p_vector_RMS_m':s['state_native_delta']['p']['vector_RMS'],
            'native_LogR_vector_RMS_rad':s['state_native_delta']['native_LogR']['vector_RMS'],'omega_vector_RMS_rad_s':s['state_native_delta']['omega']['vector_RMS'],
            'cached15_BCE':s['cached_self15']['BCE_mean_entry_F64'],'cached15_Brier':s['cached_self15']['Brier_mean_entry_F64'],
            'cached15_confusion':s['cached_self15']['confusion_total'],'pred_intensity':s['cached_self15']['pred_intensity_mean'],'true_intensity':s['cached_self15']['target_intensity_mean'],
            'Pen_proxy_error_mean':s['cached_self15']['Pen_proxy_penalty_error_F64']['mean']}
    print(json.dumps({'report':pin(out/'report.json'),'ALL':preview}))


if __name__=='__main__':main()
