"""One closed donor-gid/flat72/null fixture; no actual arrays or NN calls."""
import csv,hashlib,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
AUDITOR=ROOT/'scripts/audit_public_self15_H8_saved_cpu_v2.py'
AUDITOR_SHA='f76fe44eb74c7a672989aadfaaa86b3c7f8874b344ca5cb904d0926ece108219'

def pin(path):return {'path':str(Path(path).resolve(strict=True)),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def main():
    if AUDITOR_SHA is None or pin(AUDITOR)['sha256']!=AUDITOR_SHA:raise ValueError('Unfrozen auditor source')
    spec=importlib.util.spec_from_file_location('new_public15_H8_saved_fixture',AUDITOR);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    out=a.AUDITOR_PREP;out.mkdir(exist_ok=False)
    sources={'auditor':pin(AUDITOR),'checker':pin(__file__),'independent_H8_math':pin(a.MATH_PATH),'saved_utilities':pin(a.UTIL_PATH),'typed_norm_helper':pin(a.NORM_PATH)}
    write(out/'source_before_fixture.json',{'source_pins':sources,'actual_saved_arrays_read':False})
    identities=[];labels=[]
    for pi,path in enumerate(('A0','A2')):
        for k in range(150):
            gid=226*pi+k;g={'group_id':gid,'path':path,'mode':'donor','k':k,'branch_id':None,'source_frame':200+k,'terminal_observation':k+1,
                'first_physical_pre_index':4*gid,'last_physical_index':4*gid+3,'post20_GPU_row_index':1808+gid,'valid_next_policy_decision':k<149,'no_u150':True}
            raw=[0]*25;raw[2]=pi;target=[0.,float(pi)]+[0.]*13
            identities.append(g);labels.append({**g,'raw_found25':raw,'contact15':target,'label_time':'lastsolver_preintegration~kdt+.015','state_time':'postintegration_kdt+.020'})
        for bid in range(76):identities.append({'group_id':226*pi+150+bid,'mode':'baseline'})
    starts=np.tile(np.arange(143,dtype=np.int64),2);paths=np.repeat(np.arange(2,dtype=np.int64),143)
    rows=226*paths[:,None]+starts[:,None]+np.arange(8,dtype=np.int64)
    mapping={'window_transition_row_indices286x8':rows,'donor_row_indices2x150':np.asarray([np.arange(150),np.arange(226,376)],np.int64),'window_path_index286':paths}
    packet=a.expanded_labels(labels,identities,mapping);target=packet['target_self15_286x8x15_FP32']
    rollouts={role:{'self15_logits_FP32':np.zeros((286,8,15),np.float32)} for role in a.ROLES}
    tables=a.expected_cached72(rollouts,target,paths);allrow=tables[0];empty=tables[1];positive=tables[2]
    one=(len(tables)==72 and packet['donor_source_group_ids300'][150]==226 and rows[-1,-1]==375
        and int(packet['donor_label_repetition_count300'].sum())==2288 and (packet['donor_label_repetition_count300']>0).all()
        and target[:143,:,1].sum()==0 and target[143:,:,1].sum()==1144
        and abs(allrow['BCE_mean_entry_F64']-math.log(2))<=4*np.finfo(np.float64).eps and allrow['Brier_mean_entry_F64']==.25
        and allrow['confusion_total']['zero_logit_tie']==4290 and allrow['confusion_total']['TP']==0 and allrow['confusion_total']['FN']==143
        and empty['positive_recall'] is None and empty['positive_precision'] is None and positive['positive_recall']==0.
        and positive['positive_precision'] is None and positive['per_feature_positive_recall'][0] is None and positive['per_feature_positive_recall'][1]==0.)
    scores=a.score_cached(np.zeros((286,15),np.float32),target[:,0])
    one &= set(scores)=={'logits15_FP32','probability15_F64','BCE15_F64','Brier15_F64','decision15_bool','truth15_bool','zero_logit_tie15_bool',
        'pred_intensity_F64','target_intensity_F64','intensity_error_F64','pred_proxy_penalty_F64','target_proxy_penalty_F64','proxy_penalty_error_F64'}
    one &= (scores['logits15_FP32'].dtype==np.float32 and np.all(scores['probability15_F64']==.5) and np.all(scores['Brier15_F64']==.25)
        and np.all(abs(scores['BCE15_F64']-math.log(2))<=4*np.finfo(np.float64).eps) and not scores['decision15_bool'].any()
        and scores['zero_logit_tie15_bool'].all() and np.all(scores['pred_intensity_F64']==7.5))
    write(out/'synthetic_flat72.json',tables)
    with (out/'synthetic_flat72.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(tables[0]));writer.writeheader()
        for row in tables:writer.writerow({key:json.dumps(value,allow_nan=False) if isinstance(value,(list,dict)) else value for key,value in row.items()})
    one &= a.close(tables,a.read(out/'synthetic_flat72.json')) and a.csv_close(out/'synthetic_flat72.csv',tables)
    one &= not any(k.split('.')[0] in ('torch','warp','mujoco','dex_supermimic','mjlab') for k in sys.modules)
    mixed={'error_floor':np.asarray([.125,.5],np.float32),'norm.count':np.asarray(7.25,np.float64),'delta_norm.count':np.asarray(-0.,np.float64)}
    expected=hashlib.sha256();names={np.dtype('f4'):'torch.float32',np.dtype('f8'):'torch.float64'}
    for key,value in sorted(mixed.items()):expected.update(key.encode());expected.update(str((names[value.dtype],tuple(value.shape))).encode());expected.update(np.ascontiguousarray(value).tobytes())
    digest=a.norm_digest(mixed);dtype_changed=dict(mixed);dtype_changed['norm.count']=mixed['norm.count'].astype(np.float32)
    signed_zero_changed=dict(mixed);signed_zero_changed['delta_norm.count']=np.asarray(0.,np.float64)
    ulp_changed=dict(mixed);ulp_changed['norm.count']=np.asarray(np.nextafter(mixed['norm.count'],np.float64(np.inf)))
    integer_changed=dict(mixed);integer_changed['norm.count']=np.asarray(7,np.int64);rejected=False
    try:a.norm_digest(integer_changed)
    except ValueError:rejected=True
    two=(digest==expected.hexdigest() and digest!=a.norm_digest(dtype_changed) and digest!=a.norm_digest(signed_zero_changed) and digest!=a.norm_digest(ulp_changed) and rejected)
    checks={'single_gid_gap_expansion_flat72_closed_BCE_ties_null_and_97b_keys':bool(one),'mixed_FP32_FP64_header_shape_rawbytes_no_cast_sensitive_to_dtype_signedzero_ULP':bool(two)}
    if not all(checks.values()):raise ValueError('Closed saved-layout/typed-digest fixture failed')
    r={'schema':'full151105_public_self15_H8_saved_auditor_contract:1','checks':checks,
        'passed':sum(checks.values()),'total':len(checks),'source_pins':sources,'actual_saved_arrays_model_FW_AD_physics':0,'scope':'Synthetic label/window map/scalar saved arithmetic and mixed FP32/FP64 raw normalizer digest; no actual source packet, prediction, model or native Log qualification'}
    write(out/'report.json',r);write(out/'READY.json',{'schema':'full151105_public_self15_H8_saved_auditor_READY:1','source_pins':sources,'cpu_report':pin(out/'report.json'),'cpu_checks':{'passed':r['passed'],'total':r['total']}})
    print(json.dumps({'report':pin(out/'report.json'),'READY':pin(out/'READY.json'),'passed':r['passed'],'total':r['total']}));return 0

if __name__=='__main__':raise SystemExit(main())
