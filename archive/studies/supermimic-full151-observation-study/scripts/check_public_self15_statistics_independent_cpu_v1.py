"""One pure closed fixture; never invokes any saved-data main or NN code."""
import hashlib,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'scripts/describe_public_self15_predictions_saved_cpu_v1.py';EXPECTED='fb4465bcf12ae98db966420ce24b25961673f523353119ffa4ec9a91843cc2c8'
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def require(x,s):
    if not x:raise ValueError(s)
def dump(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    out=ROOT/'outputs/full151105_public_self15_statistics_independent_preparation_v1';out.mkdir(exist_ok=False);require(pin(P)['sha256']==EXPECTED,'Final Root source SHA')
    spec=importlib.util.spec_from_file_location('public15_pure_metrics',P);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    common=a.module(a.COMMON,a.COMMON_SHA,'public15_old_pure');sensor=a.module(a.SENSOR,a.SENSOR_SHA,'public15_sensor_pure')
    sources={'statistics':pin(P),'checker':pin(__file__),'common_pure':pin(a.COMMON),'sensor_pure':pin(a.SENSOR),'scientific_contract':pin(ROOT/'docs/full151_boundary_public_self15_final_evaluation_contract_v1.md')}
    for k,v in sources.items():(out/(k+Path(v['path']).suffix)).write_bytes(Path(v['path']).read_bytes())
    dump(out/'source_before_fixture.json',{'source_pins':sources,'actual_arrays_labels_weights_read':False})
    ids=[]
    for pi,path in enumerate(('A0','A2')):
        for ki,k in enumerate((0,49,99,149)):
            for off in range(19):ids.append({'row_id':len(ids),'group_id':226*pi+150+19*ki+off,'physical_rows':[4*(226*pi+150+19*ki+off)+s for s in range(4)],'path':path,'k':k,'mode':'baseline' if off<3 else 'direction'})
    cohort=common.cohorts(ids);mask=cohort[0][1];zero3=np.zeros((152,3));zero23=np.zeros((152,2,3));y=np.zeros((152,15),dtype='f4');y[:,:3]=1
    f={'prediction58_FP32':np.zeros((152,58),dtype='f4'),'self15_logits_FP32':np.zeros((152,15),dtype='f4'),'z12_FP32':np.zeros((152,2,6),dtype='f4'),'mode_logits2_FP32':np.zeros((152,2),dtype='f4')}
    d={'effective_s52_FP32':np.ones(52,dtype='f4'),'target58_FP32':np.zeros((152,58),dtype='f4'),'boundary_constraint_witness':np.zeros((152,2),dtype='?'),'J_palm_F64':zero23,'K_palm_F64':zero23,'r_v_saved_m_s_F64':zero3,'r_w_saved_rad_s_F64':zero3}
    pair=np.float32(np.float32(.0625)+np.float32(.1)*np.float32(15*np.float32(math.log(2))))
    loss={'native_point_delta52_FP32':np.full((152,52),.25,dtype='f4'),'native_point_squared_scaled52_FP32':np.full((152,52),.0625,dtype='f4'),'mode_probability2_FP32':np.full((152,2),.5,dtype='f4'),'self15_probability15_FP32':np.full((152,15),.5,dtype='f4'),'self15_penalty_rate_FP32':np.full(152,-7.5,dtype='f4'),'self15_penalty_component_FP32':np.full(152,np.float32(-7.5)*np.float32(.02),dtype='f4'),'J_hat_palm_F64':zero23,'K_hat_palm_F64':zero23,'r_v_hat_F64':zero3,'r_w_hat_F64':zero3,'z_target_F64':np.zeros((152,2,6)),'native_pair_per_row_FP32':np.full(152,pair,dtype='f4')}
    c=common.summarize(f,loss,d,np.zeros((152,15)),mask);ss=sensor.summary(sensor.score_matrix(f['self15_logits_FP32'],y,np),list(range(152)),np);result=a.add_native15(c,ss,loss,mask)
    one=result[a.PRIMARY[0]]==1/16 and abs(result['cached_self15']['BCE_mean_entry_F64']-math.log(2))<1e-14 and result['cached_self15']['Brier_mean_entry_F64']==.25
    one &= result['cached_self15']['positive_recall']==0. and result['cached_self15']['positive_precision'] is None and result['cached_self15']['per_feature_positive_recall']==[0.,0.,0.]+[None]*12
    one &= result['cached_self15']['confusion_total']['FN']==456 and result['cached_self15']['confusion_total']['TN']==1824 and result['cached_self15']['confusion_total']['zero_logit_tie']==2280
    one &= abs(result['native_pair_F64_rescore']-(.0625+.1*15*math.log(2)))<1e-14 and abs(result['full_public_F64_rescore']-(.0625+.1*15*math.log(2)+math.log(2)))<1e-14 and result['captured_native_pair_row_FP32_mean_F64']==float(pair)
    empty=sensor.summary(sensor.score_matrix(f['self15_logits_FP32'],np.zeros_like(y),np),list(range(152)),np);q=a.add_native15(dict(c),empty,loss,mask)
    one &= q['cached_self15']['positive_recall'] is q['cached_self15']['positive_precision'] is None and q['cached_self15']['per_feature_positive_recall']==[None]*15
    one &= [int(m.sum()) for _,m in cohort]==[152,76,76]+[19]*8+[24,128] and not any(k.split('.')[0] in ('torch','mujoco','warp','dex_supermimic','mjlab') for k in sys.modules)
    require(pin(P)['sha256']==EXPECTED,'No reviewed source changes')
    report={'schema':'full151105_public_self15_statistics_independent_pure_contract:1','checks':{'constant_native52_zeros_logit_threepositives_BCE_Brier_empty_recall_sum15_rowpair_precision':bool(one)},'passed':int(one),'total':1,'source_pins':sources,'actual_arrays_weights_labels_NN_FW_AD_MJB_physics_GPU':0,'limits':'Synthetic captured operands; no native Log/model/kernel or actual prediction qualification.'}
    dump(out/'report.json',report);dump(out/'READY.json',{'schema':'full151105_public_self15_statistics_independent_pure_READY:1','source_pins':sources,'cpu_report':pin(out/'report.json'),'cpu_checks':{'passed':int(one),'total':1},'actual_evaluation_authorized':False})
    print(json.dumps({'report':pin(out/'report.json'),'READY':pin(out/'READY.json'),'passed':int(one),'total':1},indent=2));return 0 if one else 2
if __name__=='__main__':raise SystemExit(main())
