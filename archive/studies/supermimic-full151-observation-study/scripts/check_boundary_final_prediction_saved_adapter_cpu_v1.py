"""One small nonzero-head closed case through the independent saved adapter."""
import importlib.util,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'scripts/audit_boundary_components_final_saved_cpu_v1.py'
s=importlib.util.spec_from_file_location('final_saved_adapter',P);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def main():
    out=ROOT/'outputs/full151105_boundary_final_prediction_saved_adapter_preparation_v1';out.mkdir(exist_ok=False)
    sources={'auditor':a.pin(P),'checker':a.pin(__file__),'math':a.pin(a.MATH),'prior_saved_utils':a.pin(a.BASE)}
    for k,v in sources.items():(out/(k+'_source.py')).write_bytes(Path(v['path']).read_bytes())
    a.dump(out/'source_before_fixture.json',{'source_pins':sources,'actual_prediction_arrays_read':False})
    x=np.zeros((1,58),dtype='f4');x[:,43:52]=np.eye(3,dtype='f4').reshape(9);x[:,55:58]=[1,2,3];pred=x.copy();pred[:,52:55]=[1,2,3];pred[:,55:58]=[2,3,4]
    z=np.asarray([[[1,2,3,1,1,1],[-1,-2,-3,-1,-1,-1]]],dtype='f4');J=np.asarray([[[4,6,8],[-4,-6,-8]]],dtype='f8');K=np.asarray([[[2,6,12],[-2,-6,-12]]],dtype='f8');rv=np.asarray([[.5,1,1.5]]);rw=np.asarray([[.25,.5,.75]])
    ref=a.M.auxiliary(x,pred,z,np.ones(52,dtype='f4'),2.,np.diag([2.,3.,4.]),np.zeros(3),.5,J,K,rv,rw)
    f={'prediction58_FP32':pred,'self15_logits_FP32':np.zeros((1,15),dtype='f4'),'z12_FP32':z,'mode_logits2_FP32':np.zeros((1,2),dtype='f4')}
    d={'effective_s52_FP32':np.ones(52,dtype='f4'),'boundary_constraint_witness':np.asarray([[True,False]]),'target58_FP32':pred-np.float32(.25),'J_palm_F64':J,'K_palm_F64':K,'r_v_saved_m_s_F64':rv,'r_w_saved_rad_s_F64':rw}
    e={**ref,'native_point_delta52_FP32':np.full((1,52),.25,dtype='f4'),'mode_probability2_FP32':np.full((1,2),.5,dtype='f4'),'self15_penalty_rate_FP32':np.full(1,-7.5,dtype='f4'),'self15_penalty_component_FP32':np.full(1,np.float32(-7.5)*np.float32(.02),dtype='f4')}
    q=a.summary(f,e,d,np.zeros((1,15)),np.asarray([True]));one=q['native52_scaled_MSE_F64_of_saved_FP32_delta']==1/16 and abs(q['impulse_z_MSE_F64']-4/3)<1e-14 and abs(q['saved_defect_consistency_MSE_F64']-4.375/6)<1e-14
    one &= q['mode']['HO']['FN']==q['mode']['nonhand']['TN']==q['mode']['HO']['exact_zero_logit_ties']==1 and q['mode_Brier_F64']==.25 and abs(q['mode_BCE_F64_of_saved_FP32_logits']-math.log(2))<1e-15
    one &= abs(q['boundary_impulse_error']['J_HO']['vector_RMS']-math.sqrt(12))<1e-14 and q['boundary_impulse_error']['J_two_source_total']['vector_RMS']==q['boundary_impulse_error']['K_two_source_total']['vector_RMS']==0
    one &= abs(q['observed_defect_match_error']['linear']['vector_RMS']-math.sqrt(3.5))<1e-14 and q['self15_initial_drift']['logit']['component_RMSE']==0
    one &= not any(k.split('.')[0] in ('torch','mujoco','warp','dex_supermimic','mjlab') for k in sys.modules)
    a.dump(out/'report.json',{'schema':'full151105_boundary_final_prediction_saved_adapter_CPU_contract:1','checks':{'nonzero_z_cancelled_total_impulses_nonzero_saved_defects_units_and_mode_ties':bool(one)},'passed':int(one),'total':1,'source_pins':sources,'actual_data_Torch_models_FW_AD_MJB_physics_GPU':0,'limits':'Synthetic captured D and boundary operands only; no native Log or model output qualification.'})
    a.dump(out/'READY.json',{'schema':'full151105_boundary_final_prediction_saved_adapter_READY:1','source_pins':sources,'cpu_report':a.pin(out/'report.json'),'cpu_checks':{'passed':int(one),'total':1},'actual_prediction_arrays_read':False})
    print(json.dumps({'ready':a.pin(out/'READY.json'),'report':a.pin(out/'report.json'),'auditor':a.pin(P),'passed':int(one),'total':1},indent=2));return 0 if one else 2
if __name__=='__main__':raise SystemExit(main())
