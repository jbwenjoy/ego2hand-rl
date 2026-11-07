"""One stride5/OR/tie/negative-Pen-proxy closed case; no actual data."""
import importlib.util,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'scripts/audit_boundary_cached_self15_saved_cpu_v1.py'
s=importlib.util.spec_from_file_location('independent_cached15',P);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def main():
    out=ROOT/'outputs/full151105_cached_self15_independent_preparation_v1';out.mkdir(exist_ok=False)
    sources={'auditor':a.pin(P),'checker':a.pin(__file__),'scalar_math':a.pin(a.V.MATH),'saved_utility':a.pin(a.BASE)}
    for k,v in sources.items():(out/(k+'_source.py')).write_bytes(Path(v['path']).read_bytes())
    a.dump(out/'source_before_fixture.json',{'source_pins':sources,'actual_reward_labels_logits_read':False})
    raw=[[0]*25];raw[0][1]=2;raw[0][3]=1;raw[0][4]=3;y=a.mapped15(raw);target=np.stack((y,a.mapped15([[0]*25])))
    one=np.array_equal(np.flatnonzero(y),[0,10]) and y.dtype==np.float32
    scores=a.score(np.zeros((2,15),dtype='f4'),target);z=a.summary(scores,[0,1]);one &= z['rows']==2 and z['decisions']==30 and z['confusion_total']=={'TP':0,'FP':0,'FN':2,'TN':28,'zero_logit_tie':30,'tie_positive_target':2,'tie_negative_target':28}
    one &= abs(z['BCE_mean_entry_F64']-math.log(2))<1e-15 and z['Brier_mean_entry_F64']==.25 and z['pred_intensity_mean']==7.5 and z['target_intensity_mean']==1
    one &= z['intensity_error']['mean']==6.5 and abs(z['intensity_error']['RMS']-math.sqrt(43.25))<1e-15 and abs(z['Pen_proxy_penalty_error_F64']['mean']+.13)<1e-15
    one &= not any(k.split('.')[0] in ('torch','mujoco','warp','dex_supermimic','mjlab') for k in sys.modules)
    a.dump(out/'report.json',{'schema':'full151105_cached_self15_independent_pure_contract:1','checks':{'stride5_OR_not_countsum_zero_ties_negative_F64_BCE_intensity_Pen_sign':bool(one)},'passed':int(one),'total':1,'source_pins':sources,'actual_reward_logits_arrays_models_FW_AD_physics_GPU':0})
    a.dump(out/'READY.json',{'schema':'full151105_cached_self15_independent_READY:1','source_pins':sources,'cpu_report':a.pin(out/'report.json'),'cpu_checks':{'passed':int(one),'total':1},'actual_reward_labels_logits_read':False})
    print(json.dumps({'ready':a.pin(out/'READY.json'),'report':a.pin(out/'report.json'),'auditor':a.pin(P),'passed':int(one),'total':1},indent=2));return 0 if one else 2
if __name__=='__main__':raise SystemExit(main())
