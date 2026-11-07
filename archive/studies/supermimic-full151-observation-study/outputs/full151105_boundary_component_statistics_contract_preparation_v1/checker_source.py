"""One AST-only closed statistics fixture; no actual arrays or producer main."""
import ast,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/describe_boundary_component_predictions_saved_cpu_v1.py'
EXPECTED='7487e16b819271fd3b6d3d6562d4802fd97216afa4e3f666f705948f88cce036'
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def require(x,s):
    if not x:raise ValueError(s)
def dump(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)

def main():
    out=ROOT/'outputs/full151105_boundary_component_statistics_contract_preparation_v1';out.mkdir(exist_ok=False)
    require(pin(SOURCE)['sha256']==EXPECTED,'Final Root statistics source')
    before={'statistics':pin(SOURCE),'checker':pin(__file__),'contract':pin(ROOT/'docs/full151_boundary_component_final_evaluation_contract_v1.md')}
    (out/'statistics_source.py').write_bytes(SOURCE.read_bytes());(out/'checker_source.py').write_bytes(Path(__file__).read_bytes());dump(out/'source_before_fixture.json',{'source_pins':before,'actual_arrays_or_models_read':False})
    tree=ast.parse(SOURCE.read_text());names={'vector_stats','sigmoid','cohorts','summarize','row_table'}
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names or isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('ROLES','BLOCKS') for t in x.targets)]
    require({x.name for x in nodes if isinstance(x,ast.FunctionDef)}==names,'Only required pure functions extracted')
    env={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'frozen_statistics_pure_functions','exec'),env)
    ids=[]
    for pi,path in enumerate(('A0','A2')):
        for ki,k in enumerate((0,49,99,149)):
            for off in range(19):ids.append({'row_id':len(ids),'group_id':226*pi+150+19*ki+off,'path':path,'k':k,'mode':'baseline' if off<3 else 'direction'})
    masks=env['cohorts'](ids);require([int(m.sum()) for _,m in masks]==[152,76,76]+[19]*8+[24,128],'Closed 13 cohort counts')
    labels=np.zeros((152,2),dtype='?');labels[:76,0]=True;labels[76:,1]=True
    z=np.zeros((152,2,6),dtype='f4');z[:,0,:3]=[1,2,2];z[:,1,:3]=[-1,-2,-2];z[:,0,3:]=[2,0,0];z[:,1,3:]=[-2,0,0]
    zeros=np.zeros((152,3),dtype='f8');two=np.zeros((152,2,3),dtype='f8');a=np.float32(.25)
    f={'prediction58_FP32':np.zeros((152,58),dtype='f4'),'self15_logits_FP32':np.zeros((152,15),dtype='f4'),'z12_FP32':z,'mode_logits2_FP32':np.zeros((152,2),dtype='f4')}
    data={'effective_s52_FP32':np.ones(52,dtype='f4'),'boundary_constraint_witness':labels,'target58_FP32':np.zeros((152,58),dtype='f4'),'J_palm_F64':two.copy(),'K_palm_F64':two.copy(),'r_v_saved_m_s_F64':zeros.copy(),'r_w_saved_rad_s_F64':zeros.copy(),'source_group_id152':np.asarray([x['group_id'] for x in ids],dtype='i8')}
    losses={'native_point_delta52_FP32':np.full((152,52),a,dtype='f4'),'native_point_squared_scaled52_FP32':np.full((152,52),np.float32(1/16),dtype='f4'),'mode_probability2_FP32':np.full((152,2),.5,dtype='f4'),'self15_probability15_FP32':np.full((152,15),.5,dtype='f4'),'self15_penalty_rate_FP32':np.full(152,-7.5,dtype='f4'),'self15_penalty_component_FP32':np.full(152,np.float32(-7.5)*np.float32(.02),dtype='f4'),'r_v_hat_F64':zeros.copy(),'r_w_hat_F64':zeros.copy(),'z_target_F64':np.zeros((152,2,6),dtype='f8'),'J_hat_palm_F64':z[:,:,:3].astype('f8'),'K_hat_palm_F64':z[:,:,3:].astype('f8')}
    s=env['summarize'](f,losses,data,f['self15_logits_FP32'].astype('f8'),masks[0][1]);tol=1e-13
    require(s['native52_scaled_MSE_F64_of_saved_FP32_delta']==1/16 and abs(s['mode_BCE_F64_of_saved_FP32_logits']-math.log(2))<tol and s['mode_Brier_F64']==.25 and s['mode_Brier_from_captured_FP32_probability_F64_reduction']==.25,'Closed constant52 MSE and mode BCE/Brier')
    for name,(lo,hi,_) in env['BLOCKS'].items():
        v=s['state_native_delta'][name];require(v['component_RMSE']==.25 and abs(v['vector_RMS']-.25*math.sqrt(hi-lo))<tol and v['per_feature_signed_mean']==[.25]*(hi-lo),'Block component versus vector units')
    for name in ('HO','nonhand'):
        q=s['mode'][name];require([q[k] for k in ('TP','FP','TN','FN','exact_zero_logit_ties')]==[0,0,76,76,152],'Strict zero ties negative per source')
    require(s['mode']['both_bits_correct_groups']==0 and abs(s['impulse_z_MSE_F64']-13/6)<tol and s['saved_defect_consistency_MSE_F64']==0,'Both-bit and dimensionless impulse reductions')
    require(s['boundary_impulse_error']['J_HO']['vector_RMS']==3 and s['boundary_impulse_error']['K_HO']['vector_RMS']==2 and s['boundary_impulse_error']['J_two_source_total']['vector_RMS']==s['boundary_impulse_error']['K_two_source_total']['vector_RMS']==0,'Nonzero per-source errors retained despite total cancellation')
    drift=s['self15_initial_drift'];require(drift['logit']['component_RMSE']==0 and drift['mean_sum_sigmoid']==7.5 and drift['strict_positive_logit_bit_flips']==0 and drift['current_exact_zero_logit_ties']==2280 and drift['captured_FP32_mean_penalty_rate']==-7.5,'Initial self15 drift and separate penalty proxy')
    rows=[v for role in env['ROLES'] for v in env['row_table'](role,f,losses,data,ids,f['self15_logits_FP32'].astype('f8'))];require(len(rows)==456 and all(x['row_id']==j%152 and x['group_id']==ids[j%152]['group_id'] for j,x in enumerate(rows)),'Full row identity output')
    raw=SOURCE.read_text();require(all(k in raw for k in ('native_point_squared_scaled52_FP32','mode_probability2_FP32','self15_probability15_FP32','self15_penalty_rate_FP32','self15_penalty_component_FP32',"data['source_group_id152']","[r['row_id'] for r in ids] == list(range(152))")),'Capture keys and strict source-row ABI guards')
    require(not any(k.split('.')[0] in ('torch','mujoco','warp','dex_supermimic','mjlab') for k in sys.modules),'No model/native imports')
    report={'schema':'full151105_boundary_component_statistics_pure_contract:1','checks':{'closed52_units_mode_ties_sources_cancellation_13cohorts_and456rows':True},'passed':1,'total':1,'source_pins':before,'actual_arrays_payloads_models_FW_AD_physics_GPU':0,'limits':'Synthetic captured operands only, not native Log, loss/force provenance or actual prediction validation.'}
    dump(out/'report.json',report);dump(out/'READY.json',{'schema':'full151105_boundary_component_statistics_pure_READY:1','source_pins':before,'cpu_report':pin(out/'report.json'),'cpu_checks':{'passed':1,'total':1},'actual_data_evaluation_authorized':False});print(json.dumps({'report':pin(out/'report.json'),'ready':pin(out/'READY.json'),'passed':1,'total':1},indent=2))
if __name__=='__main__':main()
