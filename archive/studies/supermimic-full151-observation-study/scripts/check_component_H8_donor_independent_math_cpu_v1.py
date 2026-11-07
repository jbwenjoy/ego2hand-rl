"""Three synthetic H8 boundary/recurrence/unit cases; no real arrays or models."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src/full151_gradient/component_H8_donor_independent_math_v1.py'
OUT=ROOT/'outputs/full151105_component_H8_independent_math_preparation_v1'


def pin(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def dump(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)


def identities():
    # Deliberately interleave restored rows before donor rows: first300 is wrong.
    rows=[]
    for path in ('A0','A2'):
        for k in range(150):
            modes=[('baseline' if k<12 else 'direction',k)] if k<76 else []
            modes.append(('donor',None))
            for mode,bid in modes:rows.append({'group_id':len(rows),'path':path,'mode':mode,'k':k,'branch_id':bid,'source_frame':200+k,'terminal_observation':k+1,'no_u150':True})
    return rows


def rejected(fn):
    try:fn();return False
    except ValueError:return True


def main():
    OUT.mkdir(exist_ok=False);sources={'math':pin(SRC),'checker':pin(__file__)}
    for name,path in [('math',SRC),('checker',Path(__file__))]:ast.parse(path.read_bytes());(OUT/(name+'_before_execution.py')).write_bytes(path.read_bytes())
    dump(OUT/'source_freeze.json',{'source_pins':sources,'actual_H8_input_or_prediction_reads':False})
    spec=importlib.util.spec_from_file_location('H8_math_numpy',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ids=identities();mapping=m.donor_window_mapping(ids);rr=mapping['window_transition_row_indices286x8'];dr=mapping['donor_row_indices2x150']
    wrong=[dict(r) for r in ids];wrong[dr[1,0]].update(mode='baseline',branch_id=99)
    checks={'all286_keymapped_windows_no_restored_crosspath_or_u150':rr.shape==(286,8) and rr[0,0]!=0 and np.array_equal(rr[-1],dr[1,142:150]) and mapping['actual_control_phase286x8'].max()==149 and mapping['target_observation_phase286x8'].max()==150 and all(ids[int(r)]['path']==('A0' if wi<143 else 'A2') and ids[int(r)]['mode']=='donor' for wi in range(286) for r in rr[wi]) and rejected(lambda:m.donor_window_mapping(wrong))}
    pre=np.zeros((452,58),dtype='f4');target=pre.copy();u=np.zeros((452,20),dtype='f4')
    for pi in range(2):
        for k,row in enumerate(dr[pi]):pre[row]=np.float32(1000*pi+k);target[row]=np.float32(1000*pi+k+1);u[row,0]=np.float32(k)
    held=np.repeat(u[:,None,:],4,axis=1);guards=m.source_byte_guards(pre,target,u,held,mapping)
    broken=held.copy();broken[dr[0,0],2,1]=np.float32(-0.)
    initial=np.ones((286,58),dtype='f4');pred=np.empty((286,8,58),dtype='f4');x=initial.copy()
    for h in range(8):x=np.float32(2)*x+np.float32(1);pred[:,h]=x
    states=m.expected_recursive_state_inputs(initial,pred);xi=np.arange(46,dtype='f4');dynamics=m.expected_dynamics_inputs(states,xi)
    trace=m.trace_byte_guards(initial,pred,states,dynamics[...,:58],xi,dynamics)
    stale=np.repeat(initial[:,None,:],8,axis=1)
    checks['all298_links_300held_bytes_and_predicted_state_in_both_inputs']=guards['native_full58_link_count']==298 and guards['held_actual_control_intervals']==300 and rejected(lambda:m.source_byte_guards(pre,target,u,broken,mapping)) and trace['captured_full_dynamics_bytes_checked'] and rejected(lambda:m.trace_byte_guards(initial,pred,states,stale,xi)) and rejected(lambda:m.trace_byte_guards(initial,pred,stale,stale,xi))
    truth=np.zeros((286,8,58),dtype='f4');prediction=truth.copy();prediction[...,40:43]=[3,4,0];prediction[...,43:52]=np.eye(3,dtype='f4').reshape(9)
    prediction[...,0]=1;truth[...,0]=np.float32(1e-8)
    stats=m.raw58_error_statistics(prediction,truth,mapping['window_path_index286']);allstats=stats['statistics']['all_windows_all_steps'];blocks=allstats['blocks']
    checks['F64_first_difference_six_original_units_vector_vs_component_RMS']=stats['signed_error286x8x58_F64'][0,0,0]!=float(np.float32(1)-np.float32(1e-8)) and blocks['p']['vector_RMS']==5 and abs(blocks['p']['component_RMSE']-5/np.sqrt(3))<1e-15 and blocks['R_raw9']['Frobenius_RMS']==np.sqrt(3) and blocks['R_raw9']['units']=='dimensionless' and allstats['dependent_state_comparisons']==2288 and len(stats['statistics']['per_horizon'])==8 and all(stats['statistics']['by_path_per_horizon'][path]['8']['dependent_state_comparisons']==143 for path in ('A0','A2'))
    checks={k:bool(v) for k,v in checks.items()};require_backend_absent=not any(k.split('.')[0] in ('torch','warp','mujoco','dex_supermimic','mjlab') for k in sys.modules)
    if not require_backend_absent:raise ValueError('Unexpected framework in independent math fixture')
    report={'schema':'full151105_component_H8_independent_math_CPU_proof:1','source_pins':sources,'passed':sum(checks.values()),'total':len(checks),'checks':checks,
      'actual_H8_input_or_prediction_reads':0,'model_FW_AD_Env_physics_GPU':0,'scope':'Synthetic bookkeeping/recurrence/statistics only, not actual H8 accuracy or independence.'}
    dump(OUT/'report.json',report);ready={'source_pins':sources,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']}}
    dump(OUT/'READY.json',ready);print(json.dumps({'READY':pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
