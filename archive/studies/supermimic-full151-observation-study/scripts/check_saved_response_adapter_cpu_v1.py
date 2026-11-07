"""Pure synthetic adapter proof; no actual response outputs or backend imports."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ADAPTER=ROOT/'scripts/audit_saved_action_response_cpu_v1.py'
OUT=ROOT/'outputs/full151105_saved_response_adapter_preparation_v1'


def main():
    OUT.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('saved_response_adapter',ADAPTER);a=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a);m=a.module(a.MATH,a.MATH_SHA)
    pins={'auditor':a.pin(ADAPTER),'math':a.pin(a.MATH),'checker':a.pin(__file__)}
    for key,path in [('auditor',ADAPTER),('math',a.MATH),('checker',Path(__file__))]:
        ast.parse(path.read_bytes());(OUT/(key+'_source_before_execution.py')).write_bytes(path.read_bytes())
    a.dump(OUT/'predata_freeze.json',{'source_pins':pins,'actual_saved_response_outputs_read':False})
    sx=np.zeros((8,58),dtype='f4');sx[:,43:52]=np.eye(3,dtype='f4').reshape(9)
    su=np.zeros((8,20),dtype='f4');u=np.broadcast_to(su[:,None],(8,19,20)).copy()
    for ei,eps in enumerate(m.EPS):
        for j,q in enumerate(m.JOINTS):
            u[:,3+8*ei+2*j,q]=-np.float32(eps);u[:,4+8*ei+2*j,q]=np.float32(eps)
    # One zero actual denominator; retained with NaNs, never filtered away.
    u[0,3:5]=0
    # Original clipping is numeric, even when requested +0 and actual -0 differ in bits.
    su[1,0]=-np.float32(.002);u[1]=su[1]
    for ei,eps in enumerate(m.EPS):
        for j,q in enumerate(m.JOINTS):
            unit=np.eye(20,dtype='f4')[q];u[1,3+8*ei+2*j]=su[1]-np.float32(eps)*unit;u[1,4+8*ei+2*j]=su[1]+np.float32(eps)*unit
    u[1,4,0]=np.float32(-0.)
    minus=np.empty((8,4,2,20),dtype='f4');plus=np.empty_like(minus);clip=np.empty((8,4,2),dtype=bool)
    for s in range(8):
        for j,q in enumerate(m.JOINTS):
            for ei,eps in enumerate(m.EPS):
                minus[s,j,ei]=u[s,3+8*ei+2*j];plus[s,j,ei]=u[s,4+8*ei+2*j]
                unit=np.eye(20,dtype='f4')[q]
                clip[s,j,ei]=not(np.array_equal(plus[s,j,ei],su[s]+np.float32(eps)*unit) and np.array_equal(minus[s,j,ei],su[s]-np.float32(eps)*unit))
    p=np.broadcast_to((4*u.sum(-1,dtype='f4'))[...,None],(8,19,58)).copy()
    y=np.broadcast_to((2*u.sum(-1,dtype='f4'))[...,None],(8,19,58)).copy()
    pre=np.broadcast_to(sx[:,None],(8,19,58)).copy()
    # Allow other18 signedzero numerically but preserve its differing raw bits.
    pre[0,6,40]=np.float32(-0.)
    physical={'snapshot_x':sx,'snapshot_u':su,'branch_u':u.reshape(152,20).copy(),'direction__centers':su.copy(),
       'direction__minus':minus,'direction__plus':plus,'direction__d_eff':(plus.astype('f8')-minus.astype('f8'))/(2*np.asarray(m.EPS)[None,None,:,None]),
       'direction__center_offset':(plus.astype('f8')+minus.astype('f8'))*.5-su[:,None,None].astype('f8'),'direction__clipped':clip}
    state={k:np.zeros(shape,dtype='f4') for k,shape in [('pre_x452_FP32',(452,58)),('predicted_x452_FP32',(452,58)),('target_x452_FP32',(452,58)),('actual_u452_FP32',(452,20))]}
    identities=[{'group_id':i,'branch_id':None} for i in range(452)];rows=[]
    # Deliberately reverse placement in all452 to prove identity join, not slicing.
    ix=np.arange(451,299,-1,dtype='i8').reshape(8,19)
    for s in range(8):
        path=('A0','A2')[s//4];k=a.PHASES[s%4]
        for off in range(19):
            i=int(ix[s,off]);bid=s*19+off;base=off<3;j=None if base else (off-3)%8//2;ei=None if base else (off-3)//8
            row={'row_id':bid,'snapshot_id':s,'path':path,'k':k,'j':j,'epsilon_index':ei,'epsilon_rad':None if base else m.EPS[ei],
               'sign':0 if base else (-1,1)[(off-3)%2],'baseline_repeat':off if base else None,'joint_index':None if base else m.JOINTS[j],
               'type':'baseline' if base else 'direction','actualu':u[s,off].astype('f8').tolist()}
            rows.append(row);identities[i]={'group_id':i,'branch_id':bid,'path':path,'k':k,'mode':row['type']}
            for key,value in [('pre_x452_FP32',pre),('predicted_x452_FP32',p),('target_x452_FP32',y),('actual_u452_FP32',u)]:state[key][i]=value[s,off]
    cm={path:{'actuator_ids':list(range(20,40)),'joint_ids':list(range(40,60))} for path in ('A0','A2')}
    raw,g,meta=a.source_join(state,identities,rows,physical,cm,m);r,e=a.expected_decomposition(raw,m);stats,features=a.expected_stats(e,m)
    valid=e['secant_defined_mask']
    checks={'source452_identity_join_reverseplacement_not_slice':a.bits(raw['branch_group_indices'],ix) and a.bits(raw['predicted_y152_FP32'],p),
       'all14_guard_masks_pass_closedform':len(g)==14 and all(v.all() for v in g.values()),
       'numeric_clip_signedzero_keeps_original_semantics':not raw['legacy_clipped'][1,0,0] and not a.bits(plus[1,0,0],su[1]+np.float32(.002)*np.eye(20,dtype='f4')[0]),
       'other18_numeric_guard_retains_raw_signedzero':g['same_original_snapshot_numeric18'][0,6,0] and not a.bits(pre[0,6],sx[0]),
       'closedform_response_secants_two_four':np.all(e['physical_coordinate_secant_F64'][valid]==2) and np.all(e['model_coordinate_secant_F64'][valid]==4),
       'zeroactualdenominator_stays_one_mask_and58NaNs':int((~valid).sum())==1 and np.isnan(e['physical_coordinate_secant_F64'][0,0,0]).all(),
       'all64_records_s_j_epsilon_order':len(a.pair_records(raw,e,m))==64 and [(v['snapshot_id'],v['j'],v['epsilon_index']) for v in a.pair_records(raw,e,m)]==[(s,j,ei) for s in range(8) for j in range(4) for ei in range(2)],
       'all58_features_and_six_units_counts':set(stats)==set(m.BLOCKS) and features['feature_indices']==list(range(58)) and features['defined_pair_count']==63,
       'strictJSON_does_not_equate_bool_int_or_list_tuple':not a.strict_json({'a':1},{'a':True}) and not a.strict_json([1],(1,)),
       'stored_roundoff_relation_not_new_zero_gate':np.isfinite(e['centered_residual_identity_roundoff_F64']).all()}
    ids=r['pair_identities'];markers=np.arange(64,dtype='f8');remap=a.native_pair_order(markers,ids)
    checks['explicit_pair_permutation_uses_all_coordinates']=all(remap[x['snapshot_id'],x['joint_ordinal'],x['epsilon_index']]==i for i,x in enumerate(ids)) and remap[0,0,1]==4 and remap[0,1,0]==1
    try:a.native_pair_order(markers,[ids[0]]+ids[:-1]);duplicate_rejected=False
    except ValueError:duplicate_rejected=True
    checks['pair_duplicate_metadata_rejected']=duplicate_rejected
    wrong=raw.copy();wrong['pre_visible152_FP32']=pre.copy();wrong['pre_visible152_FP32'][0,0,0]=np.float32(-0.)
    blocked,_=a.expected_decomposition(wrong,m)
    checks['q40_bit_mismatch_blocks_all_no_subsetting']=not blocked['fixed_visible_state_interpretation_allowed']
    altered=rows.copy();altered[3]={**rows[3],'epsilon_index':True}
    try:a.source_join(state,identities,altered,physical,cm,m);badaccepted=True
    except ValueError:badaccepted=False
    checks['branch_type_numeric_bool_mismatch_rejected']=not badaccepted
    checks['no_actualdata_or_backend_import']=not any(x in sys.modules for x in ('torch','mujoco','warp','scipy','dex_supermimic','mjlab'))
    checks={k:bool(v) for k,v in checks.items()}
    report={'schema':'full151105_saved_response_adapter_CPU_proof:1','source_pins':pins,'passed':sum(checks.values()),'total':len(checks),'checks':checks,
       'actual_response_outputs_read':0,'model_FW_AD_MJB_Env_GPU_physics':0,'synthetic_only':True}
    a.dump(OUT/'report.json',report);a.dump(OUT/'READY.json',{'source_pins':pins,'cpu_report':a.pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']}})
    print(json.dumps({'report':a.pin(OUT/'report.json'),'passed':report['passed'],'total':report['total'],'checks':checks},indent=2))
    return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
