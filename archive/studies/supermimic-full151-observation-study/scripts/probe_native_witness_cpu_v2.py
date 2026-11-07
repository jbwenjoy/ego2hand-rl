"""One fixed164-query local derivative probe; never optimize a trajectory."""
from pathlib import Path
import argparse,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151.io import checked,pin,read,write,all_pins,load_selected


def main(protocol_path):
    start=time.perf_counter();p=read(protocol_path)
    if p['schema']!='full151_native_witness_probe:1' or p['epsilon_rad']!=1e-5 or p['states']!=[17,335,489,330] or p['queries']!=164:raise ValueError('Fixed branch-probe contract required')
    for x in all_pins(p):checked(x)
    if checked(p['sources']['producer'])!=Path(__file__).resolve():raise ValueError('Wrong producer')
    out=Path(p['output']);out.mkdir(parents=True,exist_ok=False)
    report={'schema':'full151_native_witness_probe_receipt:1','status':'STARTED','protocol':pin(protocol_path),'checks':{},'cases':[],
      'physical_qualified':False,'global_contact_Jacobian_qualified':False,'all_future_SCP_branches_qualified':False,'limitations':p['limitations']}
    oracle=None
    try:
        from scipy.spatial.transform import Rotation
        import scipy.optimize as opt
        from full151.native_witness import FrozenNativeOracle
        ref=load_selected(p['inputs']['reference'],['planned_q601_rad','object_COM_world_m','object_rotation_w','mapped_object_pose_w'])
        saved=load_selected(p['inputs']['saved_geometry'],['selected_distance_m','generated_mask','tie_count'])
        xyzw=Rotation.from_matrix(ref['object_rotation_w']).as_quat();poses=np.c_[ref['object_COM_world_m'],xyzw[:,[3,0,1,2]]];poses[::4]=ref['mapped_object_pose_w']
        lp=opt.linprog
        oracle=FrozenNativeOracle(p['inputs']['model'],p['inputs']['actual_context'],p['sources']['native_geometry'],p['sources']['kinematics'])
        snapshots=out/'samples';snapshots.mkdir();(out/'source_before').mkdir()
        for name,x in p['sources'].items():(out/'source_before'/(name+Path(x['path']).suffix)).write_bytes(checked(x).read_bytes())
        with oracle:
            write(out/'query_metadata.json',oracle.query.metadata)
            np.savez_compressed(out/'catalog.npz',pairs=oracle.query.pairs,pair_kind=oracle.query.pair_kind,geom_names=np.asarray(oracle.query.names))
            for case,sample,pair in zip(p['case_names'],p['states'],p['pair_indices']):
                q=ref['planned_q601_rad'][sample];base=oracle.evaluate(q,poses[sample]);np.savez_compressed(snapshots/(case+'_base.npz'),**base)
                dist_plus=np.full((20,273),np.nan);dist_minus=dist_plus.copy();tie_plus=np.zeros((20,273),int);tie_minus=tie_plus.copy();normal_plus=np.full((20,273,3),np.nan);normal_minus=normal_plus.copy()
                counts_plus=np.zeros((20,273),int);counts_minus=counts_plus.copy();order_plus=np.full((20,2),-1,int);order_minus=order_plus.copy()
                def selected_order(v):
                    w=np.flatnonzero((v["pair_index"]==pair)&(v["native_contact_index"]==v["selected_native_index"][pair]))
                    return np.array([-1,-1]) if len(w)!=1 else v["geom_order"][w[0]]
                base_order=selected_order(base)
                qexcess=0.
                for j in range(20):
                    for label,sign,dv,tv,nv in [('plus',1,dist_plus,tie_plus,normal_plus),('minus',-1,dist_minus,tie_minus,normal_minus)]:
                        trial=q.copy();trial[j]+=sign*1e-5;v=oracle.evaluate(trial,poses[sample],with_jacobian=False)
                        np.savez_compressed(snapshots/(case+f'_{label}_j{j:02d}.npz'),**v)
                        dv[j]=v['selected_distance_m'];tv[j]=v['tie_count'];nv[j]=v['selected_normal_canonical_w']
                        (counts_plus if sign==1 else counts_minus)[j]=v['contact_count']
                        (order_plus if sign==1 else order_minus)[j]=selected_order(v)
                        qexcess=max(qexcess,float(np.maximum(np.maximum(oracle.query.robot.lower-trial,trial-oracle.query.robot.upper),0).max()))
                finite=np.isfinite(dist_plus[:,pair])&np.isfinite(dist_minus[:,pair])
                fd=np.full(20,np.nan);fd[finite]=(dist_plus[finite,pair]-dist_minus[finite,pair])/(2e-5)
                jac=base['selected_distance_jacobian_q'][pair]
                normal0=base['selected_normal_canonical_w'][pair]
                normal_stable=bool(np.all(np.einsum('ij,j->i',normal_plus[:,pair],normal0)>1-1e-6) and np.all(np.einsum('ij,j->i',normal_minus[:,pair],normal0)>1-1e-6))
                unique=bool(base['tie_count'][pair]==1 and np.all(tie_plus[:,pair]==1) and np.all(tie_minus[:,pair]==1))
                counts_stable=bool(np.all(counts_plus[:,pair]==base['contact_count'][pair]) and np.all(counts_minus[:,pair]==base['contact_count'][pair]))
                order_stable=bool(np.all(order_plus==base_order) and np.all(order_minus==base_order))
                errors=np.abs(fd-jac);error=None if not finite.all() or not np.isfinite(errors).all() else float(errors.max())
                selected_equal=[]
                for k in np.flatnonzero(base['generated_mask']):
                    w=np.flatnonzero((base['pair_index']==k)&(base['native_contact_index']==base['selected_native_index'][k]))
                    if len(w)!=1:raise ValueError('Selected witness not uniquely present in all-witness inventory')
                    selected_equal.append(float(np.max(np.abs(base['distance_jacobian_q_m_per_rad'][w[0]]-base['selected_distance_jacobian_q'][k]))))
                oldmask=saved['generated_mask'][sample];same=bool(np.array_equal(oldmask,base['generated_mask']) and np.max(np.abs(saved['selected_distance_m'][sample,oldmask]-base['selected_distance_m'][oldmask]))<=1e-12)
                row={'case':case,'sample':sample,'pair_index':pair,'base_tie_count':int(base['tie_count'][pair]),'all20_generated_both_sides':bool(finite.all()),'all20_unique':unique,'all20_normal_stable':normal_stable,'all20_contact_count_stable':counts_stable,'all20_selected_geom_order_stable':order_stable,
                  'max_FD_error_m_per_rad':error,'max_selected_vs_allwitness_J_error':max(selected_equal,default=0.),'matches_saved_base':same,'FD_max_qhard_excess_rad':qexcess,
                  'local_unique_gate_applicable':case.startswith('worst'),'local_unique_gate_pass':bool(finite.all() and unique and normal_stable and counts_stable and order_stable and error is not None and error<=1e-3)}
                active=(base['pair_index']==pair)&(np.abs(base['contactdist_m']-base['selected_distance_m'][pair])<=1e-9)
                grads=base['distance_jacobian_q_m_per_rad'][active]
                directional_plus=np.min(grads,axis=0) if len(grads) else np.full(20,np.nan);directional_minus=-np.max(grads,axis=0) if len(grads) else np.full(20,np.nan)
                np.savez_compressed(out/(case+'_FD.npz'),FD_central=fd,selected_J=jac,FD_plus=(dist_plus[:,pair]-base['selected_distance_m'][pair])/1e-5,
                  FD_minus=(dist_minus[:,pair]-base['selected_distance_m'][pair])/1e-5,tied_branch_pred_plus=directional_plus,tied_branch_pred_minus=directional_minus,base_tied_witness_J=grads,
                  selected_dist_plus_allpairs=dist_plus,selected_dist_minus_allpairs=dist_minus,ties_plus=tie_plus,ties_minus=tie_minus,contact_counts_plus=counts_plus,contact_counts_minus=counts_minus,selected_geom_order_plus=order_plus,selected_geom_order_minus=order_minus,base_selected_geom_order=base_order)
                report['cases'].append(row)
        report['checks']['original_LP_API_restored']=opt.linprog is lp
        # Reentering the oracle must preserve the same model and cumulative ledger.
        same_query=oracle.query;before=dict(oracle.allowed)
        with oracle:pass
        report['checks']['reentry_no_constructor_or_query']=oracle.query is same_query and before==oracle.allowed and opt.linprog is lp
        report['checks']['exact164_collision165_FK_COM']=oracle.allowed['mj_collision']==164 and oracle.allowed['mj_kinematics']==165 and oracle.allowed['mj_comPos']==165
        report['checks']['forbidden_dynamics_force_optimization_zero']=not any(oracle.forbidden.values())
        report['checks']['all4_saved_base_match']=all(x['matches_saved_base'] for x in report['cases'])
        report['checks']['allwitness_selected_formula_agrees']=all(x['max_selected_vs_allwitness_J_error']<=1e-12 for x in report['cases'])
        report['checks']['both_worst_unique_local_FD_gate']=all(x['local_unique_gate_pass'] for x in report['cases'][:2])
        report['checks']['ties_retained_as_diagnostics']=all(x['base_tie_count']>1 for x in report['cases'][2:])
        report['checks']['all_source_input_pins_unchanged']=all(checked(x) for x in all_pins(p))
        report['status']='PASS_LOCAL_WORST_BRANCHES_TIES_DIAGNOSTIC_ONLY' if all(report['checks'].values()) else 'FAILED_LOCAL_PROBE_NO_COUPLED_RUN_READY'
    except Exception as e:
        report['status']='PROBE_EXCEPTION_PRESERVED';report['exception']=repr(e);report['traceback']=traceback.format_exc()
    report['actual_calls']=None if oracle is None else {'allowed':oracle.allowed,'forbidden':oracle.forbidden}
    report['wall_s']=time.perf_counter()-start;report['artifacts']=[pin(f) for f in sorted(out.rglob('*')) if f.is_file()]
    write(out/'receipt.json',report)
    if report['status']=='PASS_LOCAL_WORST_BRANCHES_TIES_DIAGNOSTIC_ONLY':write(out/'READY_LOCAL_ONLY.json',{'receipt':pin(out/'receipt.json'),'global_or_physical_qualified':False,'scope':'Limited CPU geometry SCP exploration only; allfuturetrialbranches require fresh nonlinear audit.'})
    print(__import__('json').dumps({'status':report['status'],'cases':report['cases'],'actual_calls':report['actual_calls'],'receipt':pin(out/'receipt.json')}),flush=True)
    return 0 if report['status']=='PASS_LOCAL_WORST_BRANCHES_TIES_DIAGNOSTIC_ONLY' else 1


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--protocol',required=True);args=a.parse_args();raise SystemExit(main(Path(args.protocol)))
