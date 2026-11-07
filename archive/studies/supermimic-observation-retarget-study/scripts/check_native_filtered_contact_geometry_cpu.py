"""Four prespecified CPU local checks of filtered-contact overlap derivatives.

No optimizer, contact attraction, source robot labels, force or dynamics.
All difficult/tied/appearing contacts are reported; local unique-branch gates
are kept distinct from all-pair/nonsmooth diagnostics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import mujoco
import numpy as np

from native_filtered_contact_geometry import NativeFilteredContactGeometry


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):return {"path":str(path),"sha256":sha(path)}
def same_numpy_rng(a,b):return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]
def max_abs(a,mask):return float(np.nanmax(np.abs(a[mask]))) if np.any(mask) else None
def pair_record(g,k):return {"pair_index":int(k),"kind":str(g.pair_kind[k]),"names":[g.names[i] for i in g.pairs[k]]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("model","context","baseline","refined","output"):parser.add_argument("--"+flag,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    files=[args.model,args.context,args.baseline,args.refined,Path(__file__),Path(__file__).with_name("native_filtered_contact_geometry.py")]
    before={str(p):sha(p) for p in files}
    with np.load(args.baseline,allow_pickle=False) as z:base_q=z['joint_pos_rad'].copy();base_pose=z['mapped_object_pose_w'].copy()
    with np.load(args.refined,allow_pickle=False) as z:ref_q=z['joint_pos_rad'].copy();ref_pose=z['mapped_object_pose_w'].copy()
    numpy_rng=np.random.get_state();python_rng=random.getstate()
    forbidden={name:0 for name in ['mj_geomDistance','mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_contactForce']}
    def deny(name):
        def f(*a,**kw):forbidden[name]+=1;raise AssertionError('Forbidden '+name)
        return f
    for name in forbidden:setattr(mujoco,name,deny(name))
    g=NativeFilteredContactGeometry(args.model,args.context)
    checks={'actual_catalog_21_enabled210self63object':len(g.hand_ids)==21 and len(g.pairs)==273 and (g.pair_kind=='hand_self').sum()==210 and (g.pair_kind=='hand_object').sum()==63,
        'disabled_link1_mount_absent_from_catalog':len(g.disabled_hand_ids)==5 and not set(g.disabled_hand_ids)&set(g.pairs.ravel()) and all('mount' not in g.names[i] for i in g.pairs.ravel()),
        'original_MJB_numerical_options':g.query_options['ccd_tolerance']==1e-6 and g.query_options['ccd_iterations']==50}
    invalid_q=[0.,[0.],np.zeros(19),np.zeros((1,20)),np.r_[np.nan,np.zeros(19)],np.r_[np.inf,np.zeros(19)]]
    for j,q in enumerate(invalid_q):
        try:g.evaluate(q)
        except ValueError:checks[f'q_shape_finite_guard{j}']=True
        else:checks[f'q_shape_finite_guard{j}']=False
    for j,pose in enumerate([np.zeros(6),np.zeros(7),np.r_[np.zeros(3),[2.,0,0,0]],np.r_[np.nan,np.zeros(2),[1.,0,0,0]]]):
        try:g.set_object_pose(pose)
        except ValueError:checks[f'pose_shape_finite_unit_guard{j}']=True
        else:checks[f'pose_shape_finite_unit_guard{j}']=False
    cases=[('baseline0',0,base_q[0],base_pose[0]),('baseline75',75,base_q[75],base_pose[75]),('baseline150',150,base_q[150],base_pose[150]),('nominal_refined65',65,ref_q[65],ref_pose[65])]
    rows=[];arrays={};all_qualified_errors=[];all_native_order_types=set();all_ungenerated=0
    for name,frame,q,pose in cases:
        g.set_object_pose(pose);value=g.evaluate(q);count=g.collision_query_count
        checks[name+'_same_q_and_pose_cached']=g.evaluate(q.copy()) is value and g.collision_query_count==count
        g.set_object_pose(pose.copy());checks[name+'_same_pose_cache_preserved']=g.evaluate(q) is value and g.collision_query_count==count
        absent=~value.generated_mask;all_ungenerated+=int(absent.sum())
        checks[name+'_absent_is_NaN_not_gap0']=np.isnan(value.signed_distance_m[absent]).all() and np.isnan(value.selected_distance_jacobian[absent]).all() and not value.residual[absent].any() and not value.residual_jacobian[absent].any()
        checks[name+'_returned_arrays_readonly']=all(not getattr(value,k).flags.writeable for k in ['signed_distance_m','residual','residual_jacobian','generated_mask'])
        details=[]
        for eps in [1e-5,2e-5,5e-5]:
            fd_distance=np.full_like(value.selected_distance_jacobian,np.nan);fd_residual=np.empty_like(value.residual_jacobian)
            common=value.generated_mask.copy();unique=value.generated_mask&(value.tie_count==1);normal_stable=value.generated_mask.copy();hinge_stable=np.ones(len(g.pairs),dtype=bool);appearing=np.zeros(len(g.pairs),dtype=bool)
            for j in range(20):
                plus=q.copy();minus=q.copy();plus[j]+=eps;minus[j]-=eps
                p=g.evaluate(plus);n=g.evaluate(minus)
                both=value.generated_mask&p.generated_mask&n.generated_mask
                fd_distance[both,j]=(p.signed_distance_m[both]-n.signed_distance_m[both])/(2*eps)
                fd_residual[:,j]=(p.residual-n.residual)/(2*eps)
                common&=both;unique&=(p.tie_count==1)&(n.tie_count==1)
                with np.errstate(invalid='ignore'):
                    normal_stable&=(np.sum(p.selected_normal_a_to_b_w*value.selected_normal_a_to_b_w,axis=1)>1-1e-6)&(np.sum(n.selected_normal_a_to_b_w*value.selected_normal_a_to_b_w,axis=1)>1-1e-6)
                hinge_stable&=(p.active_mask==value.active_mask)&(n.active_mask==value.active_mask)
                appearing|=(~value.generated_mask)&(p.generated_mask|n.generated_mask)
            eligible=common&unique&normal_stable
            active_eligible=eligible&value.active_mask
            error=fd_distance-value.selected_distance_jacobian
            eligible_error=max_abs(error,eligible)
            passed=bool(eligible.any() and eligible_error<=1e-3)
            checks[name+'_unique_local_distance_FD_'+str(eps)]=passed
            all_qualified_errors.append(eligible_error)
            details.append({'epsilon_rad':eps,'all_common_generated_pair_count':int(common.sum()),'all_common_distance_max_abs_error_m_per_rad':max_abs(error,common),'all_active_residual_max_abs_error_per_rad':max_abs(fd_residual-value.residual_jacobian,value.active_mask),'unique_normalstable_generated_pairs':int(eligible.sum()),'unique_normalstable_active_pairs':int(active_eligible.sum()),'unique_normalstable_distance_max_abs_error_m_per_rad':eligible_error,'fixed_gate_m_per_rad':1e-3,'unique_local_gate_passed':passed,'unique_hingestable_residual_max_abs_error_per_rad':max_abs(fd_residual-value.residual_jacobian,eligible&hinge_stable),'generated_or_normal_or_tie_switch_pairs':[pair_record(g,k) for k in np.flatnonzero(value.generated_mask&~eligible)],'residual_hinge_crossing_pairs':[pair_record(g,k) for k in np.flatnonzero(~hinge_stable)],'contact_appearing_from_ungenerated_pairs':[pair_record(g,k) for k in np.flatnonzero(appearing)]})
            arrays[name+'_FD_distance_'+str(eps)]=fd_distance;arrays[name+'_FD_residual_'+str(eps)]=fd_residual;arrays[name+'_unique_normalstable_'+str(eps)]=eligible
        g.evaluate(q)  # Restore actual body kinematics before point-Jac sign checks.
        swap_errors=[];formula_errors=[];object_jac_norm=[]
        for k in np.flatnonzero(value.generated_mask):
            a,b=map(int,value.selected_native_geom_order[k]);position=value.selected_position_w[k];canonical=tuple(g.pairs[k]);normal=value.selected_normal_a_to_b_w[k] if (a,b)==canonical else -value.selected_normal_a_to_b_w[k]
            ja,jra=np.zeros((3,g.model.nv)),np.zeros((3,g.model.nv));jb,jrb=np.zeros((3,g.model.nv)),np.zeros((3,g.model.nv))
            mujoco.mj_jac(g.model,g.data,ja,jra,position,int(g.model.geom_bodyid[a]));mujoco.mj_jac(g.model,g.data,jb,jrb,position,int(g.model.geom_bodyid[b]));ja=ja[:,g.robot.va];jb=jb[:,g.robot.va]
            original=normal@(jb-ja);swapped=(-normal)@(ja-jb)
            swap_errors.append(float(np.max(np.abs(original-swapped))));formula_errors.append(float(np.max(np.abs(original-value.selected_distance_jacobian[k]))))
            if g.pair_kind[k]=='hand_object':
                if a in g.object_ids:object_jac_norm.append(float(np.linalg.norm(ja)));all_native_order_types.add('object_first')
                if b in g.object_ids:object_jac_norm.append(float(np.linalg.norm(jb)));all_native_order_types.add('hand_first')
        checks[name+'_opposite_geom_order_same_gap_gradient']=max(swap_errors,default=0)<1e-12 and max(formula_errors,default=0)<1e-12
        checks[name+'_fixed_object_robotJac_zero']=max(object_jac_norm,default=0)==0
        tied=np.flatnonzero(value.generated_mask&(value.tie_count>1))
        rows.append({'case':name,'source_frame':frame,'generated_pairs':int(value.generated_mask.sum()),'active_pairs':int(value.active_mask.sum()),'ungenerated_pairs':int(absent.sum()),'native_contact_records':value.native_contact_count,'ties':[dict(pair_record(g,k),contact_count=int(value.contact_count[k]),diagnostic_tie_count=int(value.tie_count[k]),second_contact_gap_m=float(value.second_contact_gap_m[k])) for k in tied],'FD':details,'opposite_geom_order_max_error':max(swap_errors,default=0),'fixed_object_robotJac_max_norm':max(object_jac_norm,default=0)})
        for key in ['generated_mask','signed_distance_m','selected_distance_jacobian','contact_count','tie_count','residual','residual_jacobian','active_mask','selected_native_geom_order']:arrays[name+'_'+key]=getattr(value,key)
    checks['actual_native_hand_object_orders_both_present']=all_native_order_types=={'object_first','hand_first'}
    checks['ungenerated_catalog_rows_actually_tested']=all_ungenerated>0
    checks['numpy_Python_RNG_unchanged']=same_numpy_rng(numpy_rng,np.random.get_state()) and python_rng==random.getstate()
    checks['no_geomDistance_dynamics_or_force_API']=not any(forbidden.values())
    checks['all_input_module_files_unchanged']={str(p):sha(p) for p in files}==before
    checks['clock_zero_velocity_zero']=g.data.time==0 and not g.data.qvel.any()
    # Options guard must also run on same-q cache returns.
    g.model.opt.ccd_iterations+=1
    try:g.evaluate(cases[-1][2])
    except ValueError:checks['same_q_cache_does_not_bypass_frozen_options_guard']=True
    else:checks['same_q_cache_does_not_bypass_frozen_options_guard']=False
    g.model.opt.ccd_iterations-=1
    np.savez_compressed(args.output/'cases.npz',**arrays)
    report={'status':'passed_local_unique_contact_branch_checks' if all(checks.values()) else 'failed_local_unique_contact_branch_checks','checks':checks,'case_rows':rows,'catalog_metadata':g.metadata,'pairs':g.pairs.tolist(),'pair_kind':g.pair_kind.tolist(),'source_files':[pin(p) for p in files],'cases':pin(args.output/'cases.npz'),'forbidden_call_counts':forbidden,'MuJoCo_version':mujoco.__version__,'dynamics_steps':0,'physical_reference_qualified':False,'known_observed_contact_labels_inferred':False,'scope':'Four prespecified CPU poses and all20 coordinate directions at three epsilons. Gates only unique, generated, normal-stable contact branches; ALLties/switches/hinge crossings and all-common errors are retained, not deleted to manufacture success. No global smoothness, optimization convergence, dynamics or qualified contact predictor claim.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda value:value.item() if isinstance(value,np.generic) else str(value))+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],'case_counts':[{k:r[k] for k in ['case','generated_pairs','active_pairs','ungenerated_pairs']} for r in rows],'max_unique_local_distance_error_m_per_rad':max(v for v in all_qualified_errors if v is not None)},indent=2));raise SystemExit(not all(checks.values()))


if __name__=='__main__':main()
