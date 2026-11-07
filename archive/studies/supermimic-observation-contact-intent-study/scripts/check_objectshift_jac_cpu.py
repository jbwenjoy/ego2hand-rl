"""Local real-native-contact coupled Jacobian checks; never optimization."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import mujoco
import refine_initial_contacts_objectshift_cpu as coupled
from refine_initial_contacts_objectshift_cpu import (ROOT,ObjectShiftObjective,ball_translation,
    geometry_only,read,write,sha)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe',type=Path,default=ROOT/'configs/frame0-h23-objectshift-review-v1.json')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/objectshift_cpu_checks_v1')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();checks={};cases=[];raw={}
    recipe=read(args.recipe)
    optimizer_calls=[]
    def no_optimizer(*a,**kw):
        optimizer_calls.append(True);raise AssertionError('Checker must not optimize')
    coupled.least_squares=no_optimizer
    with geometry_only():
        o=ObjectShiftObjective(recipe);g=o.g
        initial_mocap_pos=g.data.mocap_pos.copy();initial_mocap_quat=g.data.mocap_quat.copy()
        with np.load(ROOT/'outputs/frame0_h23_distal_v2/initialization_candidate.npz',allow_pickle=False) as z:
            failed_q=z['joint_pos_rad'][0].copy()
        probes=[('baseline_zero',o.x0.copy()),('failed_fixed_pose_q_zero',np.r_[failed_q,np.zeros(3)]),
                ('baseline_local_shift',np.r_[o.qbase,[.05,-.03,.02]]),
                ('failed_fixed_pose_q_local_shift',np.r_[failed_q,[.05,-.03,.02]])]
        for i,z in enumerate((np.zeros(3),np.array([.3,-.2,.1]),np.array([10.,10.,10.]))):
            dt,D=ball_translation(z);fd=np.column_stack([(ball_translation(z+np.eye(3)[a]*1e-6)[0]-ball_translation(z-np.eye(3)[a]*1e-6)[0])/2e-6 for a in range(3)])
            checks[f'ballmap_{i}_local_FD']=bool(np.max(np.abs(D-fd))<1e-10)
            checks[f'ballmap_{i}_norm_strictly_under_20mm']=bool(np.linalg.norm(dt)<.020)
        for key,bad in [('qshape',np.zeros(22)),('nonfinite',np.r_[o.qbase,[np.nan,0,0]])]:
            try:o.evaluate(bad);rejected=False
            except ValueError:rejected=True
            checks['reject_'+key]=rejected
        for label,x in probes:
            r,J=o.evaluate(x);base=o.last_components;native=base['native'];dt=base['dt_m'];D=base['dt_dz']
            distances=native.signed_distance_m.copy(); generated=native.generated_mask.copy()
            analytic=np.c_[native.selected_distance_jacobian,base['distance_dt']@D]
            # Both epsilon levels and every pair survive in the saved arrays.
            reports=[]
            for eps in (1e-6,5e-7):
                fd=np.full_like(analytic,np.nan);fdR=np.zeros_like(J)
                stable=generated.copy();no_tie=(native.tie_count==1)
                for axis in range(23):
                    xp=x.copy();xm=x.copy();xp[axis]+=eps;xm[axis]-=eps
                    rp,_=o.evaluate(xp);np_=o.last_components['native'];dp=np_.signed_distance_m.copy()
                    rm,_=o.evaluate(xm);nm=o.last_components['native'];dm=nm.signed_distance_m.copy()
                    fd[:,axis]=(dp-dm)/(2*eps);fdR[:,axis]=(rp-rm)/(2*eps)
                    stable &= np_.generated_mask & nm.generated_mask
                    stable &= (np_.tie_count==1)&(nm.tie_count==1)
                    # Retain ambiguous/changed contact feature normals separately.
                    stable &= np.sum(np_.selected_normal_a_to_b_w*native.selected_normal_a_to_b_w,axis=1)>.99999
                    stable &= np.sum(nm.selected_normal_a_to_b_w*native.selected_normal_a_to_b_w,axis=1)>.99999
                stable &= no_tie
                native_error=np.max(np.abs(fd[stable]-analytic[stable])) if stable.any() else None
                hand_object=stable&(g.pair_kind=='hand_object')
                object_error=np.max(np.abs(fd[hand_object,20:]-analytic[hand_object,20:])) if hand_object.any() else None
                # Check non-collision residuals independently; collision feature
                # switching/ties remain retained, not falsely called smooth.
                other=np.r_[np.arange(45),np.arange(45+len(g.pairs),len(r))]
                other_error=float(np.max(np.abs(fdR[other]-J[other])))
                active_stable=stable&native.active_mask
                active_residual_error=float(np.max(np.abs(fdR[45+np.flatnonzero(active_stable)]-J[45+np.flatnonzero(active_stable)]))) if active_stable.any() else None
                prefix=f'{label}__eps{eps:g}'
                for name,value in [('distance_fd',fd),('distance_analytic',analytic),('residual_fd',fdR),('residual_analytic',J),('stable_unique_pair_mask',stable)]:raw[prefix+'__'+name]=value
                checks[f'{prefix}_noncollision_residual_FD']=other_error<1e-4
                checks[f'{prefix}_stable_distance_FD']=bool(native_error is not None and native_error<1e-3)
                checks[f'{prefix}_active_overlap_residual_FD']=bool(active_residual_error is not None and active_residual_error<1.)
                reports.append({'epsilon_x':eps,'stable_unique_generated_pairs':int(stable.sum()),'stable_unique_hand_object_pairs':int(hand_object.sum()),
                                'distance_FD_max_abs_m_per_x':float(native_error) if native_error is not None else None,
                                'object_z_distance_FD_max_abs_m_per_z':float(object_error) if object_error is not None else None,
                                'noncollision_residual_FD_max_abs_per_x':other_error,'active_stable_overlap_residual_FD_max_abs_per_x':active_residual_error,
                                'stable_unique_pair_rows':np.flatnonzero(stable).tolist()})
            # Exact native-order sign conversion is checked independently from
            # the catalogue-order implementation, on actual generated contacts.
            o.evaluate(x);native=o.last_components['native'];order_errors=[];orders=[]
            for row in np.flatnonzero(native.generated_mask&(g.pair_kind=='hand_object')):
                c=g.data.contact[native.selected_contact_index[row]];a,b=map(int,c.geom)
                force_on_object=c.frame[:3]*(1. if b in g.object_ids else -1.)
                order_errors.append(float(np.max(np.abs(force_on_object-o.last_components['distance_dt'][row]))))
                orders.append({'row':int(row),'native_geom_order':[a,b],'object_is_native_geom1':bool(b in g.object_ids)})
            checks[label+'_native_order_force_on_object_conversion']=bool(order_errors and max(order_errors)<1e-12)
            checks[label+'_self_translation_zero']=bool(np.all(o.last_components['distance_dt'][native.generated_mask&(g.pair_kind=='hand_self')]==0))
            checks[label+'_fixed_rotation_wrist_and_zero_clock_velocity']=bool(np.array_equal(o.last_components['pose'][3:],o.pose0[3:]) and g.data.time==0 and not np.any(g.data.qvel) and np.array_equal(g.data.mocap_pos,initial_mocap_pos) and np.array_equal(g.data.mocap_quat,initial_mocap_quat))
            checks[label+'_absent_native_gap_remains_unknown']=bool(np.isnan(native.signed_distance_m[~native.generated_mask]).all())
            raw[label+'__x']=x;raw[label+'__distances']=native.signed_distance_m;raw[label+'__generated_mask']=native.generated_mask;raw[label+'__ties']=native.tie_count
            cases.append({'name':label,'object_delta_w_m':dt.tolist(),'generated_pairs':int(native.generated_mask.sum()),'hand_object_pairs':int(np.count_nonzero(native.generated_mask&(g.pair_kind=='hand_object'))),
                          'active_pairs':int(native.active_mask.sum()),'tied_pair_rows':np.flatnonzero(native.generated_mask&(native.tie_count>1)).tolist(),
                          'missing_pair_rows_unknown_gap':np.flatnonzero(~native.generated_mask).tolist(),'native_order_checks':orders,'epsilon_checks':reports})
        checks['no_optimizer_invoked']=not optimizer_calls;checks['no_Torch_Warp_import']=not any(k.split('.')[0] in ('torch','warp','mujoco_warp') for k in sys.modules)
        checks['frozen_recipe_optimization_not_authorized']=recipe['optimization_authorized'] is False
        raw['pair_indices']=g.pairs;raw['pair_kind']=g.pair_kind
    np.savez_compressed(args.output/'local_jacobian_arrays.npz',**raw)
    result={'status':'CPU_LOCAL_BRANCH_CHECKS_PASS' if all(checks.values()) else 'CPU_LOCAL_BRANCH_CHECKS_FAIL',
            'checks':checks,'cases':cases,'recipe_sha256':sha(args.recipe),'module_sha256':sha(ROOT/'scripts/refine_initial_contacts_objectshift_cpu.py'),'checker_sha256':sha(__file__),
            'arrays_sha256':sha(args.output/'local_jacobian_arrays.npz'),'optimization_runs':0,'physics_steps':0,'GPU_used':False,'confidence':None,'physical_reference_qualified':False,
            'wall_seconds':time.monotonic()-started,'scope':'Local finite differences on all reported generated/native contact cases. Absent contacts stay NaN; ties/feature changes retained as nonsmooth. Stable selected branches are not global optimization, reachability, force or dynamic qualification.'}
    write(args.output/'report.json',result)
    print(json.dumps({'status':result['status'],'passed':sum(checks.values()),'total':len(checks),'cases':[{'name':c['name'],'generated':c['generated_pairs'],'hand_object':c['hand_object_pairs'],'ties':len(c['tied_pair_rows']),'errors':c['epsilon_checks']} for c in cases],'wall_seconds':result['wall_seconds']},indent=2))
    raise SystemExit(not all(checks.values()))


if __name__=='__main__':main()
