"""Exactly two optimistic point-force LPs on all three saved rank0 contacts."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
from scipy.optimize import linprog


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True)
    a=ap.parse_args();p=json.loads(a.protocol.read_text(encoding='utf-8'))
    if p['schema']!='saved_native_rank0_contact_wrench_upper_bound_protocol:1':raise ValueError('schema')
    for rec in [p['source'],*p['inputs'].values()]:
        if pin(rec['path'])!=rec:raise ValueError('changed pinned source/input')
    if p['source']!=pin(__file__) or p['seed']!=0 or p['rank']!=0:raise ValueError('fixed source/geometry')
    if p['contact_count']!=3 or p['mu']!=.7 or p['per_finger_cap_N']!=3 or p['LP_solves']!=2:raise ValueError('fixed LP scope')
    if p['force_scale_N']!=1.962 or p['moment_scale_Nm']!=.151074:raise ValueError('fixed scales')
    if p['gravity_world_m_s2']!=[0,0,-9.81]:raise ValueError('fixed gravity')
    out=Path(p['output']);out.mkdir(exist_ok=False)
    parent=json.loads(Path(p['inputs']['native_search_report']['path']).read_text(encoding='utf-8'))
    seed=next(s for s in parent['seed_results'] if s['seed']==0)
    candidate=next(c for c in seed['shortlist'] if c['rank']==0)
    n=candidate['native'];raw=np.array(n['raw']['contact_rows24'],dtype=np.float64)
    details=n['native_contact_details'];COM=np.array(n['raw']['phone_COM_world_m'],dtype=np.float64)
    if raw.shape!=(3,24) or len(details)!=3 or n['native_contact_count']!=3:raise ValueError('all three contacts required')
    mass=n['model_configuration']['phone_mass_kg']
    if mass!=.2 or n['unknown']!=[] or not parent['model_audit']['passed']:raise ValueError('fixed known native state')
    rays=[];moments=[];ray_meta=[];point_records=[];groups=[]
    for i,(row,d) in enumerate(zip(raw,details)):
        if d['contact_index']!=i or not d['object_boundary'] or row[3]!=3:raise ValueError('contact identity/condim3')
        names=[d['geom1_name'],d['geom2_name']]
        if names.count('phone24_collision_convex')!=1:raise ValueError('one object-boundary side')
        expected_sign=1. if names[1]=='phone24_collision_convex' else -1.
        sign=d['force_on_object_sign']
        if sign!=expected_sign or d['category'] not in ('thumb_pad','index_pad'):raise ValueError('native force sign/group')
        frame=row[9:18].reshape(3,3);position=row[6:9]
        if np.max(np.abs(frame@frame.T-np.eye(3)))>1e-10:raise ValueError('native orthonormal contact frame')
        fw=sign*(frame.T@row[18:21])
        if np.max(np.abs(fw-np.asarray(d['force_on_object_world_N'])))>1e-12:raise ValueError('saved native force-frame convention')
        point_records.append({'contact_index':i,'raw_row24':row.tolist(),'native_detail':d,
            'position_world_m':position.tolist(),'frame_rows_normal_t1_t2':frame.tolist()})
        for tangent in (1,2):
            for direction in (-1.,1.):
                f=sign*(frame[0]+direction*p['mu']*frame[tangent])
                rays.append(f);moments.append(np.cross(position-COM,f));groups.append(d['category'])
                ray_meta.append({'contact_index':i,'finger_group':d['category'],'tangent_axis':tangent,'sign':direction})
    force=np.array(rays).T;moment=np.array(moments).T
    W=np.vstack((force,moment));scale=np.array([p['force_scale_N']]*3+[p['moment_scale_Nm']]*3)
    target=np.r_[-mass*np.asarray(p['gravity_world_m_s2'],dtype=float),np.zeros(3)]
    A=W/scale[:,None];b=target/scale
    base_A=np.vstack((np.c_[A,-np.ones(6)],np.c_[-A,-np.ones(6)]));base_b=np.r_[b,-b]
    costs=np.r_[np.zeros(12),1.];results=[]
    for capped in (True,False):
        aub,bub=base_A.copy(),base_b.copy()
        if capped:
            caps=np.array([[float(g==name) for g in groups]+[0.] for name in ('thumb_pad','index_pad')])
            aub=np.vstack((aub,caps));bub=np.r_[bub,[3.,3.]]
        sol=linprog(costs,A_ub=aub,b_ub=bub,bounds=[(0,None)]*13,method='highs',options=p['linprog_options'])
        result={'per_finger_cap_N':3. if capped else None,'success':bool(sol.success),'status':int(sol.status),
            'message':sol.message,'objective_scaled_Linf':float(sol.fun) if sol.fun is not None else None,
            'coefficients12_N':None,'wrench_world6_N_Nm':None,'residual_world6_N_Nm':None,'finger_normal_load_N':None}
        if sol.x is not None:
            lam=np.asarray(sol.x[:12]);w=W@lam;res=w-target
            result.update({'coefficients12_N':lam.tolist(),'wrench_world6_N_Nm':w.tolist(),
                'residual_world6_N_Nm':res.tolist(),'scaled_residual6':(res/scale).tolist(),
                'recomputed_scaled_Linf':float(np.max(np.abs(res/scale))),
                'max_inequality_violation':float(max(0,np.max(aub@sol.x-bub))),
                'all_solution_values_finite':bool(np.isfinite(sol.x).all()),
                'finger_normal_load_N':{g:float(lam[np.array(groups)==g].sum()) for g in ('thumb_pad','index_pad')},
                'point_forces_world_N':(np.array(rays)*lam[:,None]).reshape(3,4,3).sum(axis=1).tolist()})
        results.append(result)
    if any(name in sys.modules for name in ('torch','mujoco','warp','mujoco_warp')):raise ValueError('no native framework')
    report={'schema':'saved_native_rank0_contact_wrench_upper_bound:1','protocol':pin(a.protocol),'source':pin(__file__),
        'inputs':p['inputs'],'seed':0,'rank':0,'phone_COM_world_m':COM.tolist(),'mass_kg':mass,
        'all_three_generated_contacts':point_records,'force_rays3x12':force.tolist(),'moment_rays3x12':moment.tolist(),
        'ray_order':ray_meta,'target_contact_wrench6_N_Nm':target.tolist(),'normalization_scale6':scale.tolist(),
        'results':results,'actual_LP_solves':2,'new_native_model_forward_step_NN_GPU_calls':0,
        'scope':'Optimistic fixed saved point-force geometry only; ignores joint torque/PD, soft contact law and stability. Not native equilibrium/holding or global infeasibility qualification.'}
    f=out/'report.json';f.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'report':pin(f),'LP_solves':2,'results':results}))


if __name__=='__main__':main()
