"""Bounded deterministic geometry candidates; no dynamics or training.

All objective weights/budgets are in the prewritten protocol. Never reflects
reference/phone, changes the fixed palm, scales assets, or transfers22 angles.
"""
from pathlib import Path
import csv, hashlib, importlib.util, json, os, time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['MPLBACKEND']='Agg'
ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent


def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def write(p,obj):
    with Path(p).open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)


class BudgetStop(RuntimeError):pass


def main():
    import numpy as np
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    require=lambda v,s: None if v else (_ for _ in ()).throw(ValueError(s))
    require(not (ROOT/'report.json').exists(),'Fresh solve only')
    protocol=json.loads((ROOT/'retarget_protocol_v1.json').read_text(encoding='utf8'))
    for p in protocol['source_inputs'].values():require(pin(p['path'])==p,'Changed protocol input')
    targets=json.loads(Path(protocol['source_inputs']['targets']['path']).read_text(encoding='utf8'))
    loader_path=Path(protocol['source_inputs']['hand_loader']['path'])
    module=importlib.util.spec_from_file_location('fixed_hand2_geometry_loader',loader_path)
    loader=importlib.util.module_from_spec(module);module.loader.exec_module(loader)
    model=loader.load_model();data=mujoco.MjData(model);initial=loader.initial_state()
    q0=initial['qpos'][:20].copy();qlo=model.jnt_range[:20,0].copy();qhi=model.jnt_range[:20,1].copy()
    require(model.nq==27 and model.nu==20 and targets['frame']==42,'Fixed model/target dimensions')
    pid=model.body('right_palm_link').id;oid=model.body('hot3d_phone24').id
    phone_gid=model.geom('phone24_collision_convex').id;table_gid=model.geom('stage1_single_table').id
    hand_gids=[g for g in range(model.ngeom) if (model.geom_contype[g] or model.geom_conaffinity[g]) and model.geom_bodyid[g]!=oid and g!=table_gid]
    tip_sids=[model.site(f'right_finger{f}_tip').id for f in range(1,6)]
    tip_gids=[model.geom(f'right_finger{f}_tip_sensor_col').id for f in range(1,6)]
    require(all(g in hand_gids for g in tip_gids),'Source distal collision geometries')
    t_object=np.array([m['reference_tip_object_m'] for m in targets['mapping']])
    contact_ids=[i for i,m in enumerate(targets['mapping']) if m['uniform_near_contact_candidate']]
    contact_points=np.array([targets['mapping'][i]['nearest_scan_surface_point_object_m'] for i in contact_ids])
    palmT=initial['T_world_from_palm'];Rp=palmT[:3,:3].copy();pp=palmT[:3,3].copy()
    original_model_digest=hashlib.sha256(b''.join(np.ascontiguousarray(getattr(model,n)).tobytes() for n in ('jnt_range','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange','dof_armature','dof_damping','dof_frictionloss','body_pos','body_quat'))).hexdigest()
    counters={'mj_kinematics':0,'mj_geomDistance':0};started=time.perf_counter();trace=[];seeds=[];results=[]
    total_limit=protocol['budgets']['global_static_API_calls_max'];seconds=protocol['budgets']['total_optimization_wall_seconds_max']
    seed_static_start=0;seed_cap=total_limit;seed_index=-1;last=None

    def count(kind):
        if sum(counters.values())>=total_limit or time.perf_counter()-started>=seconds:raise BudgetStop('Global static/time cap')
        counters[kind]+=1

    def set_state(q,translation,rotation):
        data.qpos[:]=initial['qpos'];data.qpos[:20]=q
        worldR=Rp@Rotation.from_rotvec(rotation).as_matrix();worldp=pp+Rp@translation
        quat=Rotation.from_matrix(worldR).as_quat();data.qpos[20:23]=worldp;data.qpos[23:27]=quat[[3,0,1,2]]
        data.qvel[:]=0;data.ctrl[:]=0;data.time=0
        count('mj_kinematics');mujoco.mj_kinematics(model,data)
        require(np.array_equal(data.xpos[pid],pp) and np.array_equal(data.xmat[pid].reshape(3,3),Rp),'Fixed source palm changed')
        return worldR,worldp

    def kabsch(q):
        set_state(q,[0,0,0],[0,0,0])
        y=(data.site_xpos[tip_sids]-pp)@Rp;x=t_object
        xc=x-x.mean(0);yc=y-y.mean(0);u,s,vh=np.linalg.svd(xc.T@yc)
        rawR=vh.T@u.T;correction=np.eye(3);correction[-1,-1]=np.linalg.det(rawR)
        R=vh.T@correction@u.T;t=y.mean(0)-R@x.mean(0)
        return t,Rotation.from_matrix(R).as_rotvec(),{'raw_unconstrained_determinant':float(np.linalg.det(rawR)),'proper_determinant':float(np.linalg.det(R)),'singular_values':s.tolist(),'proper_fit_tip_RMS_m':float(np.sqrt(np.mean(np.sum((x@R.T+t-y)**2,axis=1))))}

    initialT=initial['T_palm_from_phone_model'];lower=np.r_[qlo,protocol['variables']['phone_translation_palm_m_bounds'][0],[-np.pi]*3];upper=np.r_[qhi,protocol['variables']['phone_translation_palm_m_bounds'][1],[np.pi]*3]
    seeds.append({'name':protocol['seeds'][0],'x':np.r_[q0,initialT[:3,3],[0,0,0]],'Kabsch':None})
    for q,name in ((q0,protocol['seeds'][1]),((qlo+qhi)/2,protocol['seeds'][2])):
        t,r,descriptor=kabsch(q);rawseed=np.r_[q,t,r];bounded=np.clip(rawseed,lower,upper)
        seeds.append({'name':name,'x':bounded,'Kabsch':descriptor,'seed_was_clipped_to_declared_search_box':bool(not np.array_equal(rawseed,bounded))})

    def evaluate(x):
        nonlocal last
        needed=1+len(hand_gids)+1
        if sum(counters.values())+needed>total_limit or sum(counters.values())-seed_static_start+needed>seed_cap:raise BudgetStop('Reserved per-seed static-call cap')
        R,p=set_state(x[:20],x[20:23],x[23:26]);pred=data.site_xpos[tip_sids].copy();desired=t_object@R.T+p
        distances=[];witness={}
        for gid in hand_gids+[table_gid]:
            count('mj_geomDistance');segment=np.zeros(6,np.float64)
            d=float(mujoco.mj_geomDistance(model,data,phone_gid,gid,1.,segment))
            require(np.isfinite(d) and np.isfinite(segment).all(),'Nonfinite static pair distance')
            distances.append(d);witness[gid]=segment.copy()
        distances=np.array(distances);tip_error=pred-desired
        blocks={'all5_tip_xyz':(tip_error/.01).reshape(-1)}
        if contact_ids:
            surface=np.stack([witness[tip_gids[i]][3:] for i in contact_ids]);desired_surface=contact_points@R.T+p
            blocks['candidate_surface_witness_xyz']=(.5*(surface-desired_surface)/.005).reshape(-1)
            blocks['candidate_signed_gap']=np.array([distances[hand_gids.index(tip_gids[i])] for i in contact_ids])/.002
        else:
            blocks['candidate_surface_witness_xyz']=np.zeros(0);blocks['candidate_signed_gap']=np.zeros(0)
        blocks['all_phone_hand_and_table_penetration']=2*np.maximum(-distances,0)/.001
        blocks['joint_regularizer']=.05*(x[:20]-q0)/(qhi-qlo)
        residual=np.concatenate(list(blocks.values()));require(np.isfinite(residual).all(),'Finite objective')
        total=float(residual@residual);maxpen=float(np.maximum(-distances,0).max())
        row={'seed_index':seed_index,'evaluation':sum(t['seed_index']==seed_index for t in trace)+1,'x26':x.tolist(),'objective':total,'block_squared_norms':{k:float(v@v) for k,v in blocks.items()},'tip_error_norm_m':np.linalg.norm(tip_error,axis=1).tolist(),'minimum_phone_hand_distance_m':float(distances[:-1].min()),'phone_table_distance_m':float(distances[-1]),'maximum_penetration_m':maxpen,'strict_nonpenetration':bool((distances>=0).all()),'static_call_counts':dict(counters),'elapsed_seconds':time.perf_counter()-started}
        trace.append(row)
        last={'row':row,'qpos27':data.qpos.copy(),'qvel26':data.qvel.copy(),'ctrl20':data.ctrl.copy(),'xpos':data.xpos.copy(),'xquat':data.xquat.copy(),'geom_xpos':data.geom_xpos.copy(),'geom_xmat':data.geom_xmat.copy(),'site_xpos':data.site_xpos.copy(),'tip_error_world_m':tip_error,'pair_distances_m':distances,'phone_R_world':R,'phone_p_world':p,'witness':witness}
        return residual

    for seed_index,seed in enumerate(seeds):
        last=None;seed_static_start=sum(counters.values());seed_cap=protocol['budgets']['per_seed_static_calls_reserved_max']
        status='NOT_STARTED';message='';solver_info=None
        try:
            fit=least_squares(evaluate,seed['x'],bounds=(lower,upper),method='trf',jac='2-point',max_nfev=80,ftol=1e-8,xtol=1e-8,gtol=1e-8,loss='linear')
            status='SOLVER_RETURNED';message=str(fit.message);solver_info={'success':bool(fit.success),'status':int(fit.status),'nfev':int(fit.nfev),'njev':int(fit.njev) if fit.njev is not None else None,'cost':float(fit.cost),'x_returned':fit.x.tolist()}
            # Preserve the actual returned point, not the last Jacobian perturbation.
            if not np.array_equal(np.array(last['row']['x26']),fit.x):evaluate(fit.x)
        except BudgetStop as e:status='BOUNDED_STOP_PRESERVED';message=str(e)
        except Exception as e:status='FAILED_PRESERVED';message=f'{type(e).__name__}: {e}'
        seed_meta={'seed_index':seed_index,'seed_name':seed['name'],'seed_x26':seed['x'].tolist(),'Kabsch':seed['Kabsch'],'status':status,'message':message,'solver':solver_info,'static_calls_this_seed':sum(counters.values())-seed_static_start}
        if last is not None:
            snapshot={k:v for k,v in last.items() if isinstance(v,np.ndarray)}
            np.savez_compressed(ROOT/f'seed{seed_index}_candidate_state.npz',**snapshot)
            noncandidate_gids=[g for g in hand_gids if any(f'right_finger{f+1}_' in model.geom(g).name for f in range(5) if f not in contact_ids)]
            palm_gids=[g for g in hand_gids if model.geom_bodyid[g]==pid]
            seed_meta.update({'last_complete':last['row'],'pair_distance_table':[{'geom_id':gid,'geom_name':model.geom(gid).name,'phone_signed_distance_m':float(d)} for gid,d in zip(hand_gids+[table_gid],last['pair_distances_m'])],
                'worst_phone_palm_distance_m':min((float(last['pair_distances_m'][hand_gids.index(g)]) for g in palm_gids),default=None),
                'worst_phone_noncandidate_finger_distance_m':min((float(last['pair_distances_m'][hand_gids.index(g)]) for g in noncandidate_gids),default=None),
                'phone_palm_SE3':{'translation_m':last['row']['x26'][20:23],'rotation_vector_rad':last['row']['x26'][23:26]},'q20_rad':last['row']['x26'][:20],
                'qpos27':last['qpos27'].tolist(),'all_hard_joint_bounds_satisfied':bool(((last['qpos27'][:20]>=qlo)&(last['qpos27'][:20]<=qhi)).all()),
                'fixed_palm_exact':True,'snapshot':pin(ROOT/f'seed{seed_index}_candidate_state.npz')})
        results.append(seed_meta)
    candidates=[r for r in results if 'last_complete' in r]
    order=sorted(candidates,key=lambda r:(not r['last_complete']['strict_nonpenetration'],r['last_complete']['maximum_penetration_m'],r['last_complete']['objective'],r['seed_index']))
    selected=order[0]['seed_index'] if order else None
    with (ROOT/'all_geometry_evaluations.jsonl').open('x',encoding='utf8') as f:
        for row in trace:f.write(json.dumps(row,allow_nan=False)+'\n')
    write(ROOT/'all3_seed_results.json',results)
    after=hashlib.sha256(b''.join(np.ascontiguousarray(getattr(model,n)).tobytes() for n in ('jnt_range','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange','dof_armature','dof_damping','dof_frictionloss','body_pos','body_quat'))).hexdigest()
    require(after==original_model_digest,'Robot parameter/basis arrays unchanged')
    plot_candidates(model,results,targets,np,Rotation)
    report={'schema':'fixed_hand2_frame42_geometry_candidate_report:1','status':'COMPLETED_BOUNDED_GEOMETRY_CANDIDATES_NOT_GRASP_OR_DYNAMICS','protocol':pin(ROOT/'retarget_protocol_v1.json'),'source':pin(__file__),'target_reference':pin(ROOT/'frame42_targets.json'),
        'all_seed_results':results,'selected_seed_by_predeclared_order':selected,'ranking_seed_indices':[r['seed_index'] for r in order],'selection_is_not_grasp_success':True,
        'all_source_robot_parameters_unchanged':True,'model_parameter_digest_before':original_model_digest,'model_parameter_digest_after':after,
        'fixed_wrist_source_frame':palmT.tolist(),'contact_candidate_anatomical_fingers':[targets['mapping'][i]['finger'] for i in contact_ids],
        'static_API_counts':counters,'total_static_API_calls':sum(counters.values()),'complete_objective_evaluations':len(trace),'optimization_elapsed_seconds':time.perf_counter()-started,
        'counts':{'MJB_loads':1,'MjData_instances':1,'mj_step_dynamics_training_GPU':0},'physical_holding_ego_method_or_contact_force_qualified':False,
        'limitations':['Left anatomical points map to right anatomical digits without reflection; mismatch remains in residual.','Only frame42 was fitted. Three deterministic seeds are all reported; no effect-based frame or weight changes.','Surface witnesses/native geomDistance use existing approximate hand colliders and phone single convex hull, while reference target is original scan.','Candidate skin point is not measured contact, and static signed distance is not a force or stability test.','Hand-self collisions are not exhaustively constrained; all phone-hand and phone-table pairs are checked.','No wrist movement, asset scaling, human-angle transfer, dynamics integration, or training occurred.']}
    report['outputs']={p.name:pin(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='report.json'}
    write(ROOT/'report.json',report);print(json.dumps({'report':pin(ROOT/'report.json'),'selected_seed':selected,'static_calls':sum(counters.values()),'states':len(trace),'seed_statuses':[r['status'] for r in results]}))


def plot_candidates(model,results,targets,np,Rotation):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(15,8),layout='constrained')
    for col,result in enumerate(results):
        ax=fig.add_subplot(2,3,col+1,projection='3d');ax2=fig.add_subplot(2,3,col+4)
        if 'snapshot' not in result:
            ax.text2D(.05,.5,result['status'],transform=ax.transAxes);ax2.axis('off');continue
        with np.load(result['snapshot']['path'],allow_pickle=False) as a:
            gp=a['geom_xpos'];gr=a['geom_xmat'].reshape(-1,3,3);tips=a['site_xpos'][[model.site(f'right_finger{f}_tip').id for f in range(1,6)]];R=a['phone_R_world'];p=a['phone_p_world']
        clouds=[]
        for gid in range(model.ngeom):
            if int(model.geom_type[gid])!=7:continue
            name=model.geom(gid).name
            if 'mesh_disabled' in name or name=='phone24_collision_convex':continue
            mid=int(model.geom_dataid[gid]);start=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);verts=model.mesh_vert[start:start+n].astype(np.float64)@gr[gid].T+gp[gid]
            fa=int(model.mesh_faceadr[mid]);nf=int(model.mesh_facenum[mid]);faces=model.mesh_face[fa:fa+nf]
            color='#888888' if name=='phone24_visual_only' else '#36a1bd';alpha=.3 if name=='phone24_visual_only' else .25
            ax.add_collection3d(Poly3DCollection(verts[faces]*1000,facecolor=color,edgecolor='none',alpha=alpha));clouds.append(verts)
            ax2.scatter(verts[::3,1]*1000,verts[::3,2]*1000,c=color,s=1,alpha=.3)
        target=np.array([m['reference_tip_object_m'] for m in targets['mapping']])@R.T+p
        ax.scatter(*tips.T*1000,c='#08732d',s=15);ax.scatter(*target.T*1000,c='#c83333',marker='x',s=20)
        for a,b in zip(tips,target):ax.plot(*np.stack([a,b]).T*1000,c='#c83333',lw=1)
        cloud=np.concatenate(clouds);lo=cloud.min(0)*1000;hi=cloud.max(0)*1000;ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]));ax.set_box_aspect(hi-lo);ax.view_init(elev=22,azim=-65);ax.set_axis_off()
        ax.set_title(f"Seed{col}: {result['status']}\nobjective {result['last_complete']['objective']:.3g}, max penetration {result['last_complete']['maximum_penetration_m']*1000:.3f}mm")
        ax2.scatter(tips[:,1]*1000,tips[:,2]*1000,c='#08732d',s=20,label='Robot tips');ax2.scatter(target[:,1]*1000,target[:,2]*1000,c='#c83333',s=25,marker='x',label='Same-digit left reference')
        ax2.set_aspect('equal');ax2.set(xlabel='World Y (mm)',ylabel='World Z (mm)');ax2.legend(fontsize=8);ax2.grid(alpha=.15)
    fig.suptitle('Fixed-wrist right Hand2 / unmirrored frame42 left reference — geometry only, no holding claim')
    fig.savefig(ROOT/'all3_geometry_candidates.png',dpi=150,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
