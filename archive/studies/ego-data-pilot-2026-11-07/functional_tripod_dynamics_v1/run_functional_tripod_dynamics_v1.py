"""UNFROZEN SOURCE: three designated-pad constant-PD native holding diagnostic.

No execution without a reviewed CPU-authorized fresh protocol. Native contact,
force, energy and actuator values are solver PRE; qpos/qvel/COM are POST. Normal
preload is an engineering command, not a force certificate or static entry gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent
P=ROOT.parent
SCHEMA='functional_tripod_native_dynamic_protocol:1'
RESULT_SCHEMA='functional_tripod_native_dynamic_result:1'
CANDIDATE_SCHEMA='functional_tripod_geometric_reference_candidate:1'
CANDIDATE_REPORT_SCHEMA='functional_tripod_geometric_reference_result:1'
DESIGNATED=[15,28,41]
DTS=[.0003125,.00015625]
DURATION=2.
CONTROL_ROUTES=['no_preload_ctrl_equal_q','normal_1N_each_pad','normal_3N_each_pad']
OLD_GATES={
    'duration_s':2.,'final_COM_drift_m':.005,'final_angle_rad':.08726646259971647,
    'tail_duration_s':.5,'tail_linear_velocity_RMS_m_s':.01,
    'tail_angular_velocity_RMS_rad_s':.2,'max_native_penetration_m':.002,
    'tail_each_pad_normal_force_N':.1,'tail_dual_pad_force_coverage':.95,
    'max_phone_nonpad_normal_force_N':.1,'max_self_penetration_m':.0002,
    'max_joint_limit_excursion_rad':.001,'max_table_normal_force_N':1e-8,
    'warning_count':0,'abort_COM_drift_m':.05,'abort_angle_rad':.7853981633974483}
GATES={**{k:v for k,v in OLD_GATES.items() if k!='tail_dual_pad_force_coverage'},
       'tail_triple_pad_force_coverage':.95}
PAIR_GATES={'final_COM_difference_m':.002,'final_orientation_difference_rad':.03490658503988659,
            'require_both_duration_complete':True}

def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
    if pin(r['path'])!=r:raise ValueError('Changed input: '+r['path'])
    return Path(r['path'])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,j):
    with Path(p).open('x',encoding='utf8') as f:json.dump(j,f,indent=2,allow_nan=False)
def finite_vector(v,n):
    return isinstance(v,list) and len(v)==n and all(type(x) in (int,float) and math.isfinite(x) for x in v)
def model_digest_except_dt(m,np):
    """Full exposed array/options before/after; only timestep may differ."""
    h=hashlib.sha256()
    for scope,obj in [('model',m),('option',m.opt)]:
        for name in sorted(n for n in dir(obj) if not n.startswith('_')):
            if scope=='option' and name=='timestep':continue
            value=getattr(obj,name)
            if isinstance(value,np.ndarray):
                a=np.ascontiguousarray(value);h.update((scope+':'+name+':'+a.dtype.str+':'+str(a.shape)).encode());h.update(a.tobytes())
            elif scope=='option':
                try:scalar=float(value)
                except (TypeError,ValueError):continue
                h.update((scope+':'+name+':'+repr(scalar)).encode())
    return h.hexdigest()
def metadata_preflight(protocol_path):
    c=read(protocol_path)
    if (c['schema']!=SCHEMA or c['dt_s']!=DTS or c['duration_s']!=DURATION or c['gates']!=GATES
        or c['old_two_pad_gates']!=OLD_GATES or c['dt_pair_gates']!=PAIR_GATES
        or c['control_routes']!=CONTROL_ROUTES or c['designated_geom_ids']!=DESIGNATED
        or type(c['CPU_run_authorized']) is not bool or c['training_or_GPU_authorized'] is not False):
        raise ValueError('Frozen diagnostic fields/explicit CPU authorization boundary')
    if set(c['inputs'])!={'model_mjb','model_binding','candidate_report'}:raise ValueError('Exact three inputs')
    for q in [*c['inputs'].values(),*c['source_pins'].values()]:checked(q)
    if c['source_pins']['runner']!=pin(__file__):raise ValueError('Own source binding')
    out=Path(c['output'])
    if not out.is_absolute() or out.exists() or out.parent.resolve()!=ROOT.resolve():raise ValueError('Fresh own output')
    binding=read(c['inputs']['model_binding']['path'])
    if binding['model']!=c['inputs']['model_mjb']:raise ValueError('Original compiled model binding')
    report=read(c['inputs']['candidate_report']['path'])
    if report['schema']!=CANDIDATE_REPORT_SCHEMA or report['model_mjb']!=c['inputs']['model_mjb']:raise ValueError('Tripod geometry family')
    entries=report['candidates']
    if not isinstance(entries,list) or len(entries)>4:raise ValueError('At most four predeclared candidates')
    selected=[];screen=[];ids=[]
    for entry in entries:
        candidate=read(checked(entry['candidate']));checked(entry['state_npz'])
        cid=candidate['candidate_id']
        if type(cid) is not str or not cid or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in cid):raise ValueError('Safe unique candidate ID')
        ids.append(cid);g=candidate['geometry'];reasons=[]
        if candidate['schema']!=CANDIDATE_SCHEMA or candidate['designated_geom_ids']!=DESIGNATED:raise ValueError('Candidate identity')
        if not finite_vector(candidate['qpos27_F64'],27) or not finite_vector(candidate['qvel26_F64'],26):raise ValueError('Finite state')
        if any(candidate['qvel26_F64']):raise ValueError('Initial zero velocity only, no zero-qacc gate')
        if (g.get('passed') is not True or g.get('phone_pair_count')!=22 or g.get('self_pair_count')!=189
            or g.get('table_complete') is not True or g.get('unknown')!=[]):reasons.append('Incomplete/failed tripod full geometry')
        pads=candidate['pad_bindings']
        if not isinstance(pads,list) or [v['hand_geom_id'] for v in pads]!=DESIGNATED:raise ValueError('Exact three pad bindings')
        for p in pads:
            n=p['force_on_object_unit_normal_world']
            if (type(p['hand_body_id']) is not int or not finite_vector(p['hand_anchor_world_m'],3)
                or not finite_vector(p['geom_local_hand_anchor_m'],3) or not finite_vector(p['geom_local_force_on_object_normal'],3)
                or not finite_vector(n,3) or abs(sum(x*x for x in n)-1)>1e-10
                or abs(sum(x*x for x in p['geom_local_force_on_object_normal'])-1)>1e-10):raise ValueError('Finite physical hand point/unit force normal')
        screen.append(dict(candidate_id=cid,eligible_geometry=not reasons,reasons=reasons,candidate=entry['candidate']))
        if not reasons:selected.append(dict(candidate=candidate,**entry))
    if len(set(ids))!=len(ids):raise ValueError('Unique candidate IDs')
    if c['budget']!=dict(max_candidates=4,control_routes=3,dt_cases=2,maximum_cases=24,maximum_integration_steps=230400):raise ValueError('Fixed finite budget')
    return c,out,binding,selected,screen

def assert_model(m,mujoco,np,binding):
    if mujoco.__version__!='3.11.0' or (m.nq,m.nv,m.nu,m.na,m.neq,m.nmocap,m.nflex,m.nplugin,m.ntendon)!=(27,26,20,0,0,0,0,0,0):raise ValueError('Original rigid no-aid native model')
    for name in ['control','passive','contactfilter','sensor','act_dyn','act_gain','act_bias']:
        if getattr(mujoco,'get_mjcb_'+name)() is not None:raise ValueError('Unexpected callback '+name)
    if np.any(m.body_gravcomp):raise ValueError('No gravity compensation')
    if not np.array_equal(m.opt.gravity,np.asarray([0.,0.,-9.81])):raise ValueError('Original gravity')
    if (m.opt.tolerance,m.opt.ls_tolerance,m.opt.iterations,m.opt.ls_iterations)!=(1e-6,.01,100,20):raise ValueError('Frozen solver')
    if (int(m.opt.integrator),int(m.opt.solver),int(m.opt.cone))!=(int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST),int(mujoco.mjtSolver.mjSOL_NEWTON),int(mujoco.mjtCone.mjCONE_PYRAMIDAL)):raise ValueError('Frozen algorithm')
    if not np.array_equal(m.actuator_trnid[:,0],np.arange(20)) or not np.array_equal(m.actuator_gear,np.c_[np.ones(20),np.zeros((20,5))]):raise ValueError('Direct unit gear')
    if not np.all(m.actuator_ctrllimited) or not np.all(m.actuator_forcelimited) or not np.all(m.jnt_actfrclimited[:20]):raise ValueError('All actual PD/actuator/generalized bounds required')
    if not np.all(m.actuator_gaintype==int(mujoco.mjtGain.mjGAIN_FIXED)) or not np.all(m.actuator_biastype==int(mujoco.mjtBias.mjBIAS_AFFINE)):raise ValueError('Fixed-gain affine-bias PD only')
    if len(binding['explicit_pairs'])!=22 or binding['mu_phone_hand']!=.7 or binding['mu_phone_table']!=.6:raise ValueError('Original contacts/materials')
    phone=m.body('hot3d_phone24').id;pg=m.geom('phone24_collision_convex').id;table=m.geom('stage1_single_table').id
    pads=[m.geom(f'right_finger{i}_tip_sensor_col').id for i in (1,2,3)]
    if pads!=DESIGNATED or m.body_mass[phone]!=.2 or (m.jnt_qposadr[m.body_jntadr[phone]],m.jnt_dofadr[m.body_jntadr[phone]])!=(20,20):raise ValueError('Exact designated IDs/object layout')
    return phone,pg,table,pads

def contact_snapshot(m,d,phone_geom,table,pads,mujoco,np,counts,tick):
    pad=np.zeros(3);other=table_force=penetration=self_pen=0.;rows=[]
    for ci in range(d.ncon):
        con=d.contact[ci]
        if not np.isfinite(np.r_[con.dist,con.pos,con.frame]).all():raise ValueError('Nonfinite native contact geometry')
        g1,g2=map(int,con.geom);force=np.empty(6);mujoco.mj_contactForce(m,d,ci,force);counts['contact_force_reads']+=1
        if not np.isfinite(force).all():raise ValueError('Nonfinite native local wrench')
        penetration=max(penetration,-float(con.dist))
        if phone_geom in (g1,g2):
            other_geom=g2 if g1==phone_geom else g1
            if other_geom in pads:pad[pads.index(other_geom)]+=max(0.,float(force[0]))
            elif other_geom==table:table_force+=max(0.,float(force[0]))
            else:other+=max(0.,float(force[0]))
        elif table not in (g1,g2):self_pen=max(self_pen,-float(con.dist))
        rows.append(np.r_[tick,g1,g2,int(con.dim),int(con.efc_address),con.dist,con.pos,con.frame,force])
    return pad,other,table_force,penetration,self_pen,rows

def prepare_controls(m,item,mujoco,np,counts):
    """One fresh initialization forward/3 physical-point Jacobians, zero steps.

    Hand reaction is -F_object, hence tau_act=bias-passive+J_hand.T F_object.
    Desired normal commands are not measured contact loads; no tangential load
    or static-qacc criterion is inserted. All routes remain predeclared.
    """
    c=item['candidate'];q=np.asarray(c['qpos27_F64'],dtype=np.float64)
    with np.load(checked(item['state_npz']),allow_pickle=False) as z:
        if z['qpos27_F64'].dtype!=np.float64 or z['qvel26_F64'].dtype!=np.float64 or z['qpos27_F64'].tobytes()!=q.tobytes() or z['qvel26_F64'].tobytes()!=np.asarray(c['qvel26_F64'],dtype=np.float64).tobytes():raise ValueError('Raw candidate state binding')
    if np.any(q[:20]<m.jnt_range[:20,0]) or np.any(q[:20]>m.jnt_range[:20,1]) or abs(float(q[23:]@q[23:])-1)>1e-12:raise ValueError('Hardware joints/native quaternion')
    d=mujoco.MjData(m);counts['MjData']+=1;d.qpos[:]=q;d.qvel[:]=0;d.ctrl[:]=q[:20];d.qacc_warmstart[:]=0
    mujoco.mj_forward(m,d);counts['preload_preparation_forward']+=1
    if np.any(d.warning.number) or np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise ValueError('Preparation warning/aid')
    J=[];anchor_checks=[]
    for pad in c['pad_bindings']:
        if int(m.geom_bodyid[pad['hand_geom_id']])!=pad['hand_body_id']:raise ValueError('Hand anchor actual body')
        gid=pad['hand_geom_id'];GR=d.geom_xmat[gid].reshape(3,3)
        point=GR@np.asarray(pad['geom_local_hand_anchor_m'])+d.geom_xpos[gid]
        normal=GR@np.asarray(pad['geom_local_force_on_object_normal'])
        pe=float(np.linalg.norm(point-np.asarray(pad['hand_anchor_world_m'])));ne=float(np.linalg.norm(normal-np.asarray(pad['force_on_object_unit_normal_world'])))
        if pe>1e-10 or ne>1e-10:raise ValueError('Same-q hand anchor/normal FK mismatch')
        anchor_checks.append(dict(geom_id=gid,world_point=point.tolist(),world_force_normal=normal.tolist(),point_error_m=pe,normal_error=ne))
        jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,point,pad['hand_body_id']);counts['preload_point_jacobians']+=1;J.append(jp[:,:20])
    J=np.asarray(J);normals=np.asarray([p['force_on_object_unit_normal_world'] for p in c['pad_bindings']])
    gain=m.actuator_gainprm[:,0].copy();offset=m.actuator_biasprm[:,0]+m.actuator_biasprm[:,1]*d.actuator_length+m.actuator_biasprm[:,2]*d.actuator_velocity
    if not np.isfinite(gain).all() or np.any(gain<=0):raise ValueError('Positive actual native PD gain')
    base=d.qfrc_bias[:20]-d.qfrc_passive[:20];routes=[]
    for name,load in zip(CONTROL_ROUTES,[0.,1.,3.]):
        f=load*normals;tau=gain*q[:20]+offset if load==0 else base+np.einsum('pik,pi->k',J,f)
        ctrl=q[:20].copy() if load==0 else (tau-offset)/gain
        cm=np.c_[ctrl-m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]-ctrl]
        am=np.c_[tau-m.actuator_forcerange[:,0],m.actuator_forcerange[:,1]-tau]
        jm=np.c_[tau-m.jnt_actfrcrange[:20,0],m.jnt_actfrcrange[:20,1]-tau]
        available=bool(np.isfinite(np.r_[ctrl,tau]).all() and np.all(cm>=0) and np.all(am>=0) and np.all(jm>=0))
        routes.append(dict(name=c['candidate_id']+'__'+name,candidate_id=c['candidate_id'],route=name,available=available,
            unavailable_reason=None if available else 'INVERSE_PD_OUTSIDE_REAL_CTRL_ACTUATOR_OR_JOINT_BOUNDS_NO_CLAMP',
            qpos27=q.tolist(),ctrl20=ctrl.tolist(),requested_normal_N_each=load,
            intended_force_on_object_world_N3x3=f.tolist(),desired_generalized_actuator_tau20=tau.tolist(),
            ctrl_margin20x2=cm.tolist(),actuator_force_margin20x2=am.tolist(),joint_actuator_force_margin20x2=jm.tolist(),
            state_npz=item['state_npz'],candidate=item['candidate']))
    prep_ctrl_effective=np.clip(q[:20],m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    raw=dict(qpos27=q.tolist(),J_hand_anchor3x3x20=J.tolist(),normals_force_on_object_world=normals.tolist(),anchor_FK_checks=anchor_checks,
        preparation_raw_ctrl20=q[:20].tolist(),preparation_effective_limited_ctrl20=prep_ctrl_effective.tolist(),
        preparation_control_scope='Zero-step preparation only; ctrl=q may be natively limited. This is explicit and is not a qualified dynamic route or silent repair of any requested route.',
        qfrc_bias26=d.qfrc_bias.tolist(),qfrc_passive26=d.qfrc_passive.tolist(),qacc26_descriptor=d.qacc.tolist(),
        gain20=gain.tolist(),offset20=offset.tolist(),actuator_length20=d.actuator_length.tolist(),actuator_velocity20=d.actuator_velocity.tolist())
    return routes,raw

def evaluate_case(m,variant,dt,case_dir,phone,pg,table,pads,mujoco,np,counts):
    """Original 2s native loop and old gates; three-pad classification additive."""
    q0=np.asarray(variant['qpos27'],dtype=float);ctrl=np.asarray(variant['ctrl20'],dtype=float);m.opt.timestep=dt
    d=mujoco.MjData(m);counts['MjData']+=1;d.qpos[:]=q0;d.qvel[:]=0;d.ctrl[:]=ctrl;d.qacc_warmstart[:]=0
    mujoco.mj_forward(m,d);counts['initial_forward']+=1
    if np.any(d.xfrc_applied) or np.any(d.qfrc_applied) or np.any(d.warning.number):raise ValueError('Initial aid/warning')
    R=np.empty(9)
    def COM(q):
        mujoco.mju_quat2Mat(R,q[23:27]);return q[20:23]+R.reshape(3,3)@m.body_ipos[phone]
    COM0=COM(q0).copy();initial=contact_snapshot(m,d,pg,table,pads,mujoco,np,counts,-1)
    initial_record=dict(phase='RESET_FORWARD',pad_normal_N3=initial[0].tolist(),phone_nonpad_normal_N=initial[1],
        old_two_pad_phone_nonpad_normal_N=initial[1]+float(initial[0][2]),table_normal_N=initial[2],
        contact_rows24=np.asarray(initial[5],dtype=float).reshape(-1,24).tolist(),qacc26_descriptor=d.qacc.tolist(),
        actuator_force20=d.actuator_force.tolist(),qfrc_actuator26=d.qfrc_actuator.tolist())
    steps=round(DURATION/dt);sample_stride=round(.01/dt)
    keys=['post_time','pre_time_from_integer_tick','post_COM','post_angle','post_qpos','post_qvel',
        'pre_native_energy','pre_ncon','pre_pad_normal_N','pre_phone_nonpad_normal_N','pre_table_normal_N',
        'pre_penetration_m','pre_self_penetration_m','pre_qfrc_actuator','pre_actuator_force','pre_solver_niter',
        'pre_actuator_length20','pre_actuator_velocity20','pre_actuator_force_margin20x2','pre_joint_actuator_force_margin20x2']
    rows={k:[] for k in keys};contact_rows=[];geometry_samples=[];terminate='COMPLETE';attempted_steps=0
    for tick in range(steps):
        mujoco.mj_step(m,d);counts['integration_steps']+=1;attempted_steps+=1
        values=[d.time,d.qpos,d.qvel,d.qacc,d.energy,d.qfrc_actuator,d.actuator_force]
        if not all(np.isfinite(x).all() for x in values):terminate='ABORT_NONFINITE';break
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied) or not np.array_equal(d.ctrl,ctrl):raise ValueError('Object aid/controller mutation')
        pad,other,table_force,penetration,self_pen,cr=contact_snapshot(m,d,pg,table,pads,mujoco,np,counts,tick);contact_rows.extend(cr)
        com=COM(d.qpos).copy();angle=float(2*np.arccos(np.clip(abs(d.qpos[23:27]@q0[23:27]),0,1)))
        am=np.c_[d.actuator_force-m.actuator_forcerange[:,0],m.actuator_forcerange[:,1]-d.actuator_force]
        jm=np.c_[d.qfrc_actuator[:20]-m.jnt_actfrcrange[:20,0],m.jnt_actfrcrange[:20,1]-d.qfrc_actuator[:20]]
        for k,v in zip(keys,[float(d.time),tick*dt,com,angle,d.qpos.copy(),d.qvel.copy(),d.energy.copy(),int(d.ncon),
            pad,other,table_force,penetration,self_pen,d.qfrc_actuator.copy(),d.actuator_force.copy(),np.asarray(d.solver_niter).copy(),
            d.actuator_length.copy(),d.actuator_velocity.copy(),am,jm]):rows[k].append(v)
        if tick%sample_stride==0:geometry_samples.append(dict(pre_tick=tick,geom_xpos=d.geom_xpos.copy(),geom_xmat=d.geom_xmat.copy(),pre_qpos=q0.copy() if tick==0 else rows['post_qpos'][-2].copy()))
        if np.any(d.warning.number):terminate='ABORT_NATIVE_WARNING';break
        if np.linalg.norm(com-COM0)>GATES['abort_COM_drift_m'] or angle>GATES['abort_angle_rad']:terminate='ABORT_LOSS_OF_HOLD';break
    r={k:np.asarray(v) for k,v in rows.items()};completed=len(r['post_time'])
    if not completed:raise ValueError('No finite completed record')
    tail_start_tick=round((DURATION-GATES['tail_duration_s'])/dt);tail=np.arange(1,completed+1)>=tail_start_tick;pre_tail=np.arange(completed)>=tail_start_tick
    qlimits=m.jnt_range[:20];qvio=np.maximum(qlimits[:,0]-r['post_qpos'][:,:20],0)+np.maximum(r['post_qpos'][:,:20]-qlimits[:,1],0)
    lin=r['post_qvel'][tail,20:23];omega=r['post_qvel'][tail,23:26]
    metrics=dict(final_COM_drift_m=float(np.linalg.norm(r['post_COM'][-1]-COM0)),final_angle_rad=float(r['post_angle'][-1]),
        tail_linear_velocity_RMS_m_s=float(np.sqrt(np.mean(np.sum(lin*lin,axis=1)))) if len(lin) else None,
        tail_angular_velocity_RMS_rad_s=float(np.sqrt(np.mean(np.sum(omega*omega,axis=1)))) if len(omega) else None,
        tail_triple_pad_force_coverage=float(np.mean(np.all(r['pre_pad_normal_N'][pre_tail]>.1,axis=1))) if np.any(pre_tail) else None,
        max_native_penetration_m=float(r['pre_penetration_m'].max()),max_self_penetration_m=float(r['pre_self_penetration_m'].max()),
        max_phone_nonpad_normal_force_N=float(r['pre_phone_nonpad_normal_N'].max()),max_table_normal_force_N=float(r['pre_table_normal_N'].max()),
        max_joint_limit_excursion_rad=float(qvio.max()),warning_count=int(np.sum(d.warning.number)))
    gates=dict(duration_complete=terminate=='COMPLETE' and completed==steps)
    for k,v in metrics.items():gates[k]=v is not None and (v>=GATES[k] if k=='tail_triple_pad_force_coverage' else v<=GATES[k])
    oldmetrics={k:v for k,v in metrics.items() if k!='tail_triple_pad_force_coverage'}
    oldmetrics['tail_dual_pad_force_coverage']=float(np.mean(np.all(r['pre_pad_normal_N'][pre_tail,:2]>.1,axis=1))) if np.any(pre_tail) else None
    oldnonpad=r['pre_phone_nonpad_normal_N']+r['pre_pad_normal_N'][:,2]
    oldmetrics['max_phone_nonpad_normal_force_N']=float(oldnonpad.max());oldgates=dict(duration_complete=gates['duration_complete'])
    for k,v in oldmetrics.items():oldgates[k]=v is not None and (v>=OLD_GATES[k] if k=='tail_dual_pad_force_coverage' else v<=OLD_GATES[k])
    np.savez_compressed(case_dir/'trace.npz',**r,ctrl20=ctrl,ctrl_margin20x2=np.asarray(variant['ctrl_margin20x2']),initial_qpos27=q0,initial_COM_world3=COM0,
        pre_old_two_pad_phone_nonpad_normal_N=oldnonpad,contact_rows=np.asarray(contact_rows,dtype=float).reshape(-1,24),
        sample_pre_tick=np.asarray([v['pre_tick'] for v in geometry_samples]),sample_pre_qpos=np.asarray([v['pre_qpos'] for v in geometry_samples]),
        sample_pre_geom_xpos=np.asarray([v['geom_xpos'] for v in geometry_samples]),sample_pre_geom_xmat=np.asarray([v['geom_xmat'] for v in geometry_samples]))
    summary=dict(name=variant['name'],candidate_id=variant['candidate_id'],route=variant['route'],dt_s=dt,termination=terminate,
        integration_steps=attempted_steps,finite_steps_saved=completed,actual_final_time_s=float(d.time),metrics=metrics,gates=gates,
        nominal_holding_dynamics_pass=all(gates.values()),old_two_pad_metrics=oldmetrics,old_two_pad_gates=oldgates,old_two_pad_pass=all(oldgates.values()),
        trace=pin(case_dir/'trace.npz'),warnings=np.asarray(d.warning.number).tolist(),initial_native=initial_record,
        source_candidate=variant['candidate'],source_state_npz=variant['state_npz'],command=variant,
        energy_force_geometry_phase='PRE(tick*dt); qpos/qvel/COM POST((tick+1)*dt)',
        velocity_metric_semantics='Preserved root translational freejoint qvel RMS; angular body qvel norm is frame invariant, not a new COM-velocity gate.',
        contact_row_columns=['pre_tick','geom1','geom2','dim','efc_address','distance_m','position3','frame9','local_wrench6'],
        nonpad_metric_definition='Maximum across PRE ticks of the SUM of positive Fn over every phone contact with non-designated hand geoms; same aggregation as original two-pad gate. old_two_pad also includes all positive Fn on geom41.',
        no_external_forces_equality_mocap_or_pose_reset=True,static_zero_acceleration_entry_required=False)
    write(case_dir/'summary.json',summary)
    endpoint=dict(dt_s=dt,final_COM_world_m=r['post_COM'][-1].tolist(),final_quaternion_wxyz=r['post_qpos'][-1,23:27].tolist(),
        actual_final_time_s=float(d.time),termination=terminate,finite_steps_saved=completed,original_case_pass=summary['nominal_holding_dynamics_pass'])
    return summary,endpoint

def dt_pair_summary(variants,cases,endpoints,np):
    result=[]
    for v in variants:
        both=[c for c in cases if c['name']==v['name']];e=sorted(endpoints[v['name']],key=lambda x:x['dt_s'],reverse=True)
        if len(both)!=2 or sorted(c['dt_s'] for c in both)!=sorted(DTS):raise ValueError('Every available route retains both dt')
        quat=[np.asarray(x['final_quaternion_wxyz']) for x in e];unit=[q/np.linalg.norm(q) for q in quat]
        angle=float(2*np.arccos(np.clip(abs(unit[0]@unit[1]),0,1)));distance=float(np.linalg.norm(np.asarray(e[0]['final_COM_world_m'])-e[1]['final_COM_world_m']))
        complete=all(c['gates']['duration_complete'] for c in both)
        gates=dict(both_duration_complete=complete,final_COM_difference_m=distance<=.002,final_orientation_difference_rad=angle<=PAIR_GATES['final_orientation_difference_rad'])
        result.append(dict(name=v['name'],comparison_status='TWO_COMPLETE_2S_ENDPOINTS' if complete else 'INCOMPLETE_DURATION_NOT_2S_CONSISTENCY',
            final_COM_difference_m=distance,final_orientation_difference_rad=angle,endpoints=e,gates=gates,dt_pair_consistency_pass=all(gates.values()),
            both_tripod_cases_pass=all(c['nominal_holding_dynamics_pass'] for c in both),both_old_two_pad_cases_pass=all(c['old_two_pad_pass'] for c in both),
            orientation_observer='F64 endpoint normalization for pair comparison only; no state repair'))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    c,out,binding,selected,screen=metadata_preflight(args.protocol)
    if not args.run:
        print(json.dumps(dict(status='METADATA_VALID_NO_BACKEND_MODEL_OR_STEPS',screening=screen,geometry_candidates=len(selected),maximum_cases=6*len(selected),maximum_steps=57600*len(selected))));return
    if c['CPU_run_authorized'] is not True:raise ValueError('Root reviewed explicit CPU authorization required')
    out.mkdir();write(out/'protocol.json',c);shutil.copyfile(__file__,out/'execution_source.py');start=time.perf_counter()
    counts=dict(MJB_reads=0,MjData=0,preload_preparation_forward=0,preload_point_jacobians=0,initial_forward=0,integration_steps=0,contact_force_reads=0,neural_training_GPU=0)
    phase='metadata';cases=[];routes=[]
    try:
        if selected:
            import mujoco
            import numpy as np
            m=mujoco.MjModel.from_binary_path(str(checked(c['inputs']['model_mjb'])));counts['MJB_reads']+=1
            phone,pg,table,pads=assert_model(m,mujoco,np,binding)
            model_before=model_digest_except_dt(m,np)
            parameter_arrays={k:np.asarray(getattr(m,k)).copy() for k in ['actuator_gainprm','actuator_biasprm','actuator_gear','actuator_ctrlrange','actuator_forcerange','actuator_ctrllimited','actuator_forcelimited','actuator_trnid','jnt_range','jnt_actfrcrange','jnt_actfrclimited','dof_armature','dof_frictionloss','body_mass','body_inertia','body_ipos','body_iquat','body_gravcomp','geom_friction','geom_solref','geom_solimp','geom_margin','geom_gap','geom_contype','geom_conaffinity','pair_dim','pair_geom1','pair_geom2','pair_friction','pair_solref','pair_solreffriction','pair_solimp','pair_margin','pair_gap']}
            parameter_arrays['option_gravity']=np.asarray(m.opt.gravity).copy()
            option_before={n:float(getattr(m.opt,n)) for n in ['tolerance','ls_tolerance','iterations','ls_iterations','integrator','solver','cone','disableflags','enableflags','impratio','noslip_iterations','noslip_tolerance']}
            write(out/'frozen_options_except_timestep.json',option_before)
            np.savez_compressed(out/'actual_frozen_model_parameters.npz',**parameter_arrays)
            preparation=[]
            for item in selected:
                phase='prepare_'+item['candidate']['candidate_id'];rv,raw=prepare_controls(m,item,mujoco,np,counts);routes.extend(rv);preparation.append(dict(candidate=item['candidate'],raw=raw,routes=rv))
            write(out/'control_preparation.json',preparation)
            available=[v for v in routes if v['available']];endpoints={v['name']:[] for v in available}
            for variant in available:
                for dt in DTS:
                    phase=variant['name']+'__dt'+str(round(dt*1e9))+'ns';d=out/phase;d.mkdir()
                    summary,endpoint=evaluate_case(m,variant,dt,d,phone,pg,table,pads,mujoco,np,counts)
                    cases.append(summary);endpoints[variant['name']].append(endpoint)
                    print(json.dumps(dict(case=phase,termination=summary['termination'],steps=summary['finite_steps_saved'],tripod_pass=summary['nominal_holding_dynamics_pass'],old_two_pad_pass=summary['old_two_pad_pass'])),flush=True)
            pairs=dt_pair_summary(available,cases,endpoints,np)
            model_after=model_digest_except_dt(m,np)
            if model_before!=model_after:raise ValueError('Full model arrays or options except timestep changed')
            model_audit=dict(before=model_before,after=model_after,all_exposed_arrays_and_numeric_options_unchanged_except_dt=True,allowed_timestep_values=DTS)
        else:pairs=[];model_audit=dict(model_loaded=False)
        if counts['integration_steps']>230400:raise ValueError('Actual integration budget exceeded')
        report=dict(schema=RESULT_SCHEMA,status='COMPLETED_TRIPOD_CONSTANT_PD_DYNAMIC_DIAGNOSTIC' if cases else 'NO_AVAILABLE_TRIPOD_CONTROL_NO_INTEGRATION',
            protocol=pin(args.protocol),counts=counts,screening=screen,control_routes=routes,cases=cases,dt_pair_consistency=pairs,model_audit=model_audit,elapsed_seconds=time.perf_counter()-start,
            all_tripod_cases_and_dt_pairs_pass=bool(cases and all(v['nominal_holding_dynamics_pass'] for v in cases) and all(v['dt_pair_consistency_pass'] for v in pairs)),
            all_old_two_pad_cases_and_dt_pairs_pass=bool(cases and all(v['old_two_pad_pass'] for v in cases) and all(v['dt_pair_consistency_pass'] for v in pairs)),
            mass_or_friction_calibrated=False,ego_or_learned_policy_benefit_qualified=False,
            scope='Functional three-designated rigid collision patches, not verified tactile/soft-pad anatomy. q8..11 genuinely available within original hardware limits. All three predeclared controls retained or explicitly unreachable. Ideal1N/3N normal commands are not actual measured loads or hard dynamic Fn caps. No static all-qacc-zero admission gate, helper forces, object aid, retuning or result-based selection.')
        write(out/'report.json',report)
    except BaseException as e:
        write(out/'failure.json',dict(phase=phase,error=repr(e),counts=counts,cases_completed=len(cases),automatic_retry=False));raise

if __name__=='__main__':main()
