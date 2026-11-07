"""Saved-only convex CPU screening/confirmation; no simulator imports or calls.

Uses frozen independent v1 quaternion-vector/energy arithmetic, not producer
physics helpers. Scientific bounds are unchanged protocol6745/8981 values.
"""
import argparse
import collections
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
import pathlib
import traceback

import numpy as np

BASE = pathlib.Path(__file__).resolve().parents[1]
HELPER = BASE/'simulation_stage1/cpu_saved_assessment_v1/assess_cpu_saved_v1.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == 'ebcf5f4cbdc3b588bb729c0eea28d8f17b6fd8ac4e3e2897fe51cc7ecd929ce1'
spec = importlib.util.spec_from_file_location('independent_saved_math_v1', HELPER)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
PROTO = BASE/'protocols/phone_convex_contact_v2.json'
PROTO_SHA = '6745a6e58ce4507c56ca6fe053a4d16fc2e4e2195cfdae6c74ea1797e6bd5ace'
COMMON_SHA = '4460ccba501e46a033e304460796103afb7b2ef4ef822e6d8653f1a496228ae9'
RUNNER_SHA = '1b6889af2d5d8f1d4366532eab745ddd55aa9d0246aa1a410a5566d89e172965'
TESTS = ['drop','table_settling','force_push_release']
DTS = [.00125,.000625]


def inspect_case(c, run_dir):
    pin = A.check_pin(c['trajectory'])
    with np.load(pin['path'], allow_pickle=False) as z:
        d = {k:z[k] for k in z.files}
    dt,mass,mu,tau,test = [c[k] for k in ['dt_s','mass_kg','sliding_friction','timeconst_s','test']]
    n = round(3/dt)
    expected_shapes = {'qpos':(n+1,7),'qvel':(n+1,6),'qacc':(n+1,6),'com_world_m':(n+1,3),
       'com_velocity_world_m_s':(n+1,3),'angular_velocity_world_rad_s':(n+1,3),
       'physical_energy_U_T_J':(n+1,2),'native_energy_potential_kinetic_J':(n+1,2),
       'time_s':(n+1,),'tick':(n+1,),'tick_time_s':(n+1,),'penetration_m':(n+1,),
       'contact_count':(n+1,),'previous_interval_applied_force_world_x_N':(n+1,)}
    assert all(d[k].shape == s for k,s in expected_shapes.items())
    assert d['contact_long_table'].ndim==2 and d['contact_long_table'].shape[1]==24
    assert all(np.isfinite(v).all() for v in d.values())
    assert d['tick'].dtype.kind in 'iu' and np.array_equal(d['tick'],np.arange(n+1))
    assert A.close(d['time_s'],d['tick']*dt,1e-9) and np.array_equal(d['tick_time_s'],d['tick']*dt)
    before_path=run_dir/(c['case']+'_before_steps.json');before=A.read(before_path)
    opt=c['compiled_options'];assert opt==before['compiled_options'];A.check_pin(c['model_xml']);assert c['model_xml']==before['model_xml']
    assert opt['nu']==opt['neq']==opt['ntendon']==0 and opt['body_mass']==[0.,mass]
    o=opt['option'];assert o['tolerance']==1e-6 and o['ls_tolerance']==.01 and o['timestep']==dt
    assert o['integrator']==3 and o['solver']==2 and o['iterations']==100 and o['ls_iterations']==20
    assert o['cone']==0 and o['disableflags']==0 and o['density']==o['viscosity']==0
    assert o['gravity']==[0.,0.,-9.81] and opt['body_gravcomp']==[0.,0.]
    assert tau>=2*dt and max(tau,2*dt)==tau
    names=opt['geom_names'];plane=names.index('bench_support');hull=names.index('phone24_collision_convex')
    for k in ['geom_condim','geom_friction','geom_priority','geom_solmix','geom_solref','geom_solimp','geom_margin','geom_gap']:
        assert opt[k][plane]==opt[k][hull],k
    assert opt['geom_condim'][hull]==3 and opt['geom_friction'][hull]==[mu,.005,.0001]
    assert opt['geom_solref'][hull]==[tau,1.] and opt['geom_solimp'][hull]==[.9,.95,.001,.5,2.]
    assert opt['geom_margin'][hull]==opt['geom_gap'][hull]==0
    center=np.array(before['COM_body_m']);assert A.close(opt['body_ipos'][1],center,1e-14)
    full=np.array([.07292929805013726,.010315921319752897,.15395519224900234])
    I=np.array([mass*math.fsum(float(full[j])**2 for j in range(3) if j!=i)/12 for i in range(3)])
    assert A.close(I,opt['body_inertia'][1],1e-15) and A.close(I,before['inertia_kg_m2'],1e-15)
    q,v=d['qpos'],d['qvel'];qnorm=np.linalg.norm(q[:,3:],axis=1);assert np.all(qnorm>0)
    off=A.qvec(q[:,3:],center);com=q[:,:3]+off;omega=A.qvec(q[:,3:],v[:,3:]);vc=v[:,:3]+np.cross(omega,off)
    iq=np.array(opt['body_iquat'][1])*[1.,-1.,-1.,-1.];wi=A.qvec(iq,v[:,3:])
    U=mass*9.81*com[:,2];T=.5*mass*np.sum(vc*vc,axis=1)+.5*np.sum(I*wi*wi,axis=1);E=U+T
    errs={'COM_position_maxabs_m':float(np.max(abs(com-d['com_world_m']))),
          'COM_velocity_maxabs_m_s':float(np.max(abs(vc-d['com_velocity_world_m_s']))),
          'world_omega_maxabs_rad_s':float(np.max(abs(omega-d['angular_velocity_world_rad_s']))),
          'physical_UT_maxabs_J':float(np.max(abs(np.column_stack((U,T))-d['physical_energy_U_T_J'])))}
    assert A.close(com,d['com_world_m']) and A.close(vc,d['com_velocity_world_m_s']) and A.close(omega,d['angular_velocity_world_rad_s'])
    assert A.close(np.column_stack((U,T)),d['physical_energy_U_T_J'])
    verts=np.array(before['local_collision_vertices_m'],dtype=np.float64);assert verts.shape==(352,3)
    q0=np.array([math.sqrt(.5),math.sqrt(.5),0,0]);rel=A.qvec(q0,verts-center)
    support=float(rel[:,2].min());clearance=.1 if test=='drop' else .0005
    assert A.close(support,before['initial_minimum_collision_height_relative_to_COM_m'],1e-14)
    expected_com=np.array([0,0,clearance-support]);expected_root=expected_com-A.qvec(q0,center)
    assert A.close(q[0,:3],expected_root,1e-14) and A.close(q[0,3:],q0,1e-14)
    assert np.array_equal(v[0],np.zeros(6)) and A.close(q[0],before['initial_requested_qpos'],1e-14)
    bottom=float((q[0,:3]+A.qvec(q[0,3:],verts))[:,2].min());assert abs(bottom-clearance)<1e-12
    cl=d['contact_long_table'];assert np.array_equal(cl[:,[0,2,3,4]],np.trunc(cl[:,[0,2,3,4]]))
    ticks=cl[:,0].astype(np.int64);assert np.all((ticks>=0)&(ticks<=n))
    assert np.all(np.isin(cl[:,3:5],[plane,hull])) and np.all(cl[:,3]!=cl[:,4]) and A.close(cl[:,1],ticks*dt,1e-9)
    counts=np.bincount(ticks,minlength=n+1);assert np.array_equal(counts,d['contact_count'])
    offsets=np.r_[0,np.cumsum(counts)];assert np.array_equal(cl[:,2],np.arange(len(cl))-np.repeat(offsets[:-1],counts))
    minimum=np.full(n+1,np.inf);np.minimum.at(minimum,ticks,cl[:,5]);depth=np.maximum(0,-minimum)
    assert np.array_equal(depth,d['penetration_m'])
    force=A.force_schedule(test,n,dt);assert np.array_equal(force,d['previous_interval_applied_force_world_x_N'])
    impulse=float(math.fsum(force[1:])*dt);expected_J=.4 if test=='force_push_release' else 0.
    Jlim=16*np.finfo(force.dtype).eps*max(1.,abs(expected_J));work=float(math.fsum(force[1:]*np.diff(com[:,0])))
    rest=d['tick']>=round(2.5/dt);energy=A.passive_energy(E,dt,test)
    first=float(np.flatnonzero(counts>0)[0]*dt) if np.any(counts>0) else None
    metrics={'max_penetration_m':float(depth.max()),'rest_max_penetration_m':float(depth[rest].max()),
       'rest_COM_speed_RMS_m_s':A.rms_vec(vc[rest]),'rest_COM_speed_max_m_s':float(np.linalg.norm(vc[rest],axis=1).max()),
       'rest_angular_speed_RMS_rad_s':A.rms_vec(omega[rest]),'rest_angular_speed_max_rad_s':float(np.linalg.norm(omega[rest],axis=1).max()),
       'passive_energy_windows':energy,'force_impulse_Ns':impulse,'force_impulse_error_Ns':abs(impulse-expected_J),
       'force_impulse_roundoff_limit_Ns':Jlim,'known_external_work_J':work,'final_E_minus_initial_minus_work_J':float(E[-1]-E[0]-work),
       'first_contact_time_s':first,'contact_rows':len(cl),'minimum_kinetic_energy_J':float(T.min()),'qacc_maxabs':float(abs(d['qacc']).max()),
       'initial_actual_hull_bottom_world_z_m':bottom,'initial_requested_clearance_m':clearance,'saved_compiled_hull_support_relative_COM_m':support,
       'arithmetic_errors':errs}
    assert c['metrics']['completed_steps']==c['metrics']['expected_steps']==n
    assert A.close(c['metrics']['max_penetration_m'],metrics['max_penetration_m'])
    assert c['metrics']['extra_nonintegrating_forward_calls']==n+1
    warnings_ok=not any(c['metrics']['warning_terminal_counts']) and not c['metrics']['warning_history']
    gates={'G1_identity_and_coverage':'PASS','G2_finite_and_solver':A.status(warnings_ok and c['metrics']['error'] is None and T.min()>=-1e-12),
       'G3_penetration':A.status(metrics['max_penetration_m']<=.002 and metrics['rest_max_penetration_m']<=.0005),
       'G4_rest_motion':A.status(metrics['rest_COM_speed_RMS_m_s']<=.01 and metrics['rest_COM_speed_max_m_s']<=.02 and metrics['rest_angular_speed_RMS_rad_s']<=.1 and metrics['rest_angular_speed_max_rad_s']<=.2),
       'G5_energy':A.status(all(x['status']=='PASS' for x in energy)),
       'G6_contacts_and_input':A.status(np.any(counts[rest]>0) and abs(impulse-expected_J)<=Jlim),'G8_CPU_Warp_parity':'NOT_RUN'}
    row={k:c[k] for k in ['case','test','mass_kg','sliding_friction','timeconst_s','dt_s']}
    row.update(backend='CPU_MuJoCo',trajectory_pin=pin,before_steps_pin=A.pin(before_path),model_xml_pin=c['model_xml'],metrics=metrics,gates=gates,
               warning_history=c['metrics']['warning_history'],warning_terminal_counts=c['metrics']['warning_terminal_counts'])
    return row,{'q':q,'com':com,'v':vc,'w':omega,'first':first,'steps':n}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=pathlib.Path,required=True);parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    p=A.read(PROTO);assert A.pin(PROTO)['sha256']==PROTO_SHA
    fixture=A.focused_closed_form()
    rp=args.run_dir/'result.json';r=A.read(rp);mode=r['mode'];assert mode in ['screen','confirm']
    assert r['schema']==('phone_convex_contact_CPU_screening:2' if mode=='screen' else 'phone_convex_contact_CPU_confirmation:2')
    assert A.check_pin(r['protocol'])['sha256']==PROTO_SHA
    bp=A.check_pin(r['binding']);assert pathlib.Path(bp['path']).samefile(args.run_dir/'execution_binding.json');binding=A.read(bp['path'])
    for k in ['protocol','runner','common','legacy_helpers','asset','geometry_report','source_GLB','collision_OBJ','visual_OBJ']:
        A.check_pin(binding[k])
    assert binding['runner']['sha256']==RUNNER_SHA and binding['common']['sha256']==COMMON_SHA
    assert binding['asset']['sha256']==p['source_pins']['convex_asset']['sha256']
    assert binding['integration_steps_before_binding']==0
    if mode=='screen':
        params=[(.2,.6,tau,dt,t) for tau in [.02,.005,.0025] for dt in DTS for t in TESTS]
        selected=None;selection_pin=None
    else:
        selection_pin=A.check_pin(binding['selection']);s=A.read(selection_pin['path']);selected=s['selected_timeconst_s']
        assert s['schema']=='phone_convex_contact_selection:2' and s['screening_complete'] and s['selection_complete'] and s['audit_integrity_qualified']
        assert s['protocol_pin']['sha256']==PROTO_SHA and selected==max(s['eligible_timeconst_s'])
        pairs=[(.2,.6)]+[x for x in itertools.product([.1,.2,.3],[.3,.6,.9]) if x!=(.2,.6)]
        params=[(m,mu,selected,dt,t) for m,mu in pairs for dt in DTS for t in TESTS]
    assert binding['parameters']==[list(x) for x in params]
    assert len(r['cases'])==r['case_count']==len(params)
    rows=[];traces={}
    for c,key in zip(r['cases'],params):
        assert tuple(c[k] for k in ['mass_kg','sliding_friction','timeconst_s','dt_s','test'])==key
        expected_id=f'{key[4]}_mass{key[0]:g}_mu{key[1]:g}_tau{key[2]:g}_dt{key[3]:g}'
        assert c['case']==expected_id
        row,trace=inspect_case(c,args.run_dir);rows.append(row);traces[key]=trace
    expected_steps=64800 if mode=='screen' else 194400
    assert sum(x['steps'] for x in traces.values())==r['integration_steps']==expected_steps
    pair_rows=[];limits=p['engineering_gates']['G7_timestep_pair']
    for m,mu,tau,test in dict.fromkeys((x[0],x[1],x[2],x[4]) for x in params):
        a,b=traces[(m,mu,tau,DTS[0],test)],traces[(m,mu,tau,DTS[1],test)]
        diff={'final_COM_position_difference_m':float(np.linalg.norm(a['com'][-1]-b['com'][-1])),
              'final_orientation_difference_deg':float(np.degrees(A.quat_angle(a['q'][-1,3:],b['q'][-1,3:]))),
              'final_COM_velocity_vector_difference_m_s':float(np.linalg.norm(a['v'][-1]-b['v'][-1])),
              'final_angular_velocity_vector_difference_rad_s':float(np.linalg.norm(a['w'][-1]-b['w'][-1])),
              'shared_tick_COM_trajectory_RMS_difference_m':A.rms_vec(a['com']-b['com'][::2]),
              'first_contact_time_difference_s':abs(a['first']-b['first']) if a['first'] is not None and b['first'] is not None else None}
        checks={k:(v is not None and v<=limits[k]) for k,v in diff.items()};state=A.status(all(checks.values()))
        pair_id=f'{test}_mass{m:g}_mu{mu:g}_tau{tau:g}'
        pair_rows.append({'pair_id':pair_id,'mass_kg':m,'sliding_friction':mu,'timeconst_s':tau,'test':test,'metrics':diff,'checks':checks,'status':state})
        for row in rows:
            if (row['mass_kg'],row['sliding_friction'],row['timeconst_s'],row['test'])==(m,mu,tau,test):row['gates']['G7_timestep_pair']=state
    g=A.read(binding['geometry_report']['path']);geom_checks={
       'outside':g['maximum_original_vertex_outside_hull_m']<=.0005,
       'p95':g['full_sampled_hull_to_scan']['p95_absolute_m']<=.002,
       'max':g['full_sampled_hull_to_scan']['maximum_m']<=.004}
    geometry_ok=all(geom_checks.values());per_tau=[]
    for tau in ([.02,.005,.0025] if mode=='screen' else [selected]):
        cr=[x for x in rows if x['timeconst_s']==tau];pr=[x for x in pair_rows if x['timeconst_s']==tau]
        eligible=geometry_ok and all(all(v=='PASS' for k,v in x['gates'].items() if not k.startswith('G8')) for x in cr)
        per_tau.append({'timeconst_s':tau,'case_ids':[x['case'] for x in cr],
          'case_gate_results':{x['case']:x['gates'] for x in cr},'timestep_pair_ids':[x['pair_id'] for x in pr],
          'timestep_pair_status':{x['pair_id']:x['status'] for x in pr},'eligible':eligible,
          'rejection_reasons':[{'case':x['case'],'failed_gates':[k for k,v in x['gates'].items() if v=='FAIL']} for x in cr if any(v=='FAIL' for v in x['gates'].values())]})
    trajectory_pins={x['case']:x['trajectory_pin'] for x in rows};assert len(trajectory_pins)==len(rows)
    gate_counts={k:dict(collections.Counter(x['gates'][k] for x in rows)) for k in rows[0]['gates']}
    CPU_ok=all(all(v=='PASS' for k,v in x['gates'].items() if not k.startswith('G8')) for x in rows)
    report={'schema':('phone_convex_contact_CPU_screening_assessment:2' if mode=='screen' else 'phone_convex_contact_CPU_confirmation_assessment:2'),
      'status':'COMPLETED_SAVED_CPU_ASSESSMENT_NOT_REAL_PARAMETER_CALIBRATION','created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'auditor_pin':A.pin(__file__),'independent_math_helper_pin':A.pin(HELPER),'focused_fixture':fixture,
      'assessment_complete':True,'audit_integrity_qualified':True,'confirmation_complete':mode=='confirm',
      'producer_result_pin':A.pin(rp),'protocol_pin':A.pin(PROTO),'execution_binding_pin':bp,'selection_receipt_pin':selection_pin,
      'CPU_case_count':len(rows),'integration_steps':expected_steps,'selected_timeconst_s':selected,'trajectory_pins':trajectory_pins,
      'CPU_bench_qualified':CPU_ok,'geometry_qualified':geometry_ok,'geometry_checks':geom_checks,
      'case_results':rows,'timestep_comparisons':pair_rows,'gate_case_counts':gate_counts,'per_timeconst_gate_results':per_tau,
      'Warp_G8_status':'NOT_RUN','stage2_qualified':False,'real_parameter_calibrated':False,'ego_retargeting_qualified':False,'hand_task_qualified':False,
      'source_limits':['No native model/physics rerun. Compiled options/warnings/call counts are source-backed saved captures, not a new API interceptor.',
       'Physical U/T, COM velocity, contact depth/count and all gate arithmetic independently recomputed. Native force frame is saved but no world wrench or PRE-solver impulse reconstruction is attempted.',
       'models_info24 drives box inertia in the producer; its hash is inherited from the old immutable geometry report and checked current here, rather than invented as a new direct preintegration pin.']}
    mi=A.read(BASE/'asset_stage1/geometry_report.json')['source_pins']['models_info'];report['models_info_inherited_pin']=A.check_pin(mi)
    A.write(args.output/'all_case_results.json',rows);A.write(args.output/'all_timestep_pairs.json',pair_rows)
    if mode=='screen':
        eligible=[x['timeconst_s'] for x in per_tau if x['eligible']];chosen=max(eligible) if eligible else None
        selection={'schema':'phone_convex_contact_selection:2','protocol_pin':A.pin(PROTO),'screening_result_pin':A.pin(rp),
          'trajectory_pins':trajectory_pins,'audit_integrity_qualified':True,'screening_complete':True,'selection_complete':True,
          'eligible_timeconst_s':eligible,'selected_timeconst_s':chosen,'per_timeconst_gate_results':per_tau,
          'rule':'maximum eligible positive timeconstant; none means no confirmation; no fallback after downstream failure',
          'source_pin':A.pin(__file__),'geometry_qualified':geometry_ok,'real_parameter_calibrated':False}
        A.write(args.output/'selection.json',selection);report['selection_output_pin']=A.pin(args.output/'selection.json');report['selected_timeconst_s']=chosen
    A.write(args.output/'report.json',report)
    print(json.dumps({'report':A.pin(args.output/'report.json'),'selection':report.get('selection_output_pin'),
       'selected_timeconst_s':report['selected_timeconst_s'],'gate_case_counts':gate_counts,'per_tau':[{'tau':x['timeconst_s'],'eligible':x['eligible'],'reasons':x['rejection_reasons']} for x in per_tau]},indent=2,allow_nan=False))


if __name__=='__main__':
    try: main()
    except Exception:
        # A failed audit stays failed; no automatic data/threshold repair or retry.
        print(traceback.format_exc());raise
