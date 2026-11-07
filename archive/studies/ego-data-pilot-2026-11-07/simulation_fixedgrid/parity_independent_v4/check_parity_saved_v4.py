"""Additional independent G8 saved arithmetic. No simulator/model imports.

All54 numeric identities, raw freejoint state -> quaternion-vector COM/v/omega,
native generated-contact ticks, actual captured solver scalars and CPU casts.
"""
import collections
import datetime
import hashlib
import importlib.util
import itertools
import json
import pathlib

import numpy as np

BASE=pathlib.Path(__file__).resolve().parents[2]
OUT=pathlib.Path(__file__).resolve().parent
PURE=BASE/'simulation_stage1/cpu_saved_assessment_v1/assess_cpu_saved_v1.py'
assert hashlib.sha256(PURE.read_bytes()).hexdigest()=='ebcf5f4cbdc3b588bb729c0eea28d8f17b6fd8ac4e3e2897fe51cc7ecd929ce1'
spec=importlib.util.spec_from_file_location('independent_rigid_math',PURE)
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
PROTO=BASE/'protocols/phone_convex_fixedgrid_v4.json'
CPU=BASE/'simulation_fixedgrid/cpu_v4/result.json'
CPU_REVIEW=BASE/'simulation_fixedgrid/cpu_assessment_v4/report.json'
WARP=BASE/'simulation_fixedgrid/warp_v4/result.json'
TERMINAL=BASE/'simulation_fixedgrid/warp_v4_process_terminal.json'
SHAS={'protocol':'7a84cdda18952e24e45cfd96ba0bdcb846ff656794512112911e64602b1ed14f',
      'CPU_review':'58cfa0cff76ec5f18799a3edf019c9026848df6a1d2945a99507836967fc48be',
      'Warp_result':'2c2a437cde4337b68f00379dd946ee4d3ee7e7b94ac3cfddd6e329e19ce9b8f5'}
TESTS=['drop','table_settling','force_push_release']


def key(c):
    return (c['mass_kg'],c['sliding_friction'],c['timeconst_s'],c['dt_s'],c['test'])


def f32bits(x):
    return np.asarray(x,dtype=np.float32).ravel().view(np.uint8)


def same_pin(a,b):
    return a['sha256']==b['sha256'] and a['bytes']==b['bytes'] and pathlib.Path(a['path']).samefile(b['path'])


def group_checks(g,CPU_cases):
    ident=g['group'];m,mu,dt,tau=[ident[k] for k in ['mass_kg','mu','dt_s','timeconst_s']]
    assert ident['worlds']==TESTS and tau==.0025 and dt in [.0003125,.00015625]
    opt=g['CPU_compiled_model_options'];runtime=g['preintegration_runtime_contract']
    for t in TESTS:assert opt==CPU_cases[(m,mu,tau,dt,t)]['compiled_options']
    source_opt=opt['option'];assert source_opt['tolerance']==1e-6 and source_opt['ls_tolerance']==.01
    assert source_opt['timestep']==dt and source_opt['disableflags']==0
    for name,value in runtime['discrete'].items():
        if name in source_opt:assert value==source_opt[name]
    scalar_rows={}
    for name,r in runtime['float32'].items():
        assert r['dtype']=='float32'
        actual=np.asarray(r['value'],dtype=np.float32)
        expected=np.asarray(r['CPU_value'],dtype=np.float32)
        assert np.array_equal(actual.ravel().view(np.uint8),expected.ravel().view(np.uint8)),name
        if name in source_opt:assert r['CPU_value']==source_opt[name]
        scalar_rows[name]={'actual_saved_value':r['value'],'CPU_saved_value':r['CPU_value'],
                          'FP32_cast_bytes_equal':True,'independent_CPU_field_available':name in source_opt}
    actual=g['GPU_effective_options']
    for field,value in [('timestep_F32',dt),('tolerance_F32',1e-6),('ls_tolerance_F32',.01)]:
        assert np.array_equal(f32bits(actual[field]),f32bits(value))
        assert actual[field]==float(np.float32(value))
    for field in ['solver','integrator','cone','iterations','ls_iterations','enableflags','disableflags']:
        assert actual[field]==source_opt[field]
    assert runtime['refsafe_enabled'] is True
    authored=np.asarray(runtime['authored_contact_solref_F32'],dtype=np.float32)
    assert np.array_equal(authored,np.tile(np.array([tau,1.],dtype=np.float32),(2,1)))
    effective=np.maximum(authored[:,0],np.float32(2)*np.float32(dt))
    assert np.array_equal(f32bits(runtime['effective_refsafety_timeconst_F32']),f32bits(effective))
    assert all(v['equal_CPU_cast'] is True for v in runtime['array_guards'].values())
    n=round(3/dt);assert ident['steps_per_world']==g['batch_steps_returned']==g['final_cursor']==n
    assert g['world_steps_returned']==3*n and g['additional_forward_calls']==1
    return {'mass_kg':m,'sliding_friction':mu,'dt_s':dt,'timeconst_s':tau,
       'CPU_compiled_options_all_three_cases_equal':True,'discrete_solver_fields_match':True,
       'actual_saved_GPU_FP32_solver_scalars_match_CPU_cast':True,'scalar_comparisons':scalar_rows,
       'model_array_guard_count':len(runtime['array_guards']),
       'model_array_equality_independently_rebuilt_from_raw_device_values':False,
       'model_array_equality_scope':'Only pinned producer runtime assertion; raw device model arrays not serialized.',
       'native_CPU_style_GPU_warning_table':runtime['native_CPU_style_warning_table']}


def rigid(d,center,omega_key):
    q=np.asarray(d['qpos'],dtype=np.float64);v=np.asarray(d['qvel'],dtype=np.float64)
    assert np.isfinite(q).all() and np.isfinite(v).all() and np.all(np.linalg.norm(q[:,3:],axis=1)>0)
    off=A.qvec(q[:,3:],center);com=q[:,:3]+off;omega=A.qvec(q[:,3:],v[:,3:]);vc=v[:,:3]+np.cross(omega,off)
    assert A.close(com,d['com_world_m']) and A.close(vc,d['com_velocity_world_m_s']) and A.close(omega,d[omega_key])
    return com,vc,omega


def main():
    if (OUT/'report.json').exists():raise FileExistsError('Fresh output required')
    assert A.pin(PROTO)['sha256']==SHAS['protocol'] and A.pin(CPU_REVIEW)['sha256']==SHAS['CPU_review'] and A.pin(WARP)['sha256']==SHAS['Warp_result']
    p,c,cr,w,terminal=map(A.read,[PROTO,CPU,CPU_REVIEW,WARP,TERMINAL])
    assert cr['CPU_bench_qualified'] and cr['assessment_complete'] and cr['audit_integrity_qualified']
    assert cr['schema']=='phone_convex_fixedgrid_CPU_assessment:4'
    assert c['schema']=='phone_convex_fixedgrid_CPU:4' and w['schema']=='phone_convex_fixedgrid_Warp:4'
    for r in [c,w]:assert A.check_pin(r['protocol'])['sha256']==SHAS['protocol']
    assert same_pin(w['CPU_confirmation_join']['saved_assessment'],A.pin(CPU_REVIEW))
    assert same_pin(w['CPU_confirmation_join']['producer_result'],A.pin(CPU))
    assert w['CPU_confirmation_join']['trajectory_pins']==cr['trajectory_pins']
    assert terminal['exit_code']==0 and terminal['process_was_terminal_before_log_hashing']
    for v in terminal['artifact_pins']:A.check_pin(v)
    for v in w['sources'].values():A.check_pin(v)
    A.check_pin(w['source_archive']['archive']);A.check_pin(w['source_archive']['inventory'])
    assert c['integration_steps']==w['world_steps_returned']==777600 and w['case_count']==54
    cc={key(x):x for x in c['cases']};wc={key(x):x for x in w['cases']}
    expected=set(itertools.product([.1,.2,.3],[.3,.6,.9],[.0025],[.0003125,.00015625],TESTS))
    assert len(cc)==len(wc)==54 and set(cc)==set(wc)==expected
    groups={tuple(g['group'][k] for k in ['mass_kg','mu','timeconst_s','dt_s']):g for g in w['groups']}
    assert len(groups)==18
    option_rows=[group_checks(g,cc) for g in w['groups']]
    p2=A.read(p['source_pins']['v2_protocol']['path']);A.check_pin(p['source_pins']['v2_protocol'])
    bounds=p2['engineering_gates']['G8_CPU_Warp_parity'];rows=[]
    for ident in sorted(expected):
        cpu,wp=cc[ident],wc[ident];assert cpu['case']==wp['case'] and wp['protocol_test_name']==ident[-1]
        cp=A.check_pin(cpu['trajectory']);wpin=A.check_pin(wp['trajectory']);ctpin=A.check_pin(wp['contact_table'])
        assert same_pin(cp,cr['trajectory_pins'][cpu['case']])
        with np.load(cp['path'],allow_pickle=False) as z:
            cd={k:z[k] for k in ['qpos','qvel','com_world_m','com_velocity_world_m_s','angular_velocity_world_rad_s','tick','contact_count']}
        with np.load(wpin['path'],allow_pickle=False) as z:wd={k:z[k] for k in z.files}
        with np.load(ctpin['path'],allow_pickle=False) as z:contacts={k:z[k] for k in z.files}
        n=round(3/ident[3]);tick=np.arange(n+1);dt=ident[3]
        assert np.array_equal(cd['tick'],tick) and np.array_equal(wd['integer_tick'],tick)
        assert cd['qpos'].shape==wd['qpos'].shape==(n+1,7) and cd['qvel'].shape==wd['qvel'].shape==(n+1,6)
        assert wd['qpos'].dtype==wd['qvel'].dtype==np.float32
        assert np.array_equal(wd['tick_derived_requested_time_s'],tick*dt)
        assert wp['dt_effective_s']==float(np.float32(dt))
        assert np.array_equal(wd['tick_derived_effective_time_s'],tick*float(np.float32(dt)))
        assert np.array_equal(wd['qpos'][0].view(np.uint8),cd['qpos'][0].astype(np.float32).view(np.uint8))
        assert np.array_equal(wd['qvel'][0].view(np.uint8),cd['qvel'][0].astype(np.float32).view(np.uint8))
        group=groups[ident[:4]];world=group['group']['worlds'].index(ident[-1])
        center=np.array(group['CPU_compiled_model_options']['body_ipos'][1],dtype=np.float64)
        C,Cv,Co=rigid(cd,center,'angular_velocity_world_rad_s');W,Wv,Wo=rigid(wd,center,'omega_world_rad_s')
        assert np.isfinite(wd['solver_qacc_at_state_tick']).all() and np.isfinite(wd['native_engine_energy_at_solver_state_tick_J']).all()
        flags=wd['overflow_flags'];assert flags.dtype.kind in 'iu' and np.all((flags>=0)&(flags<512))
        all_overflow_clear=not bool(np.any(flags))
        assert contacts['state_tick'].dtype.kind in 'iu' and np.all((contacts['state_tick']>=0)&(contacts['state_tick']<=n))
        assert np.all(contacts['worldid']==world) and np.isfinite(contacts['dist']).all()
        counts=np.bincount(contacts['state_tick'],minlength=n+1);assert np.array_equal(counts,wd['contact_count'])
        cf=np.flatnonzero(cd['contact_count']>0);wf=np.flatnonzero(counts>0)
        first_C=float(cf[0]*dt) if len(cf) else None;first_W=float(wf[0]*dt) if len(wf) else None
        dc,dv,dw=C-W,Cv-Wv,Co-Wo
        metrics={'final_COM_position_difference_m':float(np.linalg.norm(dc[-1])),
          'final_orientation_difference_deg':float(np.degrees(A.quat_angle(cd['qpos'][-1,3:],wd['qpos'][-1,3:]))),
          'final_COM_velocity_vector_difference_m_s':float(np.linalg.norm(dv[-1])),
          'final_angular_velocity_vector_difference_rad_s':float(np.linalg.norm(dw[-1])),
          'same_tick_COM_trajectory_RMS_difference_m':A.rms_vec(dc),
          'first_contact_time_difference_s':abs(first_C-first_W) if first_C is not None and first_W is not None else None}
        checks={k:(v is not None and v<=bounds[k]) for k,v in metrics.items()}
        rows.append({'numeric_identity':{'mass_kg':ident[0],'sliding_friction':ident[1],'timeconst_s':ident[2],'dt_s':dt,'test':ident[4]},
          'case':cpu['case'],'CPU_trajectory_pin':cp,'Warp_trajectory_pin':wpin,'Warp_contact_pin':ctpin,
          'shared_tick_count':n+1,'metrics':metrics,'checks':checks,'G8_status':A.status(all(checks.values())),
          'descriptive_shared_tick':{'COM_max_vector_difference_m':float(np.linalg.norm(dc,axis=1).max()),
             'velocity_RMS_vector_difference_m_s':A.rms_vec(dv),'velocity_max_vector_difference_m_s':float(np.linalg.norm(dv,axis=1).max()),
             'omega_RMS_vector_difference_rad_s':A.rms_vec(dw),'omega_max_vector_difference_rad_s':float(np.linalg.norm(dw,axis=1).max())},
          'first_generated_contact_requested_tick_time_s':{'CPU':first_C,'Warp':first_W},
          'clock_diagnostic_only':{'Warp_final_accumulated_time_s':float(wd['time_s'][-1]),'Warp_tick_effective_final_time_s':n*float(np.float32(dt)),
             'Warp_accumulated_minus_tick_effective_s':float(wd['time_s'][-1])-n*float(np.float32(dt))},
          'supported_diagnostics':{'finite_state_qacc_native_energy':True,'sticky_9bit_overflow_all_clear':all_overflow_clear,
             'native_CPU_style_GPU_warning_table':'NOT_AVAILABLE; not independently equivalent warning semantics'}})
    report={'schema':'phone_convex_fixedgrid_G8_saved_independent:4','status':'COMPLETED_INDEPENDENT_G8_SAVED_REVIEW',
      'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'auditor_pin':A.pin(__file__),'independent_math_helper_pin':A.pin(PURE),
      'author_of_Warp_producer':False,'reviewer_executed_physics':False,'audit_integrity_qualified':True,
      'case_count':54,'shared_ticks_compared':sum(x['shared_tick_count'] for x in rows),
      'protocol_pin':A.pin(PROTO),'CPU_result_pin':A.pin(CPU),'CPU_saved_assessment_pin':A.pin(CPU_REVIEW),'Warp_result_pin':A.pin(WARP),
      'parent_terminal_receipt_pin':A.pin(TERMINAL),'parent_terminal_scope':'Saved Root receipt checked, not a new process/OS query by this reviewer.',
      'group_option_checks':option_rows,'case_results':rows,
      'G8_case_counts':dict(collections.Counter(x['G8_status'] for x in rows)),
      'G8_qualified':all(x['G8_status']=='PASS' for x in rows),
      'physical_parameters_calibrated':False,'ego_hand_task_qualified':False,
      'limits':['G8 uses identical integer ticks and requested dt, without clock interpolation or posthoc alignment. Actual FP32dt/clock drift is reported separately.',
        'Raw state descriptors independently rebuilt with quaternion-vector rotation and original shared F64COM assumption; no quaternion repair except prescribed sign-invariant final angle normalization.',
        'Actual device solver scalar values are saved runtime JSON captures independently compared with CPU and FP32 casts. Full device model array equality remains producer source-backed because those raw arrays were not saved.',
        'CPUpost-forward work and WarpPRE/final-forward sampling counts differ. Contact event timing is compared, not contact-force impulses or whole hidden-state equivalence.',
        'GPU native CPU-style warning table unavailable. Finite/qacc/native-energy and sticky9-bit flags checked here, not a complete rerun of G2-G7 or a general diagnostic certificate.']}
    A.write(OUT/'all54_G8.json',rows);report['all54_pin']=A.pin(OUT/'all54_G8.json');A.write(OUT/'report.json',report)
    print(json.dumps({'report':A.pin(OUT/'report.json'),'G8_qualified':report['G8_qualified'],'case_counts':report['G8_case_counts'],
       'failed_cases':[x for x in rows if x['G8_status']=='FAIL'],'maximum_G8_metrics':{k:max(x['metrics'][k] for x in rows) for k in rows[0]['metrics']}},indent=2,allow_nan=False))


if __name__=='__main__':main()
