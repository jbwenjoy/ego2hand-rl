"""Saved-only MPC arithmetic and matched constant-u comparison; no simulator."""
from pathlib import Path
import hashlib, importlib.util, json, math
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
P=ROOT.parent
HELPER=P/'hand2_holding_v1/saved_recheck_v1/audit_saved_holding_v1.py'
HELPER_SHA='7cffd6bbbc4c3c3ed508df7d778cda6a9775533e395b3b2fb1fa08b1e2336ab1'
def pin(p):
    p=Path(p).resolve(strict=True); b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def checked(v):
    p=Path(v['path']); assert pin(p)==v,'Changed pin: '+str(p)
    return p
def same(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def close(a,b,tol=1e-10): return bool(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<=tol)
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f: json.dump(x,f,indent=2,allow_nan=False)
def load(pinrec):
    with np.load(checked(pinrec),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def audit():
    assert pin(HELPER)['sha256']==HELPER_SHA
    spec=importlib.util.spec_from_file_location('saved_rotation_helper',HELPER)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    rp=ROOT/'feasibility_fine_v1/report.json';r=read(rp);cp=checked(r['protocol']);protocol=read(cp)
    assert r['schema']=='native_true_model_MPC_feasibility_result:1'
    assert read(rp.parent/'protocol.json')==protocol
    for v in protocol['inputs'].values():checked(v)
    terminalpath=ROOT/'pilot_process_terminal_v1.json';terminal=read(terminalpath)
    for k in ['source','protocol','stdout','stderr']:checked(terminal[k])
    assert terminal['exit_code']==0 and terminal['stderr']['bytes']==0 and terminal['protocol']==pin(cp)
    state=load(protocol['inputs']['candidate_state']);q0=state['qpos27'];v0=state['qvel26'];u0=state['ctrl20']
    assert not np.any(v0) and same(state['x46'][26:],u0)
    baselinepath=checked(protocol['inputs']['matched_constant_u_report']);base=read(baselinepath);bp=read(checked(base['protocol']))
    assert bp['registered_init_and_command']['state']==protocol['inputs']['candidate_state']
    assert bp['inputs']['model_mjb']==protocol['inputs']['model_mjb'] and bp['inputs']['model_binding']==protocol['inputs']['model_binding']
    assert bp['gates']==protocol['gates']
    bc=[c for c in base['cases'] if c['dt_s']==protocol['config']['native_dt_s']];assert len(bc)==1
    bc=bc[0];ba=load(bc['trace'])
    baseline_review=P/'hand2_constant_u_baseline_v1/saved_recheck_v2/report.json';br=read(baseline_review)
    assert br['producer_report']==pin(baselinepath) and br['passed']==br['total']==9 and all(v is True for v in br['checks'].values())
    metadata=P/'hand2_stage1/completed_v2/joint_actuator_interface.json';mi=read(metadata)
    bounds=np.asarray([j['hard_range_rad'] for j in mi['joints']]);ubounds=np.asarray([a['ctrl_range'] for a in mi['actuators']])
    assert all(a['ctrl_limited'] is True for a in mi['actuators']) and mi['joint_order']==r['actuator_joint_names']
    geometry=P/'retarget_geometry_v2/report.json';names={int(k):v['name'] for k,v in read(geometry)['geometry_metadata'].items()}
    asset=P/'asset_stage1/geometry_report.json';ipos=np.asarray(read(asset)['CPU_compile']['phone_asset.xml']['body_ipos_m'])
    a=load(r['trace']);N=len(a['post_qpos']);dt=protocol['config']['native_dt_s'];G=protocol['gates']
    assert a['post_qpos'].shape==(N,27) and a['post_qvel'].shape==(N,26) and a['chosen_ctrl20'].shape==(N,20)
    assert all(np.isfinite(v).all() for v in a.values())
    assert same(a['initial_qpos27'],q0) and same(a['initial_qvel26'],v0) and same(a['initial_ctrl20'],u0)
    assert same(ba['initial_qpos27'],q0) and same(ba['ctrl20'],u0)
    ticks=np.arange(N,dtype=np.int64);clocktol=128*np.finfo(float).eps*max(1,N*dt)
    assert same(a['pre_tick'],ticks) and same(a['post_tick'],ticks+1)
    assert close(a['post_time_native_s'],(ticks+1)*dt,clocktol) and close(r['actual_final_time_s'],N*dt,clocktol)
    q=a['post_qpos'];v=a['post_qvel'];u=a['chosen_ctrl20'];COM0=q0[20:23]+h.rotate(q0[23:27],ipos)
    offset=h.rotate(q[:,23:27],ipos);COM=q[:,20:23]+offset;omega=h.rotate(q[:,23:27],v[:,23:26])
    vCOM=v[:,20:23]+np.cross(omega,offset);ang=h.angle(q[:,23:27],q0[23:27]);drift=np.linalg.norm(COM-COM0,axis=1)
    assert close(COM,a['post_COM'],1e-12) and close(vCOM,a['post_v_COM'],1e-12) and close(omega,a['post_omega_world'],1e-12)
    assert close(ang,a['post_angle'],8*math.sqrt(np.finfo(float).eps)) and close(COM0,a['initial_COM_world3'])
    assert np.all(u>=ubounds[:,0]) and np.all(u<=ubounds[:,1]) and same(u[:,8:],np.tile(u0[8:],(N,1)))
    cr=a['contact_rows24'];assert cr.shape[1]==24 and np.array_equal(cr[:,:5],cr[:,:5].astype(np.int64))
    assert np.all((cr[:,0]>=0)&(cr[:,0]<N)) and np.all(np.diff(cr[:,0])>=0)
    pad=np.zeros((N,2));other=np.zeros(N);table=np.zeros(N);pen=np.zeros(N);selfpen=np.zeros(N);pairs={}
    for row in cr:
        tick,g1,g2,_,_=map(int,row[:5]);assert g1 in names and g2 in names
        fn=max(0.,float(row[18]));pen[tick]=max(pen[tick],-float(row[5]));key=tuple(sorted((g1,g2)))
        pr=pairs.setdefault(key,dict(geom_ids=list(key),names=[names[g] for g in key],rows=0,positive_rows=0,max_normal_N=0.))
        pr['rows']+=1;pr['positive_rows']+=int(fn>0);pr['max_normal_N']=max(pr['max_normal_N'],fn)
        if 71 in key:
            g=g2 if g1==71 else g1
            if g in [15,28]:pad[tick,[15,28].index(g)]+=fn
            elif g==0:table[tick]+=fn
            else:other[tick]+=fn
        elif 0 not in key:selfpen[tick]=max(selfpen[tick],-float(row[5]))
    for x,k in [(pad,'pre_pad_normal_N'),(other,'pre_phone_nonpad_normal_N'),(table,'pre_table_normal_N'),(pen,'pre_penetration_m'),(selfpen,'pre_self_penetration_m')]:assert close(x,a[k])
    crossing=(drift>G['abort_COM_drift_m'])|(ang>G['abort_angle_rad'])
    assert r['termination']=='ABORT_LOSS_OF_HOLD' and crossing[-1] and not np.any(crossing[:-1])
    assert not np.any(ticks>=round(1.5/dt)) and not np.any(ticks+1>=round(1.5/dt))
    qvio=np.maximum(np.maximum(bounds[:,0]-q[:,:20],q[:,:20]-bounds[:,1]),0)
    metrics=dict(final_COM_drift_m=float(drift[-1]),final_angle_rad=float(ang[-1]),tail_linear_velocity_RMS_m_s=None,tail_angular_velocity_RMS_rad_s=None,tail_dual_pad_force_coverage=None,max_native_penetration_m=float(pen.max()),max_self_penetration_m=float(selfpen.max()),max_phone_nonpad_normal_force_N=float(other.max()),max_table_normal_force_N=float(table.max()),max_joint_limit_excursion_rad=float(qvio.max()),warning_count=r['metrics']['warning_count'])
    for k,val in metrics.items():assert r['metrics'][k] is None if val is None else close(val,r['metrics'][k])
    gates={'duration_complete':False,**{k:False if val is None else val>=G[k] if k=='tail_dual_pad_force_coverage' else val<=G[k] for k,val in metrics.items()}}
    assert gates==r['gates'] and not r['original_2s_single_dt_holding_pass'] and not r['two_dt_qualified']
    meanpad=np.asarray([math.fsum(pad[:,j])/N for j in range(2)])
    assert close(meanpad,r['mean_pad_normal_N']) and close(pad.max(axis=0),r['peak_pad_normal_N'])
    assert bool(np.all(meanpad<=3))==r['mean_pad_3N_design_budget_pass']
    interval_evidence=[];allbranches=[];intervalpins={};selectionpins={};previous=u0.copy()
    records=sorted(rp.parent.glob('interval_*.json'));assert len(records)==r['selected_intervals']==math.ceil(N/128)
    for i,path in enumerate(records):
        j=read(path);before=rp.parent/f'selection_before_actual_{i:03d}.json';b=read(before)
        assert {k:v for k,v in j.items() if not k.startswith('actual_')}==b
        assert j['interval']==i and j['start_tick']==i*128 and j['source_digest_before_actual']==b['source_digest_before_actual']
        ev=j['evaluations'];assert len(ev)<=protocol['config']['branches_per_interval_max']
        ids=[e['branch_index'] for e in ev];assert len(set(ids))==len(ids)
        selected=[e for e in ev if e['branch_index']==j['selected_branch']];assert len(selected)==1
        e=selected[0];chosen=np.asarray(j['selected_control20'],dtype=np.float64)
        lo=np.maximum(ubounds[:8,0],previous[:8]-.05);hi=np.minimum(ubounds[:8,1],previous[:8]+.05)
        for branch in ev:
            controls=np.asarray(branch['controls2x20'],dtype=np.float64)
            assert controls.shape==(2,20) and np.isfinite(controls).all() and np.all(controls>=ubounds[:,0]) and np.all(controls<=ubounds[:,1])
            assert np.all(controls[:,:8]>=lo) and np.all(controls[:,:8]<=hi) and same(controls[:,8:],np.tile(u0[8:],(2,1)))
            assert branch['source_state_before']==branch['source_state_after']==j['source_digest_before_actual'] and branch['complete_clone_equal'] is True
            assert branch['status']=='COMPLETE_H2_PLANNING_ONLY' and branch['actual_planning_steps']==256
            ag=branch['aggregates'];safe=all(ag[k]<=limit for k,limit in [('max_native_penetration_m',G['max_native_penetration_m']),('max_self_penetration_m',G['max_self_penetration_m']),('max_phone_nonpad_normal_N',G['max_phone_nonpad_normal_force_N']),('max_table_normal_N',G['max_table_normal_force_N']),('max_joint_limit_excursion_rad',G['max_joint_limit_excursion_rad']),('warning_count',0)])
            assert branch['original_horizon_safety_limits_pass']==safe
        assert e['original_horizon_safety_limits_pass'] is True and close(e['cost'],j['selected_cost']) and e['aggregates']==j['selected_horizon_aggregates']
        assert same(np.asarray(e['controls2x20']),np.tile(chosen,(2,1)))
        end=min((i+1)*128,N);assert same(u[i*128:end],np.tile(chosen,(end-i*128,1)));previous=chosen
        snap=j['selected_first_snapshot'];sv=np.asarray(snap['integration_state'],dtype=np.float64)
        sha=hashlib.sha256(sv.tobytes()).hexdigest()
        assert sv.shape==(300,) and snap['integration_state_signature']==16383 and sha==snap['integration_state_sha256']==e['snapshots'][0]['integration_state_sha256']
        assert {k:v for k,v in snap.items() if k!='integration_state'}==e['snapshots'][0]
        assert snap['post_tick']==128 and close(snap['elapsed_time_nominal_s'],.02) and close(snap['time_nominal_s'],(i+1)*.02)
        complete=end==(i+1)*128
        detail=dict(interval=i,start_tick=i*128,actual_steps=end-i*128,selected_branch=j['selected_branch'],branch_records=len(ev),selected_snapshot_sha256=sha,complete=complete)
        if complete:
            s=snap['state'];k=end-1
            for field,raw in [('q20',q[k,:20]),('qd20',v[k,:20]),('phone_root_world_m',q[k,20:23]),('phone_quaternion_wxyz',q[k,23:27]),('v_root_world_m_s',v[k,20:23])]:assert same(np.asarray(s[field],dtype=np.float64),raw)
            assert same(np.asarray(snap['native_time_s']),a['post_time_native_s'][k])
            for field,raw in [('COM_world_m',COM[k]),('v_COM_world_m_s',vCOM[k]),('omega_world_rad_s',omega[k])]:assert close(s[field],raw,1e-12)
            basis=np.eye(3);R=np.column_stack([h.rotate(q[k,23:27],x) for x in basis]);assert close(s['R_body_world'],R,1e-12)
            assert close(snap['pad_normal_N'],pad[k]) and close(snap['pad_force_pre_time_s'],k*dt)
            assert j['actual_boundary_integration_sha256']==sha and j['actual_vs_selected_first_state_raw_bit_equal'] is True
            detail.update(saved_POST_qpos_qvel_time_bits_match=True,explicit_POST_observers_match=True,selected_hash_matches_logged_actual_hash=True,actual_300_vector_independent_reconstruction=False)
        else:
            assert j['actual_vs_selected_first_state_raw_bit_equal']=='NOT_FULL_INTERVAL' and 'actual_boundary_integration_sha256' not in j
            detail['full_boundary_status']='NOT_FULL_INTERVAL_NO_COMPARISON'
        interval_evidence.append(detail);allbranches.extend(ev);intervalpins[path.name]=pin(path);selectionpins[before.name]=pin(before)
    ids=[e['branch_index'] for e in allbranches];assert ids==list(range(1,len(ids)+1))
    pc=r['planning_counts'];assert len(ids)==pc['branches_attempted']==pc['branches_returned']==pc['branch_MjData']==pc['mj_copyData']
    assert sum(e['actual_planning_steps'] for e in allbranches)==pc['planning_steps_attempted']==pc['planning_steps_returned']==pc['planning_steps_finite']
    assert N==r['actual_integration_steps']==r['actual_finite_rows'] and len(cr)==r['actual_contact_force_reads']
    assert r['all_complete_selected_intervals_raw_bit_equal'] is True and sum(e['complete'] for e in interval_evidence)==6
    # Only actually saved prefixes; no padding to 2 s and no success/benefit test.
    bq=ba['post_qpos'];bCOM=bq[:,20:23]+h.rotate(bq[:,23:27],ipos);bd=np.linalg.norm(bCOM-COM0,axis=1);bang=h.angle(bq[:,23:27],q0[23:27])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    for label,t,d,angle in [('constant u',(np.arange(len(bq))+1)*dt,bd,bang),('MPC',(ticks+1)*dt,drift,ang)]:
        ax[0,0].plot(t,1000*d,label=label);ax[0,1].plot(t,np.degrees(angle),label=label)
    for name,t,force in [('constant u',np.arange(len(bq))*dt,ba['pre_pad_normal_N']),('MPC',ticks*dt,pad)]:
        for col,style in [(0,'-'),(1,'--')]:ax[1,0].plot(t,force[:,col],style,label=name+(' thumb' if col==0 else ' index'),lw=1)
    ax[1,1].plot(ticks*dt,np.max(np.abs(u[:,:8]-u0[:8]),axis=1),label='MPC max |u-u0|, first8')
    ax[1,1].plot(np.arange(len(bq))*dt,np.zeros(len(bq)),label='constant u')
    for aa,title,y in zip(ax.flat,['COM drift (POST)','Rotation from initial (POST)','Pad normal force (PRE)','Held absolute control change (PRE)'],['mm','degrees','N','rad']):aa.set(title=title,xlabel='actual time (s)',ylabel=y);aa.grid(alpha=.25);aa.legend(fontsize=7)
    ax[0,0].axhline(50,color='k',ls=':',lw=1)
    fig.suptitle('Matched fine-dt actual prefixes: both lose hold; 2 s target not reached')
    fig.savefig(OUT/'constant_u_vs_MPC_actual_prefix.png',dpi=180);plt.close(fig)
    checks={k:True for k in ['input_source_model_protocol_pins','terminal_exit0_empty_stderr','matched_initial_state_ctrl_model_and_original_gates','all_raw_finites_integer_PRE_POST_clocks','all_POST_COM_velocity_omega_and_angle_observers','all_419_contact_rows_and_category_sums','all_actual_and_planning_controls_real_bounds_trust_rest12','first_abort_all_original_case_metrics_gates','seven_interval_selection_records_and_selected_branch_identity','all_161_branch_counts_clone_receipts_and_safety_descriptors','six_snapshot_hashes_and_saved_POST_observable_bits','partial54_preserved_no_full_boundary_claim','all_counts_separate_actual_and_planning']}
    return dict(schema='native_MPC_saved_arithmetic_recheck:1',status='PASS_SAVED_ARITHMETIC_HOLDING_FAILED_NO_BENEFIT_QUALIFICATION',producer_report=pin(rp),protocol=pin(cp),terminal=pin(terminalpath),source_pins={'auditor':pin(__file__),'rotation_helper':pin(HELPER),'joint_actuator_metadata':pin(metadata),'geometry_metadata':pin(geometry),'COM_metadata':pin(asset)},trace=r['trace'],baseline_report=pin(baselinepath),baseline_trace=bc['trace'],baseline_saved_review=pin(baseline_review),checks=checks,passed=sum(checks.values()),total=len(checks),actual_steps=N,planning_branches=len(ids),planning_steps=sum(e['actual_planning_steps'] for e in allbranches),actual_contact_rows=len(cr),actual_final_time_s=N*dt,metrics=metrics,original_gates=gates,contact_pairs=list(pairs.values()),mean_pad_normal_N=meanpad.tolist(),peak_pad_normal_N=pad.max(axis=0).tolist(),intervals=interval_evidence,interval_pins=intervalpins,selection_before_actual_pins=selectionpins,matched_prefix_descriptors={'constant_u_steps':len(bq),'constant_u_end_s':len(bq)*dt,'MPC_steps':N,'MPC_end_s':N*dt,'end_duration_difference_s':(N-len(bq))*dt,'both_2s_gate_pass':False,'same_initial_state_and_model':True},plot=pin(OUT/'constant_u_vs_MPC_actual_prefix.png'),holding_qualified=False,method_or_learning_benefit_qualified=False,scope='Pure saved NumPy arithmetic, no model loading, native forward, steps, LP or NN. All actual rows retained. Warning absence/model immutability/no aid are bound runtime/source assertions, not independently reconstructed from missing raw diagnostics. Selected full300 prediction hashes equal logged actual hashes for six intervals; only saved time/qpos/qvel and explicit POST observations independently compared; no actual raw300/warmstart/HISTORY/solver-scratch reconstruction. Planning clone receipts and horizon descriptors are checked as saved evidence, not replayed. No independent trajectory generalization, calibrated parameters, real-time controller, learning or policy benefit claim.')

if __name__=='__main__':
    try:result=audit()
    except Exception as exc:
        save(OUT/'failure.json',dict(type=type(exc).__name__,message=str(exc),source=pin(__file__),no_automatic_retry=True));raise
    save(OUT/'report.json',result)
    print(json.dumps({'report':pin(OUT/'report.json'),'passed':result['passed'],'total':result['total']}))
