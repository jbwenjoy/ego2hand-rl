"""One saved-arithmetic recheck; no simulator, NN, FK or additional LP."""
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
    p=Path(v['path']); assert pin(p)==v, 'Pin mismatch '+str(p)
    return p
def same(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def close(a,b,tol=1e-10):
    return np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<=tol
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(x,f,indent=2,allow_nan=False)

def audit():
    assert pin(HELPER)['sha256']==HELPER_SHA
    spec=importlib.util.spec_from_file_location('saved_rotation_only',HELPER)
    h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    rp=ROOT/'cpu_v1/report.json'; r=read(rp); protocol=read(checked(r['protocol']))
    for v in [*protocol['inputs'].values(),*protocol['source_pins'].values()]: checked(v)
    assert (ROOT/'cpu_v1/execution_source.py').read_bytes()==checked(protocol['source_pins']['runner']).read_bytes()
    assert read(ROOT/'cpu_v1/protocol.json')==protocol
    assert read(checked(protocol['source_pins']['source_preservation47']))['passed']==47
    terminal=read(ROOT/'actual_process_terminal_v1.json')
    checked(terminal['stdout']); checked(terminal['stderr'])
    assert terminal['returncode']==0 and terminal['stderr']['bytes']==0
    statepin=protocol['registered_init_and_command']['state']; checked(statepin)
    with np.load(statepin['path'],allow_pickle=False) as z: state={k:z[k].copy() for k in z.files}
    q0=state['qpos27']; ctrl=state['ctrl20']
    assert same(ctrl,state['x46'][26:]) and np.count_nonzero(state['qvel26'])==0
    asset=P/'asset_stage1/geometry_report.json'; ipos=np.asarray(read(asset)['CPU_compile']['phone_asset.xml']['body_ipos_m'])
    interface=P/'hand2_stage1/completed_v2/joint_actuator_interface.json'
    bounds=np.asarray([j['hard_range_rad'] for j in read(interface)['joints']])
    geom=P/'retarget_geometry_v2/report.json'; names={int(k):v['name'] for k,v in read(geom)['geometry_metadata'].items()}
    phone,table,pads=71,0,[15,28]
    G=protocol['gates']; cases=[]; plots=[]; total=contacts=0
    checks={'protocol_input_source_model_pins':True,'execution_source_semantic_protocol_and47AST':True,'terminal_exit0_and_empty_stderr':True}
    for c in r['cases']:
        checked(c['trace']); assert c['source_candidate_state']==statepin
        with np.load(c['trace']['path'],allow_pickle=False) as z: a={k:z[k].copy() for k in z.files}
        N=len(a['post_qpos']); dt=c['dt_s']; total+=N; cr=a['contact_rows']; contacts+=len(cr)
        assert N==c['integration_steps']==c['finite_steps_saved']
        assert all(np.isfinite(v).all() for v in a.values())
        assert same(a['initial_qpos27'],q0) and same(a['ctrl20'],ctrl)
        ticks=np.arange(N); clocktol=128*np.finfo(float).eps*max(1.,N*dt)
        assert close(a['post_time'],(ticks+1)*dt,clocktol) and close(a['pre_time_from_integer_tick'],ticks*dt,clocktol)
        assert close(c['actual_final_time_s'],N*dt,clocktol)
        for tick,pre in zip(a['sample_pre_tick'],a['sample_pre_qpos']): assert same(pre,q0 if tick==0 else a['post_qpos'][tick-1])
        q=a['post_qpos']; COM0=q0[20:23]+h.rotate(q0[23:27],ipos)
        COM=q[:,20:23]+h.rotate(q[:,23:27],ipos); drift=np.linalg.norm(COM-COM0,axis=1)
        ang=h.angle(q[:,23:27],q0[23:27])
        assert close(COM,a['post_COM'],1e-12) and close(ang,a['post_angle'],8*math.sqrt(np.finfo(float).eps))
        assert close(COM0,a['initial_COM_world3'])
        assert cr.shape[1]==24 and np.array_equal(cr[:,:5],cr[:,:5].astype(np.int64))
        assert np.all((cr[:,0]>=0)&(cr[:,0]<N)) and np.all(np.diff(cr[:,0])>=0)
        pad=np.zeros((N,2)); other=np.zeros(N); tab=np.zeros(N); pen=np.zeros(N); selfpen=np.zeros(N); ncon=np.zeros(N,dtype=np.int64); pairs={}
        for row in cr:
            k,g1,g2,_,_=map(int,row[:5]); assert g1 in names and g2 in names
            ncon[k]+=1; fn=max(0.,float(row[18])); pen[k]=max(pen[k],-float(row[5]))
            key=tuple(sorted((g1,g2))); entry=pairs.setdefault(key,dict(geom_ids=list(key),names=[names[g] for g in key],rows=0,positive_rows=0,max_normal_N=0.))
            entry['rows']+=1; entry['positive_rows']+=int(fn>0); entry['max_normal_N']=max(entry['max_normal_N'],fn)
            if phone in key:
                g=g2 if g1==phone else g1
                if g in pads: pad[k,pads.index(g)]+=fn
                elif g==table: tab[k]+=fn
                else: other[k]+=fn
            elif table not in key: selfpen[k]=max(selfpen[k],-float(row[5]))
        for x,y in [(ncon,a['pre_ncon']),(pad,a['pre_pad_normal_N']),(other,a['pre_phone_nonpad_normal_N']),(tab,a['pre_table_normal_N']),(pen,a['pre_penetration_m']),(selfpen,a['pre_self_penetration_m'])]: assert close(x,y)
        crossings=(drift>G['abort_COM_drift_m']) | (ang>G['abort_angle_rad'])
        assert c['termination']=='ABORT_LOSS_OF_HOLD' and crossings[-1] and not np.any(crossings[:-1])
        tail_tick=round((G['duration_s']-G['tail_duration_s'])/dt)
        assert not np.any(ticks>=tail_tick) and not np.any(ticks+1>=tail_tick)
        qvio=np.maximum(np.maximum(bounds[:,0]-q[:,:20],q[:,:20]-bounds[:,1]),0)
        metrics=dict(final_COM_drift_m=float(drift[-1]),final_angle_rad=float(ang[-1]),tail_linear_velocity_RMS_m_s=None,tail_angular_velocity_RMS_rad_s=None,tail_dual_pad_force_coverage=None,max_native_penetration_m=float(pen.max()),max_self_penetration_m=float(selfpen.max()),max_phone_nonpad_normal_force_N=float(other.max()),max_table_normal_force_N=float(tab.max()),max_joint_limit_excursion_rad=float(qvio.max()),warning_count=sum(c['warnings']))
        for k,v in metrics.items(): assert c['metrics'][k] is None if v is None else close(v,c['metrics'][k])
        gates={'duration_complete':False,**{k:False if v is None else v>=G[k] if k=='tail_dual_pad_force_coverage' else v<=G[k] for k,v in metrics.items()}}
        assert gates==c['gates'] and not c['nominal_holding_dynamics_pass']
        last=[np.flatnonzero(pad[:,j]>0) for j in range(2)]
        cases.append(dict(dt_s=dt,steps=N,last_POST_time_s=N*dt,last_PRE_time_s=(N-1)*dt,termination=c['termination'],metrics=metrics,original_gates=gates,tail_window_status='NOT_RUN_ABSOLUTE_1P5_TO_2S',first_abort_POST_tick=N,contact_rows=len(cr),contact_pairs=list(pairs.values()),max_pad_normal_N=pad.max(axis=0).tolist(),pad_normal_impulse_Ns=[dt*math.fsum(pad[:,j]) for j in range(2)],last_positive_pad_PRE_time_s=[None if len(v)==0 else int(v[-1])*dt for v in last]))
        plots.append((dt,ticks*dt,(ticks+1)*dt,drift,ang,pad,other))
    assert len(cases)==2 and {c['dt_s'] for c in cases}==set(protocol['dt_s'])
    assert total==r['counts']['integration_steps'] and contacts==r['counts']['contact_force_reads']
    pair=r['dt_pair_consistency']; qends=[]; comends=[]
    for c in r['cases']:
        with np.load(c['trace']['path'],allow_pickle=False) as z:
            qe=z['post_qpos'][-1]; qends.append(qe[23:27]); comends.append(qe[20:23]+h.rotate(qe[23:27],ipos))
    dcom=np.linalg.norm(comends[0]-comends[1])
    dangle=float(h.angle(qends[0],qends[1])); assert close(dcom,pair['final_COM_difference_m']) and close(dangle,pair['final_orientation_difference_rad'])
    assert not pair['dt_pair_consistency_pass'] and not pair['gates']['both_duration_complete'] and not r['all_original_cases_and_dt_pairs_pass']
    checks.update(all_initial_state_and_constant_ctrl_bytes=True,integer_PRE_POST_clocks_and_sample_phase=True,all_POST_COM_and_angle_observer=True,all_contact24_category_sums_and_metrics=True,first_abort_and_all_original_case_gates=True,all_counts_and_early_endpoint_dt_pair=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    for dt,pre,post,drift,ang,pad,other in plots:
        label=f'dt={dt:g}s'
        ax[0,0].plot(post,drift*1000,label=label); ax[0,1].plot(post,np.degrees(ang),label=label)
        ax[1,0].plot(pre,pad[:,0],label=label+' thumb');ax[1,0].plot(pre,pad[:,1],'--',label=label+' index')
        ax[1,1].plot(pre,other,label=label)
    for a,title,y in zip(ax.flat,['COM drift (POST)','Rotation from initial (POST)','Pad normal force (PRE)','Phone nonpad normal force (PRE)'],['mm','degrees','N','N']): a.set(title=title,xlabel='actual time (s)',ylabel=y);a.grid(alpha=.25);a.legend(fontsize=7)
    ax[0,0].axhline(50,color='k',ls=':',lw=1)
    fig.suptitle('Matched constant-u: both abort at 0.1225 s; 2 s target not reached')
    fig.savefig(OUT/'actual_prefix.png',dpi=180);plt.close(fig)
    return dict(schema='hand2_phone_matched_constant_u_saved_recheck:1',status='PASS_SAVED_ARITHMETIC_RECHECK_HOLDING_FAILED',producer_report=pin(rp),protocol=pin(ROOT/'cpu_protocol_v1.json'),terminal=pin(ROOT/'actual_process_terminal_v1.json'),source_pins={'auditor':pin(__file__),'saved_rotation_helper':pin(HELPER),'COM_metadata':pin(asset),'joint_limits':pin(interface),'geometry_ID_metadata':pin(geom)},checks=checks,passed=sum(checks.values()),total=len(checks),case_count=2,integration_steps=total,contact_rows=contacts,cases=cases,dt_pair={'endpoint_COM_difference_m':float(dcom),'endpoint_orientation_difference_rad':dangle,'both_duration_complete':False,'pair_qualified':False},plot=pin(OUT/'actual_prefix.png'),holding_qualified=False,scope='Saved arithmetic recheck by the baseline runner adapter author. One actual physical run only; no independent model execution. COM and orientation use the pinned F64 observer; full model/zero external aid evidence is source/runtime-backed. No tail extrapolation, material calibration, policy effect or general holding qualification.')

if __name__=='__main__':
    try: result=audit()
    except Exception as exc:
        save(OUT/'failure.json',dict(type=type(exc).__name__,message=str(exc),source=pin(__file__),no_automatic_retry=True));raise
    save(OUT/'report.json',result)
    print(json.dumps({'report':pin(OUT/'report.json'),'passed':result['passed'],'total':result['total']}))
