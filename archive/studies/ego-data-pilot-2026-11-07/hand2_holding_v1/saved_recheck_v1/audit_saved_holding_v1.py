"""Independent saved NumPy/contact arithmetic only: no simulator or NN import."""
from pathlib import Path
import csv,hashlib,json,math,sys
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
P=ROOT.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def checked(x):
    if pin(x['path'])!=x:raise ValueError('Changed pin '+x['path'])
    return Path(x['path'])
def save(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)
def cross(a,b):return np.cross(a,b)
def rotate(q,v):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);u=q[...,1:];w=q[...,:1]
    return v+2*w*cross(u,v)+2*cross(u,cross(u,v))
def angle(q,q0):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);q0=q0/np.linalg.norm(q0)
    # Relative quaternion vector; atan2 avoids acos cancellation at initial angle0.
    w=q[...,0]*q0[0]+np.sum(q[...,1:]*q0[1:],axis=-1)
    v=-q[...,:1]*q0[1:]+q0[0]*q[...,1:]-cross(q[...,1:],q0[1:])
    return 2*np.arctan2(np.linalg.norm(v,axis=-1),np.abs(w))
def near(a,b,tol=1e-10):return bool(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<=tol)


def main():
    if (OUT/'report.json').exists():raise ValueError('Fresh assessment only')
    producer=ROOT/'cpu_nominal_v1/report.json';r=read(producer);protocol=read(checked(r['protocol']))
    inputs={k:checked(v) for k,v in protocol['inputs'].items()}
    for v in protocol['source_pins'].values():checked(v)
    if (ROOT/'cpu_nominal_v1/execution_source.py').read_bytes()!=Path(protocol['source_pins']['runner']['path']).read_bytes():raise ValueError('Executed source differs')
    if read(ROOT/'cpu_nominal_v1/protocol.json')!=protocol:raise ValueError('Copied protocol semantic mismatch')
    terminal=read(ROOT/'cpu_nominal_v1_process_terminal.json')
    checked(terminal['stdout']);checked(terminal['stderr'])
    if terminal['returncode']!=0 or terminal['stderr']['bytes']!=0:raise ValueError('Terminal process evidence')
    geometry=read(inputs['geometry_report']);names={int(k):v['name'] for k,v in geometry['geometry_metadata'].items()}
    reverse={v:k for k,v in names.items()};phone=reverse['phone24_collision_convex'];table=reverse['stage1_single_table'];pads=[reverse['right_finger1_tip_sensor_col'],reverse['right_finger2_tip_sensor_col']]
    prep=read(inputs['support_report']);variants={v['name']:v for v in prep['controller_variants']}
    with np.load(inputs['candidate_state'],allow_pickle=False) as z:q0=z['qpos27'].copy()
    asset=P/'asset_stage1/geometry_report.json';ipos=np.array(read(asset)['CPU_compile']['phone_asset.xml']['body_ipos_m'])
    interface=P/'hand2_stage1/completed_v2/joint_actuator_interface.json';joints=read(interface)['joints'];qlimits=np.array([x['hard_range_rad'] for x in joints])
    gates=protocol['gates'];cases=[];plot_data=[];step_csv=[];pair_reports=[];contact_csv=[]
    checks={'input_source_model_byte_pins':True,'executed_source_and_protocol_match':True,'terminal_exit0_empty_stderr':True}
    totalN=totalcontacts=0;blocked=[]
    for c in r['cases']:
        if 'trace' not in c:
            if variants[c['name']]['reachable'] or c['integration_steps']!=0:raise ValueError('Invalid blocked control')
            blocked.append(c);continue
        trace=checked(c['trace'])
        with np.load(trace,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        dt=c['dt_s'];N=len(a['post_qpos']);totalN+=N;cr=a['contact_rows'];totalcontacts+=len(cr)
        key=f"{c['name']}__dt{round(dt*1e9)}ns";pre_ticks=np.arange(N);post_ticks=pre_ticks+1
        q=a['post_qpos'];v=a['post_qvel'];norms=np.linalg.norm(q[:,23:27],axis=1)
        if not all(np.isfinite(x).all() for x in a.values()):raise ValueError('Nonfinite raw saved array')
        if not np.array_equal(a['initial_qpos27'],q0) or not np.array_equal(a['ctrl20'],np.asarray(variants[c['name']]['ctrl20'])):raise ValueError('Initial/control byte mismatch')
        if cr.ndim!=2 or cr.shape[1]!=24 or not np.array_equal(cr[:,:5],cr[:,:5].astype(np.int64)):raise ValueError('Contact24 index ABI')
        if np.any(cr[:,0]<0) or np.any(cr[:,0]>=N) or np.any(np.diff(cr[:,0])<0):raise ValueError('Contact tick order')
        if c['integration_steps']!=N or c['finite_steps_saved']!=N:raise ValueError('Attempted versus saved step count')
        clocktol=128*np.finfo(float).eps*max(1.,N*dt)
        if not near(a['post_time'],post_ticks*dt,clocktol) or not near(a['pre_time_from_integer_tick'],pre_ticks*dt,clocktol):raise ValueError('PRE/POST integer clock mismatch')
        COM=q[:,20:23]+rotate(q[:,23:27],ipos);COM0=q0[20:23]+rotate(q0[23:27],ipos)
        drift=np.linalg.norm(COM-COM0,axis=1);angles=angle(q[:,23:27],q0[23:27])
        rawacos=2*np.arccos(np.clip(np.abs(q[:,23:27]@q0[23:27]),0,1))
        # Acos is ill-conditioned at angle0: capture check has its explicit F64
        # cancellation envelope; final45deg metric checks below remain1e-10.
        if not near(COM,a['post_COM'],1e-12) or not near(rawacos,a['post_angle'],8*math.sqrt(np.finfo(float).eps)):raise ValueError('State capture arithmetic mismatch')
        pad=np.zeros((N,2));nonpad=np.zeros(N);tab=np.zeros(N);pen=np.zeros(N);selfpen=np.zeros(N);hist=np.zeros(N,dtype=int)
        pairs={};groups=[[] for _ in range(N)]
        for rowid,row in enumerate(cr):
            tick,g1,g2,dim,efc=map(int,row[:5]);hist[tick]+=1;ci=hist[tick]-1
            if g1 not in names or g2 not in names:raise ValueError('Unmapped raw geom ID')
            Fn=float(row[18]);positive=max(0.,Fn);pen[tick]=max(pen[tick],-row[5])
            if phone in (g1,g2):
                other=g2 if g1==phone else g1
                category='pad_thumb' if other==pads[0] else 'pad_index' if other==pads[1] else 'table' if other==table else 'phone_nonpad'
            else:category='self_or_other'
            if category=='self_or_other' and table not in (g1,g2):selfpen[tick]=max(selfpen[tick],-row[5])
            pair=tuple(sorted((g1,g2)));rec=pairs.setdefault(pair,{'geom_ids':list(pair),'names':[names[i] for i in pair],'category':category,'rows':0,'positive_rows':0,'first_positive_pre_tick':None,'max_normal_N':0.,'sum_normal_samples':[]})
            rec['rows']+=1;rec['max_normal_N']=max(rec['max_normal_N'],positive);rec['sum_normal_samples'].append(positive)
            if Fn>0:
                rec['positive_rows']+=1
                if rec['first_positive_pre_tick'] is None:rec['first_positive_pre_tick']=tick
            groups[tick].append((category,positive))
            contact_csv.append([key,rowid,ci,*row.tolist(),names[g1],names[g2],category])
        for tick,rows in enumerate(groups):
            for i,cat in enumerate(['pad_thumb','pad_index']):pad[tick,i]=math.fsum(f for k,f in rows if k==cat)
            nonpad[tick]=math.fsum(f for k,f in rows if k=='phone_nonpad');tab[tick]=math.fsum(f for k,f in rows if k=='table')
        if not np.array_equal(hist,a['pre_ncon']):raise ValueError('Every generated native contact row not retained')
        for actual,ref in [(pad,a['pre_pad_normal_N']),(nonpad,a['pre_phone_nonpad_normal_N']),(tab,a['pre_table_normal_N']),(pen,a['pre_penetration_m']),(selfpen,a['pre_self_penetration_m'])]:
            if not near(actual,ref,1e-10):raise ValueError('Raw contact aggregation differs')
        tail_start=round((gates['duration_s']-gates['tail_duration_s'])/dt)
        tailPOST=post_ticks>=tail_start;tailPRE=pre_ticks>=tail_start
        qvio=np.maximum(qlimits[:,0]-q[:,:20],0)+np.maximum(q[:,:20]-qlimits[:,1],0)
        metrics={'final_COM_drift_m':float(drift[-1]),'final_angle_rad':float(angles[-1]),
          'tail_linear_velocity_RMS_m_s':float(np.sqrt(np.mean(np.sum(v[tailPOST,20:23]**2,axis=1)))) if np.any(tailPOST) else None,
          'tail_angular_velocity_RMS_rad_s':float(np.sqrt(np.mean(np.sum(v[tailPOST,23:26]**2,axis=1)))) if np.any(tailPOST) else None,
          'tail_dual_pad_force_coverage':float(np.mean(np.all(pad[tailPRE]>gates['tail_each_pad_normal_force_N'],axis=1))) if np.any(tailPRE) else None,
          'max_native_penetration_m':float(pen.max()),'max_self_penetration_m':float(selfpen.max()),'max_phone_nonpad_normal_force_N':float(nonpad.max()),'max_table_normal_force_N':float(tab.max()),
          'max_joint_limit_excursion_rad':float(qvio.max()),'warning_count':sum(c['warnings'])}
        for k,val in metrics.items():
            if val is None:
                if c['metrics'][k] is not None:raise ValueError('Missing tail must remain null')
            elif abs(val-c['metrics'][k])>1e-10:raise ValueError('Producer metric mismatch '+k)
        loss=(drift>gates['abort_COM_drift_m'])|(angles>gates['abort_angle_rad'])
        if np.any(loss[:-1]) or not loss[-1] or c['termination']!='ABORT_LOSS_OF_HOLD':raise ValueError('Abort first-crossing mismatch')
        statuses={'duration_complete':'FAIL'}
        for k,val in metrics.items():statuses[k]='NOT_RUN' if val is None else 'PASS_OBSERVED_PREFIX' if (val>=gates[k] if k=='tail_dual_pad_force_coverage' else val<=gates[k]) else 'FAIL'
        expected_bools={k:val=='PASS_OBSERVED_PREFIX' for k,val in statuses.items()}
        if expected_bools!=c['gates'] or c['nominal_holding_dynamics_pass']:raise ValueError('Original gates must remain failed')
        events=[]
        for rec in pairs.values():
            rec['normal_scalar_integral_Ns']=math.fsum(rec.pop('sum_normal_samples'))*dt
            t=rec['first_positive_pre_tick']
            if t is not None:
                rec['first_positive_pre_time_s']=t*dt;rec['object_angle_before_interval_deg']=math.degrees(float(angles[t-1])) if t else 0.
                rec['object_angle_after_interval_deg']=math.degrees(float(angles[t]));events.append((t,rec['category']))
        first=min((t for t,k in events if k.startswith('pad_') or k=='phone_nonpad'),default=None)
        for k in range(N):step_csv.append([key,k,(k+1),k*dt,(k+1)*dt,drift[k],math.degrees(angles[k]),*pad[k],nonpad[k],tab[k],pen[k]])
        case={'case_id':key,'name':c['name'],'dt_s':dt,'trace':c['trace'],'steps':N,'actual_last_POST_tick':N,'actual_last_PRE_tick':N-1,'actual_last_POST_time_s':N*dt,'termination':c['termination'],
          'metrics':metrics,'gate_statuses':statuses,'holding_pass':False,'tail_POST_count':int(tailPOST.sum()),'tail_PRE_count':int(tailPRE.sum()),
          'state_COM_capture_max_difference_m':float(np.max(np.abs(COM-a['post_COM']))),'normalized_atan2_vs_native_raw_acos_max_difference_rad':float(np.max(np.abs(angles-a['post_angle']))),
          'quaternion_norm_max_error':float(np.max(np.abs(norms-1))),
          'first_object_positive_contact_PRE_tick':first,'first_object_positive_contact_PRE_time_s':first*dt if first is not None else None,
          'object_angle_before_first_force_deg':math.degrees(float(angles[first-1])) if first else 0.,
          'max_object_angular_speed_rad_s':float(np.linalg.norm(v[:,23:26],axis=1).max()),
          'max_pad_normal_N':pad.max(0).tolist(),'dual_pad_above_0p1N_observed_prefix_fraction':float(np.mean(np.all(pad>.1,axis=1))),
          'pair_summaries':list(pairs.values())}
        cases.append(case);pair_reports.append({'case':key,'pairs':list(pairs.values())});plot_data.append((case,COM,angles,pad,nonpad))
    if totalN!=r['counts']['integration_steps'] or totalcontacts!=r['counts']['contact_force_reads']:raise ValueError('Global saved step/contact counts')
    checks.update({'all4_initial_controls_match_preparation':True,'all_POST_state_and_integer_phase_recomputed':True,'all_contact24_rows_and_force_aggregates_recomputed':True,'abort_is_first_original_limit_crossing_all4':True,'original_failed_gates_and_absent_tail_preserved':True})
    dtpairs=[]
    for name in sorted({x[0]['name'] for x in plot_data}):
        records=sorted([x for x in plot_data if x[0]['name']==name],key=lambda x:x[0]['dt_s'],reverse=True)
        co,fi=records;Nc=min(co[0]['steps'],fi[0]['steps']//2);j=2*np.arange(Nc)+1
        dc=co[1][:Nc]-fi[1][j];da=np.abs(co[2][:Nc]-fi[2][j])
        dtpairs.append({'name':name,'matched_POST_count':Nc,'last_common_POST_time_s':Nc*co[0]['dt_s'],'coarse_final_time_s':co[0]['actual_last_POST_time_s'],'fine_final_time_s':fi[0]['actual_last_POST_time_s'],
           'common_COM_vector_RMS_m':float(np.sqrt(np.mean(np.sum(dc*dc,axis=1)))),'common_COM_max_m':float(np.linalg.norm(dc,axis=1).max()),'common_drift_angle_difference_max_deg':math.degrees(float(da.max())),
           'scope':'Descriptive two-step comparison of observed early-abort prefixes, not completed2s timestep qualification'})
    save(OUT/'contact_pair_summaries.json',pair_reports)
    for file,header,rows in [('step_metrics.csv',['case_id','PRE_tick','POST_tick','PRE_time_s','POST_time_s','COM_drift_m','angle_deg','thumb_N','index_N','phone_nonpad_N','table_N','penetration_m'],step_csv),
      ('contact_rows_named.csv',['case_id','original_long_row','contact_index_within_PRE_tick','pre_tick','geom1','geom2','dim','efc_address','distance_m','posx','posy','posz',*[f'frame{i}' for i in range(9)],*[f'wrench{i}' for i in range(6)],'geom1_name','geom2_name','category'],contact_csv)]:
        with (OUT/file).open('x',newline='') as f:w=csv.writer(f);w.writerow(header);w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True);colors=['#1664b1','#43a6c6','#dc781c','#b72525'];handles=[]
    for (case,COM,ang,pad,nonpad),color in zip(plot_data,colors):
        dt=case['dt_s'];N=case['steps'];post=np.arange(1,N+1)*dt;pre=np.arange(N)*dt;label=('A' if case['name']=='gravity_compensation' else 'B')+f" dt={dt:g}s"
        axes[0,0].plot(post,np.linalg.norm(COM-(q0[20:23]+rotate(q0[23:27],ipos)),axis=1)*1000,color=color)
        axes[0,1].plot(post,np.degrees(ang),color=color)
        axes[1,0].plot(pre,pad[:,0],color=color);axes[1,0].plot(pre,pad[:,1],color=color,linestyle='--')
        axes[1,1].plot(pre,nonpad,color=color);handles.append(Line2D([0],[0],color=color,label=label))
    axes[0,0].axhline(5,color='gray',ls=':',lw=1);axes[0,0].set_ylabel('COM drift [mm], POST')
    axes[0,1].axhline(45,color='gray',ls=':',lw=1);axes[0,1].set_ylabel('Orientation change [deg], POST')
    axes[1,0].set_ylabel('Pad normal sum [N], PRE');axes[1,1].set_ylabel('Phone–nonpad normal sum [N], PRE')
    axes[1,0].legend(handles=[Line2D([0],[0],color='black',label='thumb'),Line2D([0],[0],color='black',ls='--',label='index')],fontsize=8)
    for ax in axes.ravel():ax.set_xlabel('Actual saved time [s]');ax.grid(alpha=.25);ax.set_xlim(left=0)
    fig.legend(handles=handles,loc='outside lower center',ncols=4,fontsize=8)
    fig.suptitle('Nominal holding early aborts: all four stop near0.1s; target2.0s not reached\nA: gravity compensation; B: two-pad3N normal preload. No extrapolation.',fontsize=11)
    fig.savefig(OUT/'holding_summary.png',dpi=170);plt.close(fig)
    checks['no_simulator_or_neural_modules_imported']=not any(k in sys.modules for k in ['mujoco','torch','warp','mujoco_warp'])
    report={'schema':'hand2_nominal_holding_saved_independent_recheck:1','status':'PASS_SAVED_ARITHMETIC_ALL4_HOLDING_FAILED',
       'source':pin(__file__),'producer_report':pin(producer),'protocol':r['protocol'],'terminal':pin(ROOT/'cpu_nominal_v1_process_terminal.json'),
       'asset_COM_metadata':pin(asset),'actual_joint_limits_metadata':pin(interface),'checks':checks,'passed':sum(checks.values()),'total':len(checks),
       'actual_cases':len(cases),'blocked_controls':blocked,'steps_saved':totalN,'generated_contact_rows':totalcontacts,'cases':cases,'dt_pair_comparisons':dtpairs,
       'independent_saved_arithmetic_qualified':all(checks.values()),'holding_qualified':False,
       'scope':'No model load/forward/step/NN. POST COM uses saved qpos and pinned declared body COM; quaternion-vector observer normalized explicitly. Raw PRE native contactForce sums are rebuilt with fsum. Runtime warning/no-applied-force guards and actual model parameters remain pinned-source/producer assertions, not a new model/native-warning reconstruction. Contact-reaction timing and nonzero angular motion contradict torque-free free fall but do not uniquely assign the moment to a pad/nonpad contact. No proof for unrepresented geometry or calibrated real hand/material.',
       'outputs':{f.name:pin(f) for f in OUT.iterdir() if f.is_file() and f.name!='report.json'}}
    save(OUT/'report.json',report);print(json.dumps({'report':pin(OUT/'report.json'),'checks':f"{report['passed']}/{report['total']}",'holding':False,'steps':totalN,'contacts':totalcontacts}))


if __name__=='__main__':main()
