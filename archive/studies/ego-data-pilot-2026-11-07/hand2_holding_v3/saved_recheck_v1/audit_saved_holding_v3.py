"""Saved holding_v3 check and actual-prefix figure; no simulator/solver import."""
from pathlib import Path
import csv,hashlib,importlib.util,json,math,sys
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;P=ROOT.parent;RUN=ROOT/'cpu_nominal_v1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def checked(x):
    if pin(x['path'])!=x:raise ValueError('Changed pin '+x['path'])
    return Path(x['path'])
def need(x,label):
    if not x:raise AssertionError(label)
def close(a,b,tol=1e-10):return bool(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<=tol)
def save(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)


def main():
    need(not (OUT/'report.json').exists(),'fresh audit')
    helper=P/'hand2_holding_v1/saved_recheck_v1/audit_saved_holding_v1.py'
    need(pin(helper)['sha256']=='7cffd6bbbc4c3c3ed508df7d778cda6a9775533e395b3b2fb1fa08b1e2336ab1','frozen independent quaternion helpers')
    spec=importlib.util.spec_from_file_location('independent_quat_helpers_v1',helper);A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
    rf=RUN/'report.json';r=read(rf);c=read(checked(r['protocol']))
    need(c['schema']=='hand2_phone_support_CPU_holding_protocol:3' and r['protocol']['sha256']=='5a2f04b1ae6005e12f0944726c8ac3dfee9cb4159073131b099f83a842e62d94','new family actual protocol')
    for v in [*c['inputs'].values(),*c['source_pins'].values()]:checked(v)
    need((RUN/'execution_source.py').read_bytes()==checked(c['source_pins']['runner']).read_bytes() and read(RUN/'protocol.json')==c,'source/copy identity')
    terminal=read(ROOT/'cpu_nominal_v1_process_terminal.json');need(terminal['returncode']==0,'terminal0');checked(terminal['stdout']);checked(terminal['stderr']);need(terminal['stderr']['bytes']==0,'stderr empty')
    support=read(checked(c['inputs']['support_pose_report']));S={s['seed']:s for s in support['seed_results']};checks={'all_input_source_model_byte_pins_and_terminal':True}
    screen={x['seed']:x for x in r['screening']};need(screen[0]['enrolled'] and not screen[1]['enrolled'],'only seed0 enrolled')
    independent=P/'retarget_gravity_v3/support_refinement_v5_saved_recheck/report.json';ind=read(independent)
    need(ind['eligible_for_bounded_dynamic_diagnostic_seed_ids']==[0] and ind['strict_support_proposal_count']==0,'independent static enrollment binding')
    gmeta=read(P/'retarget_geometry_v2/report.json')['geometry_metadata'];names={int(k):v['name'] for k,v in gmeta.items()};ids={v:k for k,v in names.items()}
    phone=ids['phone24_collision_convex'];table=ids['stage1_single_table'];pads=[ids['right_finger1_tip_sensor_col'],ids['right_finger2_tip_sensor_col']]
    asset=P/'asset_stage1/geometry_report.json';ipos=np.array(read(asset)['CPU_compile']['phone_asset.xml']['body_ipos_m'])
    jointmeta=P/'hand2_stage1/completed_v2/joint_actuator_interface.json';limits=np.array([x['hard_range_rad'] for x in read(jointmeta)['joints']])
    need(len(r['cases'])==2 and {x['seed'] for x in r['cases']}=={0},'exact2case single variant')
    cases=[];data=[];step_rows=[];contact_rows=[];totalN=totalC=0
    for case in r['cases']:
        seed=case['seed'];ss=S[seed];need(case['source_candidate_state']==ss['selected_state'],'selected-state binding')
        with np.load(checked(ss['selected_state']),allow_pickle=False) as z:q0=z['qpos27'].copy()
        with np.load(checked(case['trace']),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        need(np.array_equal(a['initial_qpos27'],q0) and np.array_equal(a['ctrl20'],np.array(ss['proposed_ctrl20'])),'LP-derived ctrl and own initialq0 bits')
        N=len(a['post_qpos']);dt=case['dt_s'];pre=np.arange(N);post=pre+1;totalN+=N
        q=a['post_qpos'];COM=q[:,20:23]+A.rotate(q[:,23:27],ipos);COM0=q0[20:23]+A.rotate(q0[23:27],ipos)
        angle=A.angle(q[:,23:27],q0[23:27]);drift=np.linalg.norm(COM-COM0,axis=1)
        need(close(COM0,a['initial_COM_world3'],1e-12) and close(COM0,case['initial_COM_world_m'],1e-12),'per-variantCOM0')
        need(close(COM,a['post_COM'],1e-12),'post COM independent quat-vector')
        rawacos=2*np.arccos(np.clip(abs(q[:,23:27]@q0[23:27]),0,1));need(close(rawacos,a['post_angle'],8*math.sqrt(np.finfo(float).eps)),'native-angle capture arithmetic')
        need(close(a['post_time'],post*dt,128*np.finfo(float).eps) and close(a['pre_time_from_integer_tick'],pre*dt,128*np.finfo(float).eps),'PRE/POST clocks')
        need(case['integration_steps']==case['finite_steps_saved']==N,'steps saved')
        need(all(np.isfinite(x).all() for x in a.values()),'finite raw arrays')
        cr=a['contact_rows'];totalC+=len(cr);need(cr.shape[1]==24 and np.array_equal(cr[:,:5],cr[:,:5].astype(np.int64)),'contact24 index ABI')
        hist=np.zeros(N,dtype=int);groups=[[] for _ in range(N)];pen=np.zeros(N);selfpen=np.zeros(N);pairs={}
        for k,row in enumerate(cr):
            tick,g1,g2,dim,efc=map(int,row[:5]);need(0<=tick<N and g1 in names and g2 in names,'raw contact bounds/names')
            ci=hist[tick];hist[tick]+=1;Fn=max(0.,row[18]);pen[tick]=max(pen[tick],-row[5])
            if phone in (g1,g2):
                other=g2 if g1==phone else g1;cat='thumb' if other==pads[0] else 'index' if other==pads[1] else 'table' if other==table else 'nonpad'
            else:cat='other'
            if cat=='other' and table not in (g1,g2):selfpen[tick]=max(selfpen[tick],-row[5])
            groups[tick].append((cat,Fn));key=(g1,g2);rec=pairs.setdefault(key,{'names':[names[g1],names[g2]],'category':cat,'rows':0,'max_normal_N':0.,'first_positive_PRE_tick':None})
            rec['rows']+=1;rec['max_normal_N']=max(rec['max_normal_N'],float(Fn))
            if Fn>0 and rec['first_positive_PRE_tick'] is None:rec['first_positive_PRE_tick']=tick
            contact_rows.append([dt,k,int(ci),*row.tolist(),names[g1],names[g2],cat])
        need(np.array_equal(hist,a['pre_ncon']),'all raw generated contacts retained')
        sums=np.array([[math.fsum(f for kind,f in rows if kind==cat) for cat in ['thumb','index','nonpad','table']] for rows in groups])
        need(close(sums[:,:2],a['pre_pad_normal_N']) and close(sums[:,2],a['pre_phone_nonpad_normal_N']) and close(sums[:,3],a['pre_table_normal_N']),'raw force sums')
        need(close(pen,a['pre_penetration_m']) and close(selfpen,a['pre_self_penetration_m']),'raw penetration sums')
        for tick,qp in zip(a['sample_pre_tick'],a['sample_pre_qpos']):need(np.array_equal(qp,q0 if tick==0 else q[tick-1]),'sample geometry PRE qpos phase')
        gate=c['gates'];tail_start=round((c['duration_s']-gate['tail_duration_s'])/dt);tailPOST=post>=tail_start;tailPRE=pre>=tail_start
        excursion=np.maximum(limits[:,0]-q[:,:20],0)+np.maximum(q[:,:20]-limits[:,1],0)
        metrics={'final_COM_drift_m':float(drift[-1]),'final_angle_rad':float(angle[-1]),
          'tail_linear_velocity_RMS_m_s':float(np.sqrt(np.mean(np.sum(a['post_qvel'][tailPOST,20:23]**2,axis=1)))) if any(tailPOST) else None,
          'tail_angular_velocity_RMS_rad_s':float(np.sqrt(np.mean(np.sum(a['post_qvel'][tailPOST,23:26]**2,axis=1)))) if any(tailPOST) else None,
          'tail_dual_pad_force_coverage':float(np.mean(np.all(sums[tailPRE,:2]>.1,axis=1))) if any(tailPRE) else None,
          'max_native_penetration_m':float(pen.max()),'max_self_penetration_m':float(selfpen.max()),'max_phone_nonpad_normal_force_N':float(sums[:,2].max()),'max_table_normal_force_N':float(sums[:,3].max()),'max_joint_limit_excursion_rad':float(excursion.max()),'warning_count':sum(case['warnings'])}
        for k,x in metrics.items():need(case['metrics'][k] is None if x is None else abs(case['metrics'][k]-x)<=1e-10,'metric '+k)
        loss=(drift>gate['abort_COM_drift_m'])|(angle>gate['abort_angle_rad'])
        need(not any(loss[:-1]) and loss[-1] and case['termination']=='ABORT_LOSS_OF_HOLD','first original abort crossing')
        statuses={'duration_complete':'FAIL'}
        for k,x in metrics.items():statuses[k]='NOT_RUN' if x is None else 'PASS_OBSERVED_PREFIX' if (x>=gate[k] if k=='tail_dual_pad_force_coverage' else x<=gate[k]) else 'FAIL'
        need({k:v=='PASS_OBSERVED_PREFIX' for k,v in statuses.items()}==case['gates'] and not case['nominal_holding_dynamics_pass'],'original allgates failed')
        rec={'seed':seed,'dt_s':dt,'steps':N,'actual_POST_duration_s':N*dt,'last_PRE_tick':N-1,'last_POST_tick':N,'metrics':metrics,'gate_statuses':statuses,
          'tail_POST_count':int(tailPOST.sum()),'tail_PRE_count':int(tailPRE.sum()),'abort_due_COM_drift':bool(drift[-1]>gate['abort_COM_drift_m']),'abort_due_angle':bool(angle[-1]>gate['abort_angle_rad']),
          'max_pad_normal_N':sums[:,:2].max(0).tolist(),'observed_prefix_dual_pad_above_0p1_fraction':float(np.mean(np.all(sums[:,:2]>.1,axis=1))),
          'contact_pairs':list(pairs.values()),'trace':case['trace'],'holding_pass':False}
        cases.append(rec);data.append((rec,COM,angle,sums,q0))
        for k in range(N):step_rows.append([dt,k,k+1,k*dt,(k+1)*dt,drift[k],math.degrees(angle[k]),*sums[k],pen[k]])
    need(totalN==r['counts']['integration_steps']==1153 and totalC==r['counts']['contact_force_reads']==420,'global steps/contactforce reads')
    need(r['counts']['MjData']==r['counts']['initial_forward']==2 and r['counts']['MJB_reads']==1,'per-case initialization counts')
    coarse,fine=sorted(data,key=lambda x:x[0]['dt_s'],reverse=True)
    # Final endpoint quaternion comparison is a normalized observer only.
    def endpoint_quat(case):
        with np.load(checked(case['trace']),allow_pickle=False) as z:return z['post_qpos'][-1,23:27].copy()
    qa=endpoint_quat(coarse[0]);qb=endpoint_quat(fine[0]);pair_angle=float(A.angle(qa,qb));pair_COM=float(np.linalg.norm(coarse[1][-1]-fine[1][-1]))
    pg={'both_duration_complete':False,'final_COM_difference_m':pair_COM<=c['dt_pair_gates']['final_COM_difference_m'],'final_orientation_difference_rad':pair_angle<=c['dt_pair_gates']['final_orientation_difference_rad']}
    producer_pair=r['dt_pair_consistency'][0];need(close(pair_COM,producer_pair['final_COM_difference_m']) and close(pair_angle,producer_pair['final_orientation_difference_rad']),'pair endpoint arithmetic')
    need(pg==producer_pair['gates'] and not producer_pair['dt_pair_consistency_pass'] and not r['all_original_cases_and_dt_pairs_pass'],'incomplete pair not qualified')
    ncommon=min(coarse[0]['steps'],fine[0]['steps']//2);ii=2*np.arange(ncommon)+1;dc=coarse[1][:ncommon]-fine[1][ii]
    comparison={'actual_endpoint_COM_difference_m':pair_COM,'actual_endpoint_orientation_difference_rad':pair_angle,'gate_statuses':pg,'dt_pair_consistency_pass':False,
       'comparison_status':'INCOMPLETE_DURATION_NOT_2S_CONSISTENCY','coarse_stop_s':coarse[0]['actual_POST_duration_s'],'fine_stop_s':fine[0]['actual_POST_duration_s'],
       'common_POST_count':ncommon,'common_POST_COM_vector_RMS_m':float(np.sqrt(np.mean(np.sum(dc*dc,axis=1))))}
    for fn,header,rows in [('step_metrics.csv',['dt_s','PRE_tick','POST_tick','PRE_time_s','POST_time_s','COM_drift_m','angle_deg','thumb_N','index_N','nonpad_N','table_N','penetration_m'],step_rows),
       ('named_contact_rows.csv',['dt_s','original_long_row','native_contact_index_within_tick','PRE_tick','geom1','geom2','dim','efc_address','distance_m','posx','posy','posz',*[f'frame{k}' for k in range(9)],*[f'wrench{k}' for k in range(6)],'geom1_name','geom2_name','category'],contact_rows)]:
        with (OUT/fn).open('x',newline='') as f:w=csv.writer(f);w.writerow(header);w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(2,2,figsize=(10.5,7),constrained_layout=True);handles=[]
    for (case,COM,angle,sums,q0),color in zip(data,['#1769aa','#df7c17']):
        dt=case['dt_s'];N=case['steps'];pre=np.arange(N)*dt;post=np.arange(1,N+1)*dt;COM0=q0[20:23]+A.rotate(q0[23:27],ipos)
        axes[0,0].plot(post,np.linalg.norm(COM-COM0,axis=1)*1000,color=color);axes[0,1].plot(post,np.degrees(angle),color=color)
        axes[1,0].plot(pre,sums[:,0],color=color);axes[1,0].plot(pre,sums[:,1],color=color,ls='--');axes[1,1].plot(pre,sums[:,2],color=color)
        handles.append(Line2D([0],[0],color=color,label=f"dt={dt:g} s"))
    for y in [5,50]:axes[0,0].axhline(y,color='gray',ls=':',lw=1)
    for y in [5,45]:axes[0,1].axhline(y,color='gray',ls=':',lw=1)
    axes[0,0].set_ylabel('COM drift [mm], POST');axes[0,1].set_ylabel('Orientation change [deg], POST')
    axes[1,0].set_ylabel('Pad normal sum [N], PRE');axes[1,1].set_ylabel('Phone–nonpad normal sum [N], PRE')
    axes[1,0].legend(handles=[Line2D([0],[0],color='black',label='thumb'),Line2D([0],[0],color='black',ls='--',label='index')],fontsize=8)
    for ax in axes.ravel():ax.set_xlabel('Actual saved time [s]');ax.set_xlim(left=0);ax.grid(alpha=.25)
    fig.legend(handles=handles,loc='outside lower center',ncols=2)
    fig.suptitle('v5 seed0 finite-mesh diagnostic: both abort near 0.12 s; 2.0 s target not reached\nActual prefixes only. PRE contact forces and POST states remain separate.',fontsize=11)
    fig.savefig(OUT/'holding_v3_actual_prefix.png',dpi=170);plt.close(fig)
    checks.update({'only_eligible_seed0_and_LP_ctrl_initial_state_binding':True,'per_variant_COM_angle_and_PRE_POST_phase':True,'all420_raw_contact_rows_and_force_aggregates':True,
       'first_original_abort_crossing_and_missing_tail_windows':True,'all_original_case_and_duration_dt_pair_gates':True,'source_count_and_saved_step_count_consistency':True,
       'no_model_LP_FK_neural_imports':not any(k in sys.modules for k in ['mujoco','torch','warp','mujoco_warp','scipy.optimize'])})
    out={'schema':'hand2_support_holding_v3_saved_independent:1','status':'PASS_SAVED_ARITHMETIC_BOTH_NOMINAL_HOLDING_CASES_FAILED',
       'source':pin(__file__),'quaternion_arithmetic_helper':pin(helper),'producer_report':pin(rf),'protocol':r['protocol'],'static_saved_review':pin(independent),'terminal':pin(ROOT/'cpu_nominal_v1_process_terminal.json'),
       'checks':checks,'passed':sum(checks.values()),'total':len(checks),'steps_saved':totalN,'generated_contact_rows':totalC,'cases':cases,'dt_pair':comparison,'holding_qualified':False,
       'scope':'No model load/LP/FK/step/NN. Saved current controls/ownq0 and all contact rows recomputed. Convex geometry, actualJ/metadata and no-applied-force/native-warning guards remain source/prior-review evidence; no new native proof. Both actual prefixes terminate before1.5–2s tail. No cross-version improvement claim or2s consistency from differing early-stop endpoints.',
       'outputs':{f.name:pin(f) for f in OUT.iterdir() if f.is_file() and f.name!='report.json'}}
    save(OUT/'report.json',out);print(json.dumps({'report':pin(OUT/'report.json'),'checks':f"{out['passed']}/{out['total']}",'steps':totalN,'contacts':totalC,'holding':False}))


if __name__=='__main__':main()
