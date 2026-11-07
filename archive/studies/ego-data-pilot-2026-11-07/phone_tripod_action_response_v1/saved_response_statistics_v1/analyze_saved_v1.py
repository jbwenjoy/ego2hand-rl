"""All320 held-out finite-action responses, saved NumPy only. No native/NN."""
from pathlib import Path
import hashlib,importlib.util,json,math,sys,traceback
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent;P=ROOT.parent

def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):
    p=Path(v['path']);assert p.is_absolute() and pin(p)==v;return p
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def load_scalar(v):
    s=importlib.util.spec_from_file_location('tripod_response_frozen_scalar',checked(v));m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def describe(v,unit):
    a=np.asarray(v,dtype=np.float64).reshape(len(v),-1);assert np.isfinite(a).all()
    n=np.linalg.norm(a,axis=1);nz=np.any(a!=0,axis=1)
    return dict(rows=len(a),unit=unit,exact_F64_zero_rows=int((~nz).sum()),nonzero_rows=int(nz.sum()),
        vector_RMS=float(np.sqrt(np.mean(n*n))) if len(n) else None,
        nonzero_vector_RMS=float(np.sqrt(np.mean(n[nz]*n[nz]))) if nz.any() else None,
        median_vector_norm=float(np.median(n)) if len(n) else None,max_vector_norm=float(n.max()) if len(n) else None)

def momentum(a,h,physical,pair_ids):
    """Scalar/fsum world row decoding. Actual contact pos, moving PRE COM."""
    src=a['source_state58_F64'];post=a['post_state58'];ticks=a['pre_tick'];com=np.vstack([src[None,40:43],post[:-1,40:43]])
    lookup={int(t):i for i,t in enumerate(ticks)};terms=[[[] for _ in range(6)] for _ in ticks];contacts=0
    for c in a['contact_rows24']:
        g0,g1=int(c[1]),int(c[2])
        if (g0==71)==(g1==71):continue
        other=g0 if g1==71 else g1;assert other in pair_ids
        k=lookup[int(c[0])];assert ticks[k]==c[0]
        sign=1. if g1==71 else -1.;frame=c[9:18].reshape(3,3)
        F=np.array([sign*math.fsum(float(frame[j,i])*float(c[18+j]) for j in range(3)) for i in range(3)])
        D=np.array([sign*math.fsum(float(frame[j,i])*float(c[21+j]) for j in range(3)) for i in range(3)])
        r=c[6:9]-com[k];M=np.cross(r,F)+D
        for j,v in enumerate(np.r_[F,M]):terms[k][j].append(float(v))
        contacts+=1
    tick_w=np.array([[math.fsum(v) for v in row] for row in terms])
    impulse=np.array([math.fsum(float(h)*float(v) for v in tick_w[:,j]) for j in range(6)])
    y=post[-1];mass=physical['mass_kg'];g=np.array(physical['gravity_world_m_s2']);I=np.array(physical['I_body_3x3']);T=h*len(ticks)
    R0=src[43:52].reshape(3,3);R1=y[43:52].reshape(3,3)
    dP=mass*(y[52:55]-src[52:55]);dL=R1@I@R1.T@y[55:58]-R0@I@R0.T@src[55:58]
    return dict(contact_rows=contacts,world_impulse6=impulse,linear_residual_Ns=dP-mass*g*T-impulse[:3],angular_residual_Nms=dL-impulse[3:],
        stored_tick_wrench_max_abs_diff=float(np.max(abs(tick_w-a['pre_contact_wrench_about_COM_world6']))))

def run():
    report_pin=pin(ROOT/'native_completed_v1/report.json');r=read(report_pin['path']);plan=read(checked(r['selection']));p=read(checked(r['protocol']));records=read(checked(r['branches']))
    assert r['executed_branches']==r['labels_known_branches']==328 and r['partial_branches']==0 and r['counts']['integration_steps']==31488
    assert [v['branch_id'] for v in records]==list(range(328))
    scalar=load_scalar(p['sources']['scalar_codec']);physical=read(checked(p['inputs']['physical_parameters']));task=read(checked(r['task_bundle']));bounds=np.array(task['actuator_ctrlrange20x2'])
    selected={v['context_id']:v for v in plan['states']};pair_ids=r['canonical_pair_geom_ids'];baseline={};rows=[];data={};mom=[];read_count=0;phone_contacts=0
    wanted=['source_state58_F64','actual_control20_F64','source_integration_F64','target_H_state58_F64','target_delta52_F64','target_physical_heads178_F64',
        'impulse_source_BODY_moving_COM22x6_F64','mean_positive_Fn22_N_F64','Fn_gt_0p1_coverage22_F64','generated_contact_coverage22_F64','simultaneous_triple_coverage_F64']
    max_wrench_diff=0.
    for rec in records:
        with np.load(checked(rec['output']),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        read_count+=1;assert rec['status']=='EXECUTED' and rec['labels_known'] and bool(a['physical_labels_known_bool'])
        assert len(a['pre_tick'])==rec['n_substeps'] and rec['returned_substeps']==rec['n_substeps']
        m=momentum(a,rec['dt_s'],physical,pair_ids);phone_contacts+=m.pop('contact_rows');max_wrench_diff=max(max_wrench_diff,m.pop('stored_tick_wrench_max_abs_diff'));mom.append(m)
        if rec['kind']=='baseline':baseline[rec['context_id']]={k:a[k] for k in wanted};continue
        b=baseline[rec['context_id']];assert bits(a['source_state58_F64'],b['source_state58_F64']) and bits(a['source_integration_F64'],b['source_integration_F64'])
        j=rec['axis'];u=a['actual_control20_F64'];ub=b['actual_control20_F64'];assert bits(np.delete(u,j),np.delete(ub,j))
        du=float(u[j]-ub[j]);assert du==rec['actual_du_F64'] and np.isfinite(du) and du!=0 and np.all(u>=bounds[:,0]) and np.all(u<=bounds[:,1])
        par=rec['perturbation'];assert par==selected[rec['context_id']]['perturbations'][j*2+(rec['sign']==1)]
        required=np.array(par['required_normalized_action20_F64']);reach=bool(np.all(abs(required)<=1));assert reach==par['Actor_EMA_reachable_descriptor']
        y,yb=a['target_H_state58_F64'],b['target_H_state58_F64'];rotation=scalar.log_world(y[43:52].reshape(3,3)@yb[43:52].reshape(3,3).T)
        values=dict(delta52_response=a['target_delta52_F64']-b['target_delta52_F64'],physical_head178_response=a['target_physical_heads178_F64']-b['target_physical_heads178_F64'],
            endpoint_rotation_world_log3_rad=rotation,COM_endpoint_response_m=y[40:43]-yb[40:43],
            J22x6_response=a['impulse_source_BODY_moving_COM22x6_F64']-b['impulse_source_BODY_moving_COM22x6_F64'],
            meanFn22_response_N=a['mean_positive_Fn22_N_F64']-b['mean_positive_Fn22_N_F64'],coverage22_response=a['Fn_gt_0p1_coverage22_F64']-b['Fn_gt_0p1_coverage22_F64'],
            generated_coverage22_response=a['generated_contact_coverage22_F64']-b['generated_contact_coverage22_F64'],
            triple_coverage_response=np.asarray(a['simultaneous_triple_coverage_F64']-b['simultaneous_triple_coverage_F64']),
            source_state58=a['source_state58_F64'],side_y58=y,baseline_y58=yb,side_u20=u,baseline_u20=ub,required_normalized_action20=required)
        for k,v in values.items():data.setdefault(k,[]).append(v)
        rows.append(dict(branch_id=rec['branch_id'],baseline_branch_id=rec['baseline_branch_id'],context_id=rec['context_id'],family_id=rec['family_id'],step_k=rec['step_k'],native_dt_s=rec['dt_s'],axis=j,sign=rec['sign'],actual_du_rad=du,
            nominal_full_step=bool(abs(abs(du)-.01)<=8*np.finfo(float).eps*(abs(u[j])+abs(ub[j])+.01)),limited_by_bound=par['limited_by_bound'],
            actuator_min_margin_rad=float(np.minimum(u-bounds[:,0],bounds[:,1]-u).min()),Actor_EMA_reachable_descriptor=reach,required_action_max_abs=float(abs(required).max()),
            COM_response_norm_m=float(np.linalg.norm(values['COM_endpoint_response_m'])),endpoint_rotation_geodesic_rad=float(np.linalg.norm(rotation))))
    assert len(rows)==320 and len(baseline)==8
    data={k:np.stack(v) for k,v in data.items()};du=np.array([v['actual_du_rad'] for v in rows]);axis=np.array([v['axis'] for v in rows]);dt=np.array([v['native_dt_s'] for v in rows]);step=np.array([v['step_k'] for v in rows]);context=np.array([v['context_id'] for v in rows])
    for k,v in list(data.items()):
        if k.endswith('response') or k.endswith('_response_N') or k in ['COM_endpoint_response_m','endpoint_rotation_world_log3_rad']:
            data[k+'_actualdu_secant']=v/du.reshape((-1,)+(1,)*(v.ndim-1))
    masks={'ALL':np.ones(320,dtype=bool)}
    for h in sorted(set(dt)):masks['dt_'+str(h)]=dt==h
    for j in range(20):masks['axis_'+str(j)]=axis==j
    for k in [0,3]:masks['step_'+str(k)]=step==k
    for c in range(8):masks['context_'+str(c)]=context==c
    for h in sorted(set(dt)):
        for k in [0,3]:
            for j in range(20):masks[f'dt_{h}__step_{k}__axis_{j}']=(dt==h)&(step==k)&(axis==j)
    blocks=[('q_rad',slice(0,20),'rad'),('qd_rad_s',slice(20,40),'rad/s'),('COM_m',slice(40,43),'m'),('difference_of_source_worldleft_rotation_logs_rad',slice(43,46),'rad'),('vCOM_m_s',slice(46,49),'m/s'),('omega_world_rad_s',slice(49,52),'rad/s')]
    stats=[];pairs=[]
    for name,mask in masks.items():
        stats.append(dict(cohort=name,rows=int(mask.sum()),state_delta52_response={n:describe(data['delta52_response'][mask,s],u) for n,s,u in blocks},
            endpoint_rotation_world_log3=describe(data['endpoint_rotation_world_log3_rad'][mask],'rad'),COM_actualdu_secant=describe(data['COM_endpoint_response_m_actualdu_secant'][mask],'m/rad'),
            endpoint_rotation_actualdu_secant=describe(data['endpoint_rotation_world_log3_rad_actualdu_secant'][mask],'rad/rad'),
            J22_force=describe(data['J22x6_response'][mask,:,:3],'Ns'),J22_moment=describe(data['J22x6_response'][mask,:,3:],'Nm*s'),meanFn22=describe(data['meanFn22_response_N'][mask],'N'),
            coverage22=describe(data['coverage22_response'][mask],'dimensionless fraction'),triple_coverage=describe(data['triple_coverage_response'][mask],'dimensionless fraction')))
        if name=='ALL' or name.startswith('dt_') and '__' not in name or name.startswith('step_'):
            for j,gid in enumerate(pair_ids):pairs.append(dict(cohort=name,pair_index=j,other_geom_id=gid,Jforce=describe(data['J22x6_response'][mask,j,:3],'Ns'),Jmoment=describe(data['J22x6_response'][mask,j,3:],'Nm*s'),meanFn=describe(data['meanFn22_response_N'][mask,j],'N'),coverage=describe(data['coverage22_response'][mask,j],'dimensionless fraction')))
    np.savez_compressed(H/'all320_actualdu_responses.npz',**data,branch_id=np.array([v['branch_id'] for v in rows]),baseline_branch_id=np.array([v['baseline_branch_id'] for v in rows]),context_id=context,axis=axis,native_dt_s=dt,step_k=step,actual_du_F64=du,Actor_EMA_reachable=np.array([v['Actor_EMA_reachable_descriptor'] for v in rows]),canonical_pair_geom_ids=np.array(pair_ids))
    np.savez_compressed(H/'all328_momentum_readback.npz',branch_id=np.arange(328),native_dt_s=np.array([v['dt_s'] for v in records]),**{k:np.stack([m[k] for m in mom]) for k in mom[0]})
    write(H/'side_identities_and_bounds.json',rows);write(H/'response_statistics.json',stats);write(H/'per_pair_statistics.json',pairs)
    msum={k:describe(np.stack([m[k] for m in mom]),'Ns' if 'linear' in k else 'Nm*s') for k in ['linear_residual_Ns','angular_residual_Nms']}
    bydt={}
    for h in sorted(set(dt)):
        m=np.array([v['dt_s']==h for v in records]);bydt[str(h)]={k:describe(np.stack([v[k] for v in mom])[m],'Ns' if 'linear' in k else 'Nm*s') for k in ['linear_residual_Ns','angular_residual_Nms']}
    outputs={f.name:pin(f) for f in H.iterdir() if f.is_file() and f.name!='report.json'}
    result=dict(schema='phone_tripod_action_response_saved_statistics:1',status='COMPLETE_ALL320_KNOWN_MON_FINITE_ACTION_RESPONSES_NO_NEW_SAMPLING',producer=report_pin,source=pin(__file__),protocol=r['protocol'],selection=r['selection'],branches=r['branches'],physical_parameters=p['inputs']['physical_parameters'],scalar_source=p['sources']['scalar_codec'],outputs=outputs,
        side_rows=320,contexts=8,canonical_pair_geom_ids=pair_ids,cohort_statistics=len(stats),per_pair_statistics=len(pairs),ALL=stats[0],
        action=dict(actuator_legal_rows=320,Actor_EMA_inside=sum(v['Actor_EMA_reachable_descriptor'] for v in rows),Actor_EMA_outside=sum(not v['Actor_EMA_reachable_descriptor'] for v in rows),absolute_du_min_rad=float(abs(du).min()),absolute_du_max_rad=float(abs(du).max()),full_nominal_rows=sum(v['nominal_full_step'] for v in rows),bound_limited_rows=sum(v['limited_by_bound'] for v in rows),minimum_actuator_margin_rad=min(v['actuator_min_margin_rad'] for v in rows),no_reachability_filter=True),
        momentum_all328=msum,momentum_by_dt=bydt,raw_world_wrench_vs_saved_max_abs_difference=max_wrench_diff,
        counts=dict(branch_NPZ_reads=read_count,additional_preliminary_branch_schema_NPZ_reads=1,phone_contact_rows_decoded=phone_contacts,actual_native_steps=0,NN_calls=0),
        interpretation=['All320 sides are retained with their actual signed du. Absolute-control validity and current Actor residual/EMA reachability are distinct descriptors; none are selected/dropped.',
        'Every response uses the same full integration source as its own baseline. These eight known MON contexts at two times/dts do not form independent episodes or infinitesimal/20-dimensional Jacobian qualification.',
        'delta52 orientation response is a difference of source-relative world-left Log vectors. The separately saved Log(R_side R_baseline^T) is the direct endpoint world-left rotation/geodesic; they need not coincide.',
        'Impulse labels are in frozen source BODY axes with torque about each tick moving COM; the independent world contact sum uses all signed forces/direct moments and no positive-Fn filter. Fn/coverage/triple are separate positive-normal loading descriptors.',
        'Momentum residuals are saved discrete finite-horizon descriptors using nominal mass/inertia; no invented pass threshold, correction to zero, complete soft-contact certificate or externally added gyroscopic torque. No held-out response is a training input or checkpoint selection rule.'])
    write(H/'report.json',result);return result

if __name__=='__main__':
    try:r=run();print(json.dumps(dict(report=pin(H/'report.json'),ALL=r['ALL'],action=r['action'],momentum=r['momentum_all328'])))
    except BaseException as e:
        write(H/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),new_native_NN_calls=0,automatic_retry=False));raise
