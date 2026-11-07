"""Observed native22 contact labels and canonical replay, no simulator calls.

Signed world force is +on geom2/-on geom1; impulse axes are the frozen source
BODY axes and moments use each solver-PRE moving COM. Future-H labels require
complete finite no-warning traces. Partial-prefix evidence is kept separately.
This module never opens a checkpoint or loads legacy Python source.
"""
from types import SimpleNamespace
import hashlib,math
import numpy as np
from . import state_codec

PAIRS=(0,2,8,11,13,15,21,24,26,28,34,37,39,41,47,50,52,54,60,63,65,67)
PADS=(15,28,41)
SPLITS=('TRIPOD_TRAIN','TRIPOD_MONITOR')

FIELD_MAP=dict(x58='x58_F64',u20='u20_F64',y58='y58_F64',delta52='target_delta52_F64',memory83='memory83_F64',next_memory83='next_memory83_F64',
    obs222='obs222_FP32',next_obs222='next_obs222_FP32',action20='normalized_action20_F64',h='native_dt_s_F64',n='n_substeps_i64',executed_n='executed_substeps_i64',
    complete_H='complete_H_bool',WM_eligible='WM_eligible_bool',Actor_start_eligible='Actor_start_eligible_bool',actor_absorbing_done='actor_absorbing_done_bool',
    reference_timeout='reference_timeout_bool',acquisition_done='acquisition_done_bool',collection_truncated='acquisition_truncated_bool',
    row_id='row_id_i64',family_id='family_id_i64',episode_id='episode_id_i64',step_k='step_k_i64',impulse22x6='impulse_source_BODY_moving_COM22x6_F64',
    meanFn22='mean_positive_Fn22_N_F64',coverage22='Fn_gt_0p1_coverage22_F64',triple_coverage='simultaneous_triple_coverage_F64')

def need(v,msg):
    if not v:raise ValueError(msg)

def bits(a,b):
    a,b=np.asarray(a),np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def close(a,b,label):
    need(np.allclose(a,b,rtol=2e-12,atol=2e-12,equal_nan=False),label)

def require_array(a,key,shape,dtype=None):
    need(key in a,'Missing declared raw trace key '+key);v=np.asarray(a[key]);need(v.shape==shape,'Raw shape '+key)
    if dtype is not None:need(v.dtype==np.dtype(dtype),'Raw dtype '+key)
    return v

def receipt(vector,meta,tick,h):
    v=np.asarray(vector);need(v.dtype==np.float64 and v.shape==(300,),'Full integration300 F64')
    need(meta['signature']==16383 and meta['size']==300 and meta['sha256']==hashlib.sha256(v.tobytes()).hexdigest(),'Full receipt SHA/signature')
    if 'shape' in meta:need(meta['shape']==[300],'Receipt declared shape')
    if 'dtype' in meta:need(meta['dtype']==v.dtype.str,'Receipt declared dtype')
    need(meta['global_tick']==tick,'Receipt integer tick');close(meta['native_time_s'],tick*h,'Receipt native time')

def decode_window(contact_rows,pre_ticks,COM_pre,R_source,h,pair_ids,phone_gid=71):
    cr=np.asarray(contact_rows,dtype=np.float64);ticks=np.asarray(pre_ticks,dtype=np.int64);com=np.asarray(COM_pre);R=np.asarray(R_source)
    assert cr.ndim==2 and cr.shape[1]==24 and com.shape==(len(ticks),3) and R.shape==(3,3) and len(ticks)>0
    assert np.all(np.diff(ticks)==1) and np.isfinite(cr).all() and np.max(abs(R.T@R-np.eye(3)))<1e-10
    index={int(g):i for i,g in enumerate(pair_ids)};n=len(ticks);J=np.zeros((22,6));Fn=np.zeros((n,22));generated=np.zeros((n,22),dtype=np.int64);world_tick=np.zeros((n,6))
    mask=(cr[:,1]==phone_gid)|(cr[:,2]==phone_gid);c=cr[mask]
    if len(c):
        assert np.all((c[:,1]==phone_gid)^(c[:,2]==phone_gid))
        ti=np.searchsorted(ticks,c[:,0].astype(np.int64));assert np.all(ti<len(ticks)) and np.array_equal(ticks[ti],c[:,0])
        other=np.where(c[:,1]==phone_gid,c[:,2],c[:,1]).astype(np.int64)
        assert all(int(g) in index for g in other),'Every phone boundary contact needs a canonical pair'
        slot=np.array([index[int(g)] for g in other]);sign=np.where(c[:,2]==phone_gid,1.,-1.)
        frame=c[:,9:18].reshape(-1,3,3);force=c[:,18:24]
        F=sign[:,None]*np.einsum('nji,nj->ni',frame,force[:,:3])
        direct=sign[:,None]*np.einsum('nji,nj->ni',frame,force[:,3:])
        M=np.cross(c[:,6:9]-com[ti],F)+direct;W=np.c_[F,M]
        BODY=np.c_[F@R,M@R]
        np.add.at(J,slot,h*BODY);np.add.at(Fn,(ti,slot),np.maximum(force[:,0],0.))
        np.add.at(generated,(ti,slot),1);np.add.at(world_tick,ti,W)
    return dict(impulse_source_BODY_moving_COM22x6=J,mean_positive_Fn22_N=Fn.mean(0),min_positive_Fn22_N=Fn.min(0),
        max_positive_Fn22_N=Fn.max(0),last_PRE_positive_Fn22_N=Fn[-1],Fn_gt_0p1_coverage22=(Fn>.1).mean(0),
        generated_contact_coverage22=(generated>0).mean(0),world_tick_wrench6=world_tick,phone_contact_rows=len(c))

def load_math():
    """Ordinary-package shared codec/encoder, no path-selected implementation."""
    from .structured import encode_observed_numpy
    return dict(decode=SimpleNamespace(decode_window=decode_window),codec=state_codec,encode=encode_observed_numpy)

def observed_window(a,source,h,maths):
    ticks=np.asarray(a['pre_tick'],dtype=np.int64);post=np.asarray(a['post_state58']);cr=np.asarray(a['contact_rows24'])
    need(len(ticks)>0 and post.shape==(len(ticks),58) and cr.ndim==2 and cr.shape[1]==24,'Raw PRE/POST/contact ABI')
    finite=all(np.isfinite(v).all() for v in [source,post,cr])
    if not finite:return None
    COM_pre=np.vstack([source[None,40:43],post[:-1,40:43]])
    decoded=maths['decode'].decode_window(cr,ticks,COM_pre,source[43:52].reshape(3,3),h,PAIRS)
    # Keep simultaneous loading independent from pair marginal coverage.
    Fn=np.zeros((len(ticks),22));generated=np.zeros((len(ticks),22),dtype=np.int64)
    for row in cr:
        g1,g2=map(int,row[1:3])
        if 71 not in (g1,g2):continue
        gid=g2 if g1==71 else g1;need(gid in PAIRS,'Unknown phone-boundary slot')
        i=int(np.searchsorted(ticks,int(row[0])));need(i<len(ticks) and ticks[i]==row[0],'Contact tick membership')
        j=PAIRS.index(gid);Fn[i,j]+=max(0.,float(row[18]));generated[i,j]+=1
    close(Fn.mean(0),decoded['mean_positive_Fn22_N'],'Helper positive-Fn mean')
    close((Fn>.1).mean(0),decoded['Fn_gt_0p1_coverage22'],'Helper marginal coverage')
    decoded['simultaneous_triple_coverage']=float(np.mean(np.all(Fn[:,[PAIRS.index(g) for g in PADS]]>.1,axis=1)))
    decoded['per_tick_Fn22']=Fn;decoded['per_tick_generated22']=generated
    if 'pre_contact_wrench_about_COM_world6' in a:close(decoded['world_tick_wrench6'],a['pre_contact_wrench_about_COM_world6'],'Original Reader signed total wrench retained')
    return decoded

def canonical_row(a,identity,episode,task,physical,maths,*,pre_hold_alive):
    sid=int(identity['sample_id']);fid=int(identity['family_id']);ep=int(identity['episode_id']);k=int(identity['step_k'])
    split=identity['split'];need(0<=fid<16 and split==SPLITS[int(fid>=12)] and ep//2==fid,'Frozen16-family split/group')
    h=float(identity['dt_s']);n=int(identity['requested_substeps']);N=int(identity['executed_substeps'])
    need(n in [64,128] and h*n==.02 and 0<N<=n and 0<=k<20,'Actual H20ms source count')
    x=require_array(a,'source_state58',(58,),np.float64);u=require_array(a,'actual_control20',(20,),np.float64)
    observed=require_array(a,'observed_prefix_target58',(58,),np.float64)
    m=require_array(a,'memory_before83_F64',(83,),np.float64);mnext=require_array(a,'memory_after83_F64',(83,),np.float64)
    action=require_array(a,'action_normalized20_F64',(20,),np.float64);residual=require_array(a,'residual20_F64',(20,),np.float64)
    obs=require_array(a,'obs222_FP32',(222,),np.float32);nextobs=require_array(a,'next_obs222_FP32',(222,),np.float32)
    need(np.isfinite(np.r_[x,u,m,action,residual]).all() and np.all(abs(action)<=1),'Finite source/control/memory and normalized action')
    need(hashlib.sha256(u.tobytes()).hexdigest()==identity['actual_control20_raw_sha256'],'Actual control identity raw SHA')
    need(m[0]==k and m[82]==n and m[81]==h,'Memory clock matches native control interval')
    scale=np.array(task['action']['symmetric_residual_scale20_rad']);base=np.array(task['initial']['ctrl20_F64'])
    close(residual,.5*scale*action+.5*m[41:61],'Actual residual EMA, no hidden clamp')
    close(u,base+residual,'Actual executed absolute target')
    need(np.all(u>=task['action']['target_lower20']) and np.all(u<=task['action']['target_upper20']),'True target ranges')
    expected_next=np.r_[k+1,action,m[1:21],residual,u,h,n]
    close(mnext,expected_next,'Mechanical after-memory bookkeeping')
    # Known raw observation slices; exact geometry values remain source-backed, not recomputed by label export.
    need(bits(obs[:138],np.r_[x,m[1:81]].astype(np.float32)),'Observation current state/controller history')
    need(bits(obs[174:176],np.array([k/100,(100-k)/100],dtype=np.float32)),'Observation phase')
    need(bits(obs[220:222],np.array([h,n],dtype=np.float32)) and np.isin(obs[198:220],[0,1]).all(),'Observation native clock/geometry masks')
    need(np.all(obs[176:198][obs[198:220]==0]==0),'Unknown geometry neutral fill always paired with maskFalse')
    ticks=np.asarray(a['pre_tick']);need(ticks.shape==(N,) and np.array_equal(ticks,np.arange(k*n,k*n+N)),'PRE integer interval ticks')
    for name,vector,tick in [('source_receipt','source_integration_state',k*n),('source_after_action_receipt','source_after_action_integration_state',k*n),('target_receipt','target_integration_state',k*n+N)]:receipt(a[vector],identity[name],tick,h)
    before=np.asarray(a['source_integration_state']);after=np.asarray(a['source_after_action_integration_state'])
    offsets=[i for i in range(281) if bits(after[i:i+20],u) and bits(before[i:i+20],m[61:81]) and bits(before[:i],after[:i]) and bits(before[i+20:],after[i+20:])]
    need(len(offsets)==1,'Unique measured control-only receipt segment, no guessed layout')
    post=require_array(a,'post_state58',(N,58),np.float64);qpos=require_array(a,'post_qpos27',(N,27),np.float64)
    need(bits(post[-1],observed),'Observed prefix POST endpoint retained')
    complete=bool(a['complete_H_bool']) if 'complete_H_bool' in a else bool(N==n)
    need(complete==(N==n),'Complete H mask and actual ticks')
    warnings=np.asarray(identity['warning_numbers']);no_warning=not bool(np.any(warnings))
    finite=all(np.isfinite(z).all() for z in [post,qpos,a['post_qvel26'],a['contact_rows24']])
    full_known=bool(complete and finite and no_warning)
    target=require_array(a,'target_at_H_state58',(58,),np.float64);delta=require_array(a,'physical_world_delta52',(52,),np.float64)
    if complete:need(bits(target,observed),'Complete raw H target is observed endpoint')
    else:need(np.isnan(target).all() and np.isnan(delta).all(),'Partial cannot invent H targets')
    if full_known:close(delta,maths['codec'].delta52(x,target),'WORLD-left observed delta52')
    # Native per-tick descriptor is not substituted for the collector's POST-H MDP terminal.
    initial=np.array(task['initial']['native_PRE_state58_F64']);q0=np.array(task['initial']['qpos27_F64'])
    drift=np.linalg.norm(post[:,40:43]-initial[40:43],axis=1)
    R0=initial[43:52].reshape(3,3)
    relative=post[:,43:52].reshape(-1,3,3)@R0.T
    vee=np.stack([relative[:,2,1]-relative[:,1,2],relative[:,0,2]-relative[:,2,0],relative[:,1,0]-relative[:,0,1]],axis=1)/2
    angle=np.arctan2(np.linalg.norm(vee,axis=1),(np.trace(relative,axis1=1,axis2=2)-1)/2)
    tick_loss=(drift>.05)|(angle>math.pi/4)
    boundary_loss=bool(tick_loss[-1]) if complete and finite else False
    if 'task_done' in a:need(bool(a['task_done'])==boundary_loss,'Collector task POST-H stop matches task source')
    reader_done=bool(a['reader_done']) if 'reader_done' in a else identity['termination']!='COMPLETE_INTERVAL'
    reference_timeout=bool(k+1>=100 and complete)
    collect_truncated=bool(complete and k+1==20 and not boundary_loss and not reader_done)
    actor_done=bool(boundary_loss or reader_done or reference_timeout or not complete)
    decoded=observed_window(a,x,h,maths)
    J=np.full((22,6),np.nan);mean=np.full(22,np.nan);cov=mean.copy();generated=mean.copy();triple=float('nan');heads=np.full(178,np.nan)
    linear=np.full(3,np.nan);angular=linear.copy()
    if full_known:
        need(decoded is not None,'Full known contact operands')
        J=decoded['impulse_source_BODY_moving_COM22x6'];mean=decoded['mean_positive_Fn22_N'];cov=decoded['Fn_gt_0p1_coverage22'];generated=decoded['generated_contact_coverage22'];triple=decoded['simultaneous_triple_coverage']
        d=maths['encode'](x[None],target[None],J[None],np.array([h*n]),np.array(physical['gravity_world_m_s2']))
        heads=np.r_[d['joint_delta40'][0],d['COM_ballistic_residual3_m'][0],d['world_rotation_delta3_rad'][0],J.ravel()]
        R=x[43:52].reshape(3,3);R1=target[43:52].reshape(3,3);I=np.array(physical['I_body_3x3']);mass=physical['mass_kg'];g=np.array(physical['gravity_world_m_s2']);total=J.sum(0)
        linear=mass*(target[52:55]-x[52:55])-mass*g*(h*n)-R@total[:3]
        angular=R1@I@R1.T@target[55:58]-R@I@R.T@x[55:58]-R@total[3:]
    prefix_J=np.full((22,6),np.nan) if decoded is None else decoded['impulse_source_BODY_moving_COM22x6']
    output=dict(row_id_i64=sid,family_id_i64=fid,episode_id_i64=ep,step_k_i64=k,native_dt_s_F64=h,n_substeps_i64=n,executed_substeps_i64=N,
        x58_F64=x,u20_F64=u,y58_F64=target,target_delta52_F64=delta,observed_prefix_target58_F64=observed,
        memory83_F64=m,next_memory83_F64=mnext if complete and finite else np.full(83,np.nan),observed_bookkeeping_memory83_F64=mnext,
        obs222_FP32=obs,next_obs222_FP32=nextobs if complete and finite else np.full(222,np.nan,dtype=np.float32),normalized_action20_F64=action,residual20_F64=residual,
        recorded_objective_scalar_F64=float(a['reward_scalar_F64']),complete_H_bool=complete,finite_bool=finite,no_warning_bool=no_warning,
        physical_labels_known_bool=full_known,WM_eligible_bool=full_known,Actor_start_eligible_bool=bool(full_known and pre_hold_alive and np.isfinite(a['reward_scalar_F64']).all()),
        pre_hold_alive_bool=bool(pre_hold_alive),task_hold_failed_boundary_POST_bool=boundary_loss,task_hold_failed_any_native_POST_bool=bool(np.any(tick_loss)),
        actor_absorbing_done_bool=actor_done,reader_done_bool=reader_done,acquisition_truncated_bool=collect_truncated,reference_timeout_bool=reference_timeout,
        next_memory_H_known_bool=complete and finite,next_obs_H_known_bool=complete and finite,
        task_boundary_failure_known_bool=complete and finite,any_native_task_loss_known_bool=finite,
        impulse_source_BODY_moving_COM22x6_F64=J,mean_positive_Fn22_N_F64=mean,Fn_gt_0p1_coverage22_F64=cov,generated_contact_coverage22_F64=generated,
        simultaneous_triple_coverage_F64=triple,target_physical_heads178_F64=heads,
        linear_momentum_residual_world_Ns_F64=linear,angular_momentum_residual_world_Nms_F64=angular,
        observed_prefix_impulse_source_BODY_moving_COM22x6_F64=prefix_J,observed_prefix_duration_s_F64=h*N,observed_prefix_contact_labels_known_bool=decoded is not None)
    alias=(identity['source_after_action_receipt']['sha256'],hashlib.sha256(u.tobytes()).hexdigest(),h,n,identity['source_after_action_receipt']['signature'])
    note=dict(sample_id=sid,split=split,source_identity=identity,source_control_segment_offset=offsets[0],pre_hold_alive=bool(pre_hold_alive),
        task_boundary_terminal=boundary_loss,any_native_tick_taskloss_descriptor=bool(np.any(tick_loss)),complete_H=complete,physical_labels_known=full_known,
        exclusion_reasons=[] if full_known else [s for s,c in [('partial_H',not complete),('nonfinite',not finite),('native_warning',not no_warning)] if c],
        observed_prefix_is_not_full_H=not complete,actual_contact_roles=[15,28,41],future_labels_are_not_current_inputs=True)
    return output,note,alias

def analytic_tripod_semantic_fixture(maths):
    """Prepared only: changing marginal co-occurrence must not fake joint cover."""
    ticks=np.arange(3);com=np.zeros((3,3));rows=[]
    for t,active in enumerate([(15,28),(28,41),(15,41)]):
        for gid in active:rows.append(np.r_[t,gid,71,3,0,0,[0,0,0],np.eye(3).ravel(),[1,0,0,0,0,0]])
    source=np.r_[np.zeros(43),np.eye(3).ravel(),np.zeros(6)]
    a=dict(pre_tick=ticks,post_state58=np.tile(source,(3,1)),contact_rows24=np.array(rows))
    d=observed_window(a,source,.02/3,maths)
    close(d['Fn_gt_0p1_coverage22'][[PAIRS.index(g) for g in PADS]],[2/3]*3,'Marginals')
    need(d['simultaneous_triple_coverage']==0,'Three positive marginals cannot fabricate simultaneous support')
    return dict(schema='tripod_label_semantic_fixture:1',marginal_coverage3=[2/3]*3,simultaneous_triple=0,new_native_NN_FK_calls=0)

def canonical_online_row(raw,identity,episode,bundle,physical,*,pre_hold_alive,maths=None):
    """Only identities are delegated; physical/window/memory math is unchanged.

    Local family0..3 and episode2*f+dt preserve the audited canonical adapter's
    family assertions. New online global IDs and train-only role are explicit
    separate fields; this does not import the old TRAIN/MONITOR partition.
    """
    need(identity['data_role']=='ONLINE_TRAIN_ONLY_NEW_STAGE' and 0<=identity['family_id']<4,'Own online family namespace')
    delegated=dict(identity,split='TRIPOD_TRAIN')
    result,note,alias=canonical_row(raw,delegated,episode,bundle,physical,load_math() if maths is None else maths,pre_hold_alive=pre_hold_alive)
    note['split']='ONLINE_TRAIN';note['source_identity']=identity
    result['acquisition_done_bool']=bool(identity['done'])
    return result,note,alias

def stack_rows(rows):
    need(rows,'Cannot invent shape-qualified arrays with no observed rows')
    return {k:np.asarray([r[k] for r in rows],dtype=np.bool_ if k.endswith('_bool') else np.int64 if k.endswith('_i64') else np.float32 if k.endswith('_FP32') else np.float64) for k in rows[0]}
