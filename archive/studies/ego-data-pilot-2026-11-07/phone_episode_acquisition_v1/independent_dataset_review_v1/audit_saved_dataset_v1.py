"""Independent saved arithmetic only. Never import a simulator, producer codec or NN."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,math,sys
import numpy as np

ROOT=Path(__file__).resolve().parent
ATOL=RTOL=1e-10
ARRAY_KEYS={'x58_F64','u20_F64','y58_F64','target_delta52_F64','native_dt_s_F64','n_substeps_i64',
 'executed_substeps_i64','complete_H_bool','training_eligible_bool','row_id_i64','family_id_i64',
 'episode_id_i64','interval_index_i64','contact_summary_F64'}
ERRORS=defaultdict(float)
def require(ok,text):
    if not ok:raise ValueError(text)
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
    p=Path(r['path']);require(p.is_absolute() and pin(p)==r,'Changed pinned input '+str(p));return p
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def load(r):
    with np.load(checked(r),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bits(a,b,text):
    a=np.asarray(a);b=np.asarray(b);require(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),text)
def close(a,b,label):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    require(a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)),label+' shape/NaN')
    require(not np.isinf(a).any() and not np.isinf(b).any(),label+' infinity')
    finite=np.isfinite(a)&np.isfinite(b)
    if np.any(finite):
        e=float(np.max(abs(a[finite]-b[finite])));ERRORS[label]=max(ERRORS[label],e)
        require(np.all(abs(a[finite]-b[finite])<=ATOL+RTOL*abs(b[finite])),label+' arithmetic mismatch '+str(e))
def rotate(q,v):
    q=np.asarray(q);v=np.asarray(v);w=q[...,0:1];u=q[...,1:4]
    return (w*w-np.sum(u*u,axis=-1,keepdims=True))*v+2*u*np.sum(u*v,axis=-1,keepdims=True)+2*w*np.cross(u,v)
def rotation(q):return np.stack([rotate(q,np.eye(3)[j]) for j in range(3)],axis=-1)
def state58(q,v,ipos):
    q=np.asarray(q);v=np.asarray(v);R=rotation(q[...,23:27]);off=rotate(q[...,23:27],ipos)
    omega=rotate(q[...,23:27],v[...,23:26])
    return np.concatenate([q[...,:20],v[...,:20],q[...,20:23]+off,R.reshape(q.shape[:-1]+(9,)),
        v[...,20:23]+np.cross(omega,off),omega],axis=-1)
def log3(R):
    # Independent largest-quaternion-component observer, not producer atan2(skew,trace).
    R=np.asarray(R);require(np.max(abs(R.T@R-np.eye(3)))<1e-9,'SO3 reference domain')
    candidates=np.array([1+np.trace(R),1+2*R[0,0]-np.trace(R),1+2*R[1,1]-np.trace(R),1+2*R[2,2]-np.trace(R)])
    j=int(np.argmax(candidates));z=2*math.sqrt(max(0.,candidates[j]));require(z>0,'Quaternion reference branch')
    if j==0:q=np.array([z/4,(R[2,1]-R[1,2])/z,(R[0,2]-R[2,0])/z,(R[1,0]-R[0,1])/z])
    elif j==1:q=np.array([(R[2,1]-R[1,2])/z,z/4,(R[0,1]+R[1,0])/z,(R[0,2]+R[2,0])/z])
    elif j==2:q=np.array([(R[0,2]-R[2,0])/z,(R[0,1]+R[1,0])/z,z/4,(R[1,2]+R[2,1])/z])
    else:q=np.array([(R[1,0]-R[0,1])/z,(R[0,2]+R[2,0])/z,(R[1,2]+R[2,1])/z,z/4])
    q/=np.linalg.norm(q)
    if q[0]<0:q=-q
    s=float(np.linalg.norm(q[1:]));theta=2*math.atan2(s,float(q[0]))
    require(theta<math.pi-1e-6,'Unresolved principal Log halfturn')
    return np.zeros(3) if s==0 else theta/s*q[1:]
def delta52(x,y):return np.r_[y[:43]-x[:43],log3(y[43:52].reshape(3,3)@x[43:52].reshape(3,3).T),y[52:]-x[52:]]
def exp3(v):
    t=float(np.linalg.norm(v));q=np.r_[math.cos(t/2),np.zeros(3) if t==0 else math.sin(t/2)/t*v]
    return rotation(q)
def receipt(meta,a,tick,h):
    require(a.dtype==np.float64 and a.ndim==1 and list(a.shape)==meta['shape'] and len(a)==meta['size'],'Full integration vector ABI')
    require(digest(a)==meta['sha256'] and meta['dtype']==a.dtype.str,'Full integration receipt raw SHA')
    require(meta['global_tick']==tick and meta['signature']==16383,'Full integration state spec and tick')
    close(meta['native_time_s'],tick*h,'receipt_clock')
def contact_math(c,ticks,pre_COM):
    n=len(ticks);pad=np.zeros((n,2));other=np.zeros(n);table=np.zeros(n);pen=np.zeros(n);selfpen=np.zeros(n);wrench=np.zeros((n,6))
    bag=defaultdict(list);lookup={int(t):i for i,t in enumerate(ticks)}
    require(c.ndim==2 and c.shape[1]==24,'Raw contact24 ABI')
    for row in c:
        t,g1,g2=map(int,row[:3]);require(t in lookup and np.array_equal(row[:5],np.rint(row[:5])),'Original contact tick/geom integer identity')
        i=lookup[t];fn=max(0.,float(row[18]));pen[i]=max(pen[i],-float(row[5]))
        if 71 in (g1,g2):
            g=g2 if g1==71 else g1
            category=('pad', [15,28].index(g)) if g in [15,28] else ('table',0) if g==0 else ('other',0)
            bag[(i,*category)].append(fn)
            sign=1 if g2==71 else -1;F=[];T=[]
            for j in range(3):
                F.append(sign*math.fsum(float(row[9+3*k+j])*float(row[18+k]) for k in range(3)))
                T.append(sign*math.fsum(float(row[9+3*k+j])*float(row[21+k]) for k in range(3)))
            lever=np.cross(row[6:9]-pre_COM[i],F)
            for j in range(3):bag[(i,'force',j)].append(F[j]);bag[(i,'torque',j)].append(float(lever[j])+T[j])
        elif 0 not in (g1,g2):selfpen[i]=max(selfpen[i],-float(row[5]))
    for i in range(n):
        for j in range(2):pad[i,j]=math.fsum(bag[(i,'pad',j)])
        other[i]=math.fsum(bag[(i,'other',0)]);table[i]=math.fsum(bag[(i,'table',0)])
        for j in range(3):wrench[i,j]=math.fsum(bag[(i,'force',j)]);wrench[i,j+3]=math.fsum(bag[(i,'torque',j)])
    return pad,other,table,pen,selfpen,wrench
def fixture():
    q=np.r_[np.zeros(20),[4.,5.,6.],[math.sqrt(.5),0.,0.,math.sqrt(.5)]];v=np.r_[np.zeros(20),[1.,2.,3.],[0.,0.,2.]]
    x=state58(q,v,np.array([.1,0,0]));close(x[40:43],[4,5.1,6],'fixture_COM');close(x[52:55],[.8,2,3],'fixture_vCOM')
    x2=x.copy();x2[43:52]=(rotation(np.array([math.cos(.1),math.sin(.1),0,0]))@x[43:52].reshape(3,3)).ravel()
    d=delta52(x,x2);close(d[43:46],[.2,0,0],'fixture_noncommuting_world_Log');close(exp3(d[43:46])@x[43:52].reshape(3,3),x2[43:52].reshape(3,3),'fixture_Exp_roundtrip')
    return dict(schema='phone_episode_independent_arithmetic_fixture:1',passed=True,cases=1,
        scope='Offset COM velocity and noncommuting world-left Log/Exp only; no dataset/model/physics.',max_errors=dict(ERRORS))

def audit(report_path,out):
    r=read(report_path);require(r['schema']=='phone_episode_acquisition_result:1' and r['status']=='COMPLETE_GROUPED_NATIVE_ACQUISITION','Complete dataset result')
    p=read(checked(r['protocol']));require(p['schema']=='phone_episode_acquisition_protocol:1','Dataset acquisition protocol')
    for v in [*p['inputs'].values(),*p['source_pins'].values()]:checked(v)
    require(p['source_pins']['collector']['sha256']=='48f70d650f8f0f197120bc864f25a7c850b37ceae2fd068eea7e91fef40da17d','Reviewed collector v2 source')
    design=read(checked(r['design']));plan=read(checked(r['family_plan']))
    require(r['design']==p['inputs']['design'] and r['family_plan']==design['family_plan']==p['inputs']['family_plan'],'Frozen design/plan join')
    require(r['design']['sha256']=='d3f4d8e0a4c44844fc8d51fb261be32f893ced20aa97a8e6404d1b08f7c229c4' and r['family_plan']['sha256']=='7336ae9243290158c061791a28f6f5b668cc59091e6c1cbdfa51a0f8f4704739','Preregistered command/split bytes')
    families={f['family_id']:f for f in plan['families']};require(set(families)==set(range(48)),'All48 planned families')
    ids=read(checked(r['sample_identities']));episodes=read(checked(r['episodes']));require(len(episodes)==96,'All96 episodes retained')
    tablemeta=read(ROOT.parent.parent/'retarget_geometry_v2/report.json')['geometry_metadata']
    require(tablemeta['0']['name']=='stage1_single_table' and tablemeta['71']['name']=='phone24_collision_convex','Prior pinned geometry identity0/71')
    rows={};splitcounts={}
    widths={'x58_F64':58,'u20_F64':20,'y58_F64':58,'target_delta52_F64':52,'contact_summary_F64':7}
    for split in ['train','validation','test']:
        z=load(r['outputs'][split]);require(set(z)==ARRAY_KEYS,'Exact14 raw dataset keys')
        n=len(z['row_id_i64']);splitcounts[split]=dict(families=sum(f['split']==split for f in families.values()),rows=n,
            eligible_rows=int(z['training_eligible_bool'].sum()),partial_rows=int((~z['complete_H_bool']).sum()))
        for k,a in z.items():
            expected=np.float64 if k.endswith('_F64') else np.bool_ if k.endswith('_bool') else np.int64
            require(a.dtype==expected and a.shape==((n,widths[k]) if k in widths else (n,)),'Typed partition ABI '+k)
        for i,sid in enumerate(z['row_id_i64']):
            sid=int(sid);require(sid not in rows,'No duplicated global rowID');rows[sid]={k:a[i] for k,a in z.items()}|{'split':split}
    require(len(rows)==len(ids)==r['rows'] and set(rows)==set(range(len(ids))),'Every saved row retained without dedup')
    require(splitcounts==r['split_counts'],'Split/family counts exact')
    audits=r['model_audits'];require(set(audits)=={'0.0003125','0.00015625'},'Both actual dt audit records')
    a0=audits['0.0003125'];a1=audits['0.00015625']
    require(a0['shared_physics_digest_without_dt']==a1['shared_physics_digest_without_dt'],'Common physical model acrossdt')
    for a in audits.values():
        require(a['passed'] is True and a['model_pin']==p['inputs']['model_mjb'],'Source-backed model digest immutability')
        checked(a['model_pin']);bits(np.array(a['joint_range20x2']),np.array(design['actual_limits']['joint_range20x2_F64']),'Actual joint ranges')
        bits(np.array(a['actuator_ctrlrange20x2']),np.array(design['actual_limits']['actuator_ctrlrange20x2_F64']),'Actual ctrl ranges')
    bits(np.array(a0['body_ipos_body_m']),np.array(a1['body_ipos_body_m']),'Actual bodyCOM bytes acrossdt')
    ipos=np.array(a0['body_ipos_body_m']);qrange=np.array(a0['joint_range20x2']);urange=np.array(a0['actuator_ctrlrange20x2'])
    aliases=defaultdict(list);totals=Counter();dtcounts=defaultdict(Counter);terminations=Counter();summary_rows=[];seen=[]
    for ep_index,e in enumerate(episodes):
        fid=e['family_id'];f=families[fid];h=e['dt_s'];n=round(.02/h);ep=fid*2+([.0003125,.00015625].index(h))
        require(e['episode_id']==ep==ep_index and e['split']==f['split'] and e['source_branch']==f['source_branch'] and e['source_record']==f['source_record'],'Family/dt/split source provenance')
        require(e['initialization_seed']==f['initialization_seed'] and e['control_seed']==f['control_seed'],'Separate frozen seeds')
        initial=load(e['initial']);q0=initial['qpos27'];v0=initial['qvel26'];x0=state58(q0,v0,ipos)
        bits(q0,np.array(f['initial_qpos27_F64']),'Planned actual initial qpos27');bits(v0,np.zeros(26),'Zero initial velocity')
        bits(initial['ctrl20'],np.array(f['base_ctrl20_F64']),'Planned base control')
        close(initial['state58'],x0,'initial58_from_raw_qpos_qvel')
        receipt(e['initial_receipt'],initial['integration_state'],0,h)
        require(e['reset_termination']=='INITIAL_READY' and not any(e['initial_warning_numbers']),'Actual run reset diagnostics explicit zero')
        zc=initial['contact_rows24'];pa,ot,ta,pe,sp,ww=contact_math(zc,np.array([0]),x0[None,40:43])
        close(initial['pad_normal_N2'],pa[0],'initial_pad_normal');close(initial['phone_nonpad_normal_N'],ot[0],'initial_nonpad');close(initial['table_normal_N'],ta[0],'initial_table')
        dtcounts[h]['initial_contact_force_reads']+=len(zc)
        prev_q=q0;prev_v=v0;prev_context=initial['integration_state'];tick=0
        for interval,sid in enumerate(e['interval_row_ids']):
            ident=ids[sid];row=rows[sid];seen.append(sid)
            require(ident['sample_id']==sid and ident['episode_id']==ep and ident['family_id']==fid and ident['interval_index']==interval,'Every sample identity exact')
            require(row['episode_id_i64']==ep and row['family_id_i64']==fid and row['interval_index_i64']==interval and row['split']==f['split']==ident['split'],'Array/identity/split joins')
            u=np.array(f['held_ctrl10x20_F64'][interval]);z=load(ident['trace']);N=len(z['post_qpos27'])
            require(0<N<=n and ident['executed_substeps']==N and row['executed_substeps_i64']==N,'Actual native prefix length')
            require(row['n_substeps_i64']==n==ident['requested_substeps'] and row['native_dt_s_F64']==h,'Actual dt and64/128 H')
            require(np.all(u>=urange[:,0]) and np.all(u<=urange[:,1]),'True actuator command bounds')
            bits(row['u20_F64'],u,'Frozen actual row command');bits(z['actual_control20'],u,'Trace native control and plan bytes')
            require(digest(u)==ident['actual_control20_raw_sha256'],'Control SHA')
            require(np.array_equal(z['pre_tick'],np.arange(tick,tick+N)) and np.array_equal(z['post_tick'],np.arange(tick+1,tick+N+1)),'Integer PRE/POST phase')
            close(z['pre_native_time_s'],z['pre_tick']*h,'PRE_clock');close(z['post_native_time_s'],z['post_tick']*h,'POST_clock')
            x=state58(prev_q,prev_v,ipos);yall=state58(z['post_qpos27'],z['post_qvel26'],ipos)
            close(z['source_state58'],x,'source58_from_raw');close(row['x58_F64'],x,'row_source58')
            close(z['post_state58'],yall,'all_POST58_from_raw');close(z['observed_prefix_target58'],yall[-1],'observed_prefix_target')
            for label,array,tt in [('source_receipt','source_integration_state',tick),('source_after_action_receipt','source_after_action_integration_state',tick),('target_receipt','target_integration_state',tick+N)]:receipt(ident[label],z[array],tt,h)
            bits(z['source_integration_state'],prev_context,'Full integration source/previous target continuity')
            preCOM=np.vstack([x[40:43],yall[:-1,40:43]])
            pad,other,table,pen,selfpen,wrench=contact_math(z['contact_rows24'],z['pre_tick'],preCOM)
            for key,val in [('pre_pad_normal_N2',pad),('pre_phone_nonpad_normal_N',other),('pre_table_normal_N',table),('pre_penetration_m',pen),('pre_self_penetration_m',selfpen),('pre_contact_wrench_about_COM_world6',wrench)]:close(z[key],val,key)
            finite=all(np.isfinite(z[k]).all() for k in ['post_qpos27','post_qvel26','post_state58','pre_energy2','pre_actuator_force20','pre_qfrc_actuator20','contact_rows24'])
            warning=any(ident['warning_numbers']);complete=N==n;eligible=bool(complete and finite and not warning)
            require(row['complete_H_bool']==ident['complete_H20ms']==complete and row['training_eligible_bool']==ident['training_eligible']==eligible,'Recomputed completion/visiblefinite/warning mask')
            require(ident['exclusion_reasons']==([] if eligible else [k for k,yes in [('partial_H',not complete),('nonfinite',not finite),('native_warning',warning)] if yes]),'Explicit no-filter exclusion reason')
            if eligible:
                d=delta52(x,yall[-1]);close(z['physical_world_delta52'],d,'trace_independent_delta52');close(row['target_delta52_F64'],d,'row_independent_delta52')
                close(exp3(d[43:46])@x[43:52].reshape(3,3),yall[-1,43:52].reshape(3,3),'independent_Exp_world_roundtrip')
            else:require(np.isnan(row['target_delta52_F64']).all() and np.isnan(z['physical_world_delta52']).all(),'No fabricated delta on excluded prefix')
            if complete:
                close(z['target_at_H_state58'],yall[-1],'fullH_trace_target');close(row['y58_F64'],yall[-1],'fullH_row_target')
            else:require(np.isnan(z['target_at_H_state58']).all() and np.isnan(row['y58_F64']).all(),'Partial H targets all NaN; observed prefix separate')
            cov=np.count_nonzero(pad>.1,axis=0)/N;minimum=pad.min(0);last=pad[-1];cs=np.r_[cov,minimum,last,other.max()]
            close(row['contact_summary_F64'],cs,'all7_contact_labels')
            sm=ident['summary'];close(sm['pad_coverage2'],cov,'summary_coverage');close(sm['pad_min_normal_N2'],minimum,'summary_minFn');close(sm['pad_endpoint_normal_N2'],last,'summary_lastPRE')
            close(sm['phone_nonpad_max_normal_N'],other.max(),'summary_nonpad')
            first=[next((int(t) for t,pn in zip(z['pre_tick'],pad[:,j]) if pn<=.1),-1) for j in range(2)]
            require(sm['first_pad_le_0p1N_global_PRE_tick2']==first and sm['endpoint_force_global_PRE_tick']==tick+N-1,'Loss event and lastPRE tick')
            impulse=[math.fsum(float(a) for a in wrench[:,j])*h for j in range(6)];close(sm['contact_impulse_world6'],impulse,'summary_contact_impulse_PRE')
            exc=np.max(np.maximum(qrange[:,0]-z['post_qpos27'][:,:20],0)+np.maximum(z['post_qpos27'][:,:20]-qrange[:,1],0),axis=1)
            drift=np.linalg.norm(yall[:,40:43]-x0[40:43],axis=1)
            angle=2*np.arccos(np.clip(np.abs(z['post_qpos27'][:,23:27]@q0[23:27]),0,1))
            ev=[('ABORT_ACQUISITION_COM_BOUND' if drift[i]>.15 else 'ABORT_ACQUISITION_ANGLE_BOUND' if angle[i]>math.pi/2 else 'ABORT_ACQUISITION_PENETRATION_BOUND' if pen[i]>.005 else 'ABORT_ACQUISITION_JOINT_BOUND' if exc[i]>.05 else None) for i in range(N)]
            event=next(((i,e0) for i,e0 in enumerate(ev) if e0),None)
            require(finite and not warning,'Actual acquisition has no hidden nonfinite/warning episode')
            if event:require(event==(N-1,ident['termination']),'First original safety crossing exactly last returned step')
            else:require(ident['termination']=='COMPLETE_INTERVAL' and complete,'No early stop without original safety cause')
            if ident['termination']!='COMPLETE_INTERVAL':require(interval==len(e['interval_row_ids'])-1 and e['termination']==ident['termination'],'No steps after first hardstop')
            require('solver_PRE' in ident['force_phase'],'Declared native force PRE phase')
            aliases[(ident['source_after_action_receipt']['sha256'],ident['target_receipt']['sha256'],digest(u),h,n,audits[str(h)]['shared_physics_digest_without_dt'],'phone_COM_world_state58:1',ident['source_receipt']['signature'])].append(sid)
            summary_rows.append(dict(sample_id=sid,family_id=fid,episode_id=ep,split=f['split'],dt_s=h,steps=N,eligible=eligible,complete=complete,pad_coverage2=cov.tolist(),minFn2=minimum.tolist(),endpointFn2=last.tolist(),nonpad_max_N=float(other.max()),table_max_N=float(table.max()),max_penetration_m=float(pen.max())))
            dtcounts[h]['integration_steps']+=N;dtcounts[h]['step_contact_force_reads']+=len(z['contact_rows24'])
            prev_q=z['post_qpos27'][-1];prev_v=z['post_qvel26'][-1];prev_context=z['target_integration_state'];tick+=N
        require(len(e['interval_row_ids'])<=10 and (e['termination']!='MAX10_INTERVALS' or len(e['interval_row_ids'])==10),'Declared ten-H maximum without padding')
        terminations[e['termination']]+=1;dtcounts[h]['MjData']+=1;dtcounts[h]['initial_forward']+=1;dtcounts[h]['receipt_reads']+=1+3*len(e['interval_row_ids'])
    require(sorted(seen)==list(range(len(rows))),'Every row visited exactly once by episode manifest')
    for h,count in dtcounts.items():
        count['MJB_reads']=1;count['contact_force_reads']=count['initial_contact_force_reads']+count['step_contact_force_reads'];count['observer_forward']=count['NN_training_GPU']=0
        require(dict(count)==audits[str(h)]['counts'],'All per-dt native counters from raw lengths')
        totals.update(count)
    require(dict(totals)==r['counts'] and totals['integration_steps']<=92160,'All global native counts/zeroobserver budget')
    aa=[dict(sample_ids=v,family_ids=sorted({ids[i]['family_id'] for i in v}),splits=sorted({ids[i]['split'] for i in v})) for v in aliases.values() if len(v)>1]
    require(aa==r['exact_alias_groups'] and [a for a in aa if len(a['splits'])>1]==r['cross_partition_alias_groups'],'Recomputed fullreceipt/control exact aliases')
    require(r['partition_qualified']==(not any(len(a['splits'])>1 for a in aa)),'Partition qualification from actual aliases')
    parent={i:i for i in families}
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    for a in aa:
        for i in a['family_ids'][1:]:parent[find(i)]=find(a['family_ids'][0])
    groups=defaultdict(list)
    for i in families:groups[find(i)].append(i)
    termpath=report_path.parent.parent/'process_run1/terminal.json';terminal=read(termpath)
    require(terminal['returncode']==0 and terminal['protocol']==r['protocol'] and terminal['runner']==p['source_pins']['collector'],'Actual once terminal binds exact source/protocol')
    for name in ['stdout','stderr']:checked(terminal[name])
    require(terminal['stderr']['bytes']==0,'Actual stderr empty')
    fixture_proof=read(ROOT/'pure_fixture_v1.json');require(fixture_proof['passed'] is True,'Predata independent arithmetic fixture')
    checks={name:True for name in ['source_protocol_model_and_terminal_pins','all48_families_96episodes_frozen_plans','three_typed_split_files_all_rows_retained',
        'all_initial_POST_states_and_samefamily_dt_commands','all_raw_POST58_COM_velocity_arithmetic','complete_world_delta52_and_left_Exp_roundtrip',
        'all_raw_contacts_PRE_forces_min_coverage_endpoint_events','all_full_context_receipts_hashes_clocks_and_continuity','partial_H_NaN_targets_observed_prefix_and_exact_masks',
        'first_safety_stop_no_extension_and_all_native_counters','frozen_family_splits_and_exact_alias_union','no_simulator_NN_or_training_in_review']}
    require(not any(k in sys.modules for k in ['mujoco','torch','warp']),'No simulator/NN imports in saved review')
    write(out/'all_row_recomputed_summaries.json',summary_rows)
    result=dict(schema='phone_episode_acquisition_saved_independent:1',status='PASS_SAVED_ALL_EPISODE_FAMILY_H20MS_STATE_CONTACT_SPLIT_RECEIPTS',
        engineering_qualified=True,dataset_report=pin(report_path),producer_report=pin(report_path),protocol=r['protocol'],
        arrays=r['outputs'],sample_identities=r['sample_identities'],family_plan=r['family_plan'],design=r['design'],
        source_pins={'auditor':pin(Path(__file__)),'source_freeze':pin(ROOT/'SOURCE_FREEZE_v1.json'),'pure_fixture':pin(ROOT/'pure_fixture_v1.json'),
            'prior_geometry_identity':pin(ROOT.parent.parent/'retarget_geometry_v2/report.json')},
        actual_terminal=pin(termpath),checks=checks,passed=len(checks),total=len(checks),counts=dict(totals),rows=len(rows),episode_count=96,family_count=48,
        split_counts=splitcounts,episode_terminations=dict(terminations),exact_alias_groups=aa,alias_family_components=list(groups.values()),
        max_F64_arithmetic_errors=dict(ERRORS),F64_arithmetic_tolerance={'absolute':ATOL,'relative':RTOL},
        all_row_summaries=pin(out/'all_row_recomputed_summaries.json'),
        direct_saved_scope='All exported qpos/qvel/contacts/forces/energy/actuator values, clocks, integration vectors and commands reprocessed. Full context vectors hash/continuity bound; opaque solver scratch components are not rederived.',
        source_backed_scope='Original MJB/zeroaids/model-digest invariants and mj_getState meaning source-bound; no model readback. Per-step qacc26 is not exported: its finite check is bound through reviewed Reader qacc failure->NaN POST/abort logic; visible finite arrays and warning cumulative vectors independently checked.',
        qualification={'heldout_accuracy':False,'Markov_state58':False,'independent_ego_trajectories':False,'real_contact_dynamics_GT':False,'holding_policy':False},
        TEST_scope='All TEST arrays opened only for independent data engineering integrity, never model fitting/selection or accuracy-driven tuning.',
        actual_new_model_FK_LP_simulation_steps_NN_training_calls=0)
    write(out/'report.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixture',action='store_true');ap.add_argument('--report',type=Path);a=ap.parse_args()
    if a.fixture:
        x=fixture();write(ROOT/'pure_fixture_v1.json',x);print(json.dumps({'fixture':pin(ROOT/'pure_fixture_v1.json'),'passed':True}));return
    require(a.report is not None,'Explicit actual saved result path')
    require(not (ROOT/'report.json').exists() and not (ROOT/'failure.json').exists(),'Fresh actual audit, no automatic retry')
    try:
        result=audit(a.report,ROOT);print(json.dumps({'report':pin(ROOT/'report.json'),'passed':result['passed'],'total':result['total'],'counts':result['counts'],'split_counts':result['split_counts']}))
    except BaseException as error:
        write(ROOT/'failure.json',dict(schema='phone_episode_acquisition_saved_independent_failure:1',error=repr(error),auditor=pin(Path(__file__)),dataset_report=pin(a.report),automatic_retry=False,actual_new_native_steps=0));raise
if __name__=='__main__':main()
