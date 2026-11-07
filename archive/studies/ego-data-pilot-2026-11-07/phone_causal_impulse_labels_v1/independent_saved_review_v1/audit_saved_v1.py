"""Full2942-trace independent per-slot saved arithmetic; no producer import."""
from pathlib import Path
import collections,hashlib,json,math,sys,time
import numpy as np
H=Path(__file__).resolve().parent;D=H.parent;P=D.parent
FIELDS=['impulse_source_BODY_moving_COM22x6','mean_positive_Fn22_N','min_positive_Fn22_N','max_positive_Fn22_N','last_PRE_positive_Fn22_N','Fn_gt_0p1_coverage22','generated_contact_coverage22']
ATOL=2e-12;RTOL=2e-12
def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
    if pin(r['path'])!=r:raise ValueError('Changed pinned file '+r['path'])
    return Path(r['path'])
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def close(a,b):return bool(np.allclose(a,b,atol=ATOL,rtol=RTOL,equal_nan=False))
def mv(A,x):return np.asarray([math.fsum(float(A[i,j])*float(x[j]) for j in range(3)) for i in range(3)])
def cross(a,b):return np.asarray([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
def main():
    started=time.perf_counter();rp=D/'completed_v2/report.json';r=read(rp);pr=read(checked(r['protocol']));checked(pr['source'])
    for v in pr['inputs'].values():checked(v)
    meta=read(pr['inputs']['geometry_metadata']['path']);binding=read(checked(meta['binding']));checked(meta['model']);pars=read(pr['inputs']['physical_parameters']['path']);acq=read(pr['inputs']['acquisition_report']['path']);records=read(checked(acq['records']))
    for v in pars['sourcepins'].values():checked(v)
    labels=arrays(checked(r['outputs']['all_labels']));identities=read(checked(r['outputs']['row_identities']))
    pairs=meta['pairs'];ids=[v['other_geom_id'] for v in pairs];names=[v['name'] for v in pairs];index={g:i for i,g in enumerate(ids)};phone=meta['phone_geom_id']
    checks={k:True for k in ['all_pins_and_canonical22_original_compiled_pair_slots','all2952_identity_roles_and2942_known10unknown_masks',
        'every_per_slot_sourceBODY_movingCOM_impulse_recomputed','every_per_slot_Fn_mean_min_max_last_coverage_recomputed',
        'all_unexecuted_fields_NaN_and_observed_absent_slots_zero','independent_world_tick_total_vs_Reader_separate_check',
        'linear_and_angular_endpoint_momentum_residuals_recomputed','all_native_tick_and_phone_contact_counts','serialization_v2_only_bool_and_numeric_bytes_preserved','no_new_physics_or_network_execution']}
    checks['all_pins_and_canonical22_original_compiled_pair_slots']=len(ids)==len(set(ids))==22 and len(binding['explicit_pairs'])==22 and meta['model']==binding['model']==acq['inputs']['model_mjb'] and phone==71 and np.array_equal(labels['pair_other_geom_ids'],ids) and r['pair_names']==names
    for i,(pair,source) in enumerate(zip(pairs,binding['explicit_pairs'])):
        checks['all_pins_and_canonical22_original_compiled_pair_slots'] &= pair['pair_id']==i and pair['name']==source['name'] and pair['phone_geom_id']==71 and pair['other_geom_name']==meta['geom_names'][pair['other_geom_id']] and meta['geom_names'][pair['geom1_id']]==source['geom1'] and meta['geom_names'][pair['geom2_id']]==source['geom2']
    require_known=np.asarray([v['executed'] for v in records],dtype=np.bool_)
    checks['all2952_identity_roles_and2942_known10unknown_masks']=len(records)==len(identities)==2952 and np.array_equal(labels['branch_id'],np.arange(2952,dtype=np.int64)) and labels['physical_labels_known'].dtype==np.bool_ and np.array_equal(labels['physical_labels_known'],require_known) and int(require_known.sum())==2942 and int((~require_known).sum())==10
    mass=float(pars['mass_kg']);I=np.asarray(pars['I_body_3x3']);gravity=np.asarray(pars['gravity_world_m_s2'])
    if not np.isfinite(I).all() or np.linalg.eigvalsh(I).min()<=0:raise ValueError('Inherited finite positive inertia')
    reads=ticks_total=phone_rows=all_rows=0;order_counts=collections.Counter();Fn_sign_counts=collections.Counter();partition_counts=collections.Counter();maxJ=np.zeros((22,6));maxFn={k:np.zeros(22) for k in FIELDS[1:]};max_world=max_lin=max_ang=0.;linear=[];angular=[];slot_contacts=np.zeros(22,dtype=np.int64);absent_windows=np.zeros(22,dtype=np.int64)
    for i,record in enumerate(records):
        expected={k:record[k] for k in ['branch_id','context_id','partition','original_family_id','kind','executed']}
        checks['all2952_identity_roles_and2942_known10unknown_masks'] &= record['branch_id']==i and identities[i]==expected
        partition_counts[record['partition']]+=1
        if not record['executed']:
            checks['all_unexecuted_fields_NaN_and_observed_absent_slots_zero'] &= all(np.isnan(labels[k][i]).all() for k in FIELDS+['linear_momentum_residual_world_Ns','angular_momentum_residual_world_Nms']) and labels['actual_duration_s'][i]==0
            continue
        if not (record['complete_H20ms'] is True and record['training_eligible'] is True):raise ValueError('Actual full-H eligible source')
        # Read each unique saved branch exactly once; only these fields are used.
        path=checked(record['output'])
        with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in ['source_state58','observed_prefix_target58','post_state58','pre_tick','contact_rows24','pre_contact_wrench_about_COM_world6']}
        reads+=1;tick=a['pre_tick'];n=len(tick);h=record['native_h_s'];T=h*n;ticks_total+=n;cr=a['contact_rows24'];all_rows+=len(cr)
        if n!=record['executed_substeps'] or n!=record['n_substeps'] or not np.all(np.diff(tick)==1) or T!=.02:raise ValueError('All actual native ticks/H')
        s=a['source_state58'];y=a['observed_prefix_target58'];R=s[43:52].reshape(3,3);R1=y[43:52].reshape(3,3);COM=np.vstack([s[40:43],a['post_state58'][:-1,40:43]])
        if np.max(abs(R.T@R-np.eye(3)))>1e-10:raise ValueError('Frozen source BODY frame')
        bytick={int(t):j for j,t in enumerate(tick)};Fn=np.zeros((n,22));generated=np.zeros((n,22),dtype=np.int64);parts=[[[] for _ in range(6)] for _ in range(22)];world=np.zeros((n,6))
        for row in cr:
            g0,g1=map(int,row[1:3])
            if phone not in (g0,g1):continue
            if (g0==phone)==(g1==phone):raise ValueError('Object internal pair not a boundary label')
            ti=bytick[int(row[0])]
            if row[0]!=int(row[0]):raise ValueError('Integer global PRE tick')
            other=g0 if g1==phone else g1;slot=index[other];sign=1 if g1==phone else -1;order_counts['phone_geom2' if sign==1 else 'phone_geom1']+=1;phone_rows+=1;slot_contacts[slot]+=1
            frame=row[9:18].reshape(3,3);F=sign*mv(frame.T,row[18:21]);direct=sign*mv(frame.T,row[21:24]);M=cross(row[6:9]-COM[ti],F)+direct
            body=np.r_[mv(R.T,F),mv(R.T,M)]
            for k,v in enumerate(body):parts[slot][k].append(h*float(v))
            Fn[ti,slot]+=max(0.,float(row[18]));generated[ti,slot]+=1;world[ti,:3]+=F;world[ti,3:]+=M
            Fn_sign_counts['positive' if row[18]>0 else 'zero' if row[18]==0 else 'negative']+=1
        J=np.asarray([[math.fsum(v) for v in values] for values in parts]);maxJ=np.maximum(maxJ,abs(J-labels[FIELDS[0]][i]))
        checks['every_per_slot_sourceBODY_movingCOM_impulse_recomputed'] &= close(J,labels[FIELDS[0]][i])
        stats=dict(mean_positive_Fn22_N=np.asarray([math.fsum(float(v) for v in Fn[:,j])/n for j in range(22)]),min_positive_Fn22_N=Fn.min(0),max_positive_Fn22_N=Fn.max(0),last_PRE_positive_Fn22_N=Fn[-1],Fn_gt_0p1_coverage22=np.asarray([sum(float(v)>.1 for v in Fn[:,j])/n for j in range(22)]),generated_contact_coverage22=np.asarray([sum(int(v)>0 for v in generated[:,j])/n for j in range(22)]))
        for k,v in stats.items():
            maxFn[k]=np.maximum(maxFn[k],abs(v-labels[k][i]));checks['every_per_slot_Fn_mean_min_max_last_coverage_recomputed'] &= close(v,labels[k][i])
        absent=np.all(generated==0,axis=0);absent_windows+=absent
        checks['all_unexecuted_fields_NaN_and_observed_absent_slots_zero'] &= np.array_equal(labels[FIELDS[0]][i,absent],np.zeros((int(absent.sum()),6))) and all(np.array_equal(labels[k][i,absent],np.zeros(int(absent.sum()))) for k in FIELDS[1:])
        world_error=float(np.max(abs(world-a['pre_contact_wrench_about_COM_world6'])));max_world=max(max_world,world_error)
        checks['independent_world_tick_total_vs_Reader_separate_check'] &= world_error<1e-12
        total=np.asarray([math.fsum(float(v) for v in J[:,k]) for k in range(6)]);lin=mass*(y[52:55]-s[52:55])-mass*gravity*T-mv(R,total[:3])
        L0=mv(R,mv(I,mv(R.T,s[55:58])));L1=mv(R1,mv(I,mv(R1.T,y[55:58])));ang=L1-L0-mv(R,total[3:])
        max_lin=max(max_lin,float(np.max(abs(lin-labels['linear_momentum_residual_world_Ns'][i]))));max_ang=max(max_ang,float(np.max(abs(ang-labels['angular_momentum_residual_world_Nms'][i]))))
        checks['linear_and_angular_endpoint_momentum_residuals_recomputed'] &= close(lin,labels['linear_momentum_residual_world_Ns'][i]) and close(ang,labels['angular_momentum_residual_world_Nms'][i]) and labels['actual_duration_s'][i]==T
        linear.append(lin);angular.append(ang)
    vector_rms=lambda v:math.sqrt(math.fsum(math.fsum(float(t)*float(t) for t in row) for row in v)/len(v))
    lr,ar=vector_rms(linear),vector_rms(angular)
    checks['linear_and_angular_endpoint_momentum_residuals_recomputed'] &= close(lr,r['linear_momentum_vectorRMS_Ns']) and close(ar,r['angular_momentum_vectorRMS_Nms'])
    checks['all_native_tick_and_phone_contact_counts']=reads==2942 and ticks_total==282432==r['counts']['decoded_PRE_ticks'] and phone_rows==271092==r['counts']['phone_contact_rows']
    oldnpz=D/'completed_v1/all2952_pair22_physical_labels.npz';oldids=D/'completed_v1/row_identities.json';pres=read(D/'numeric_preservation_v2.json')
    checks['serialization_v2_only_bool_and_numeric_bytes_preserved']=oldnpz.read_bytes()==Path(r['outputs']['all_labels']['path']).read_bytes() and oldids.read_bytes()==Path(r['outputs']['row_identities']['path']).read_bytes() and pres['all_arrays_bit_identical'] is True and pres['actual_both_exports_saved_branch_NPZ_reads']==5884
    a=(D/'export_saved_labels_v1.py').read_text();b=(D/'export_saved_labels_v2.py').read_text();b=b.replace("out=H/'completed_v2';protocol=H/'protocol_v2.json'","out=H/'completed_v1';protocol=H/'protocol_v1.json'").replace('    checks={key:bool(value) for key,value in checks.items()}  # JSON scalar conversion only\n','')
    checks['serialization_v2_only_bool_and_numeric_bytes_preserved'] &= a==b
    out=dict(schema='causal_phone22_impulse_labels_saved_independent:1',status='PASS_ALL2942_PERPAIR_IMPULSE_AND_FORCE_LABELS_NOT_PHYSICS_CERTIFICATE' if all(checks.values()) else 'FAILED_SAVED_LABEL_REVIEW',passed=sum(map(bool,checks.values())),total=len(checks),checks={k:bool(v) for k,v in checks.items()},producer_report=pin(rp),protocol=r['protocol'],source_auditor=pin(__file__),labels=r['outputs']['all_labels'],row_identities=r['outputs']['row_identities'],geometry_metadata=pr['inputs']['geometry_metadata'],physical_parameters=pr['inputs']['physical_parameters'],numeric_preservation=pin(D/'numeric_preservation_v2.json'),
        actual_read_scope=dict(unique_branch_NPZ_values_read=reads,label_NPZ_values_read=1,old_archive_value_reads=0,old_archive_raw_bytes_compared=True,decoded_native_PRE_ticks=ticks_total,raw_contact_rows=all_rows,phone_boundary_contact_rows=phone_rows,contact_geom_orders=dict(order_counts),raw_phone_local_Fn_signs=dict(Fn_sign_counts),reviewer_native_NN_training_calls=0),
        pair_reviews=[dict(slot=i,other_geom_id=g,name=names[i],raw_phone_contact_rows=int(slot_contacts[i]),observed_windows_with_no_pair_contact=int(absent_windows[i]),max_independent_impulse_abs_error_sourceBODY6=maxJ[i].tolist(),max_Fn_stats_abs_errors={k:float(v[i]) for k,v in maxFn.items()}) for i,g in enumerate(ids)],
        independent_world_tick_sum_max_abs_error=max_world,independent_linear_momentum_vectorRMS_Ns=lr,independent_angular_momentum_vectorRMS_Nms=ar,max_saved_linear_residual_difference=max_lin,max_saved_angular_residual_difference=max_ang,
        role_rows=dict(partition_counts),numeric_tolerance=dict(absolute=ATOL,relative=RTOL,scope='Scalar fsum/3x3 arithmetic versus vectorized NumPy rounding; not a scientific accuracy gate'),elapsed_s=time.perf_counter()-started,
        scope=['All2942 executed branches and all22slots independently decoded, not inferred from the separate total-wrench check. Ten unavailable rows preserve all future labels NaN/knownFalse.',
            'Canonical slot order is independently joined to original compiled pair_id/name/geom IDs and original explicit-pair binding; no remapping by geometric closeness or positive-force subset.',
            'Signed force and direct torque use geom2 phone plus/geom1 phone minus, frame.T, lever about each actual PRE moving COM; both blocks rotate into fixed sourceBODY axes. Actual geom-order counts disclose whether a sign branch was observed.',
            'PositiveFn summaries and generated-contact coverage are distinct; raw signed impulse includes every phone contact, even nonpositive/subthresholdFn. Zero only means observed absence at those native ticks.',
            'No new MJB/kinematics/physics or nativeLog/NN/AD. Rsource/COM/omega endpoints and assumed mass/inertia/gravity are saved source authority; no measured-human-force, fullwrench/continuum certificate or holding success.',
            'Future H targets are labels, not deployable current inputs. Both roles are originalTRAIN-state families: responsefit and heldout-actioncheck are retained, not unseen-state/trajectory/ego validation.'])
    with (H/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(dict(report=pin(H/'report.json'),passed=out['passed'],total=out['total'],checks=out['checks'],reads=reads,phone_rows=phone_rows,max_pair_impulse_error=float(maxJ.max()),reader_sum_error=max_world,linearRMS=lr,angularRMS=ar,roles=out['role_rows'])))
    return 0 if all(checks.values()) else 1
if __name__=='__main__':sys.exit(main())
