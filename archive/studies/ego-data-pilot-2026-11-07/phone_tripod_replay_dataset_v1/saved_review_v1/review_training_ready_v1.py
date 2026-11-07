"""Source-author saved labels/data-ready review; Root executed the exporter.

Independent scalar fsum arithmetic for all22 signed impulse/positiveFn labels.
Raw300-state/trajectory integrity is delegated to the separate pinned Windows
review. No Native/FK/query/NN, no old CHECK/fresh/old TRAIN array value reads.
"""
from pathlib import Path
import hashlib,importlib.util,json,math,sys
from collections import defaultdict
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent.parent
PAIR=(0,2,8,11,13,15,21,24,26,28,34,37,39,41,47,50,52,54,60,63,65,67)
PAD=(15,28,41)
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):assert pin(v['path'])==v;return Path(v['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def npz(v):
    with np.load(checked(v),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
ERRORS={}
def close(a,b,label):
    a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),label
    err=float(np.max(abs(a-b))) if a.size else 0.;ERRORS[label]=max(ERRORS.get(label,0.),err)
    assert np.allclose(a,b,atol=2e-12,rtol=2e-12),label

def scalar_labels(a,x,h):
    ticks=a['pre_tick'];N=len(ticks);preCOM=np.vstack([x[None,40:43],a['post_state58'][:-1,40:43]])
    R=x[43:52].reshape(3,3);acc=[[[] for _ in range(6)] for _ in PAIR]
    fn=defaultdict(list);generated=np.zeros((N,22),dtype=np.int64);orders=defaultdict(int);phone_rows=0
    for row in a['contact_rows24']:
        g1,g2=map(int,row[1:3])
        if 71 not in [g1,g2]:continue
        phone_rows+=1;sign=1. if g2==71 else -1.;orders['phone_geom2' if sign>0 else 'phone_geom1']+=1
        gid=g1 if sign>0 else g2;j=PAIR.index(gid);t=int(np.searchsorted(ticks,int(row[0])));assert ticks[t]==row[0]
        frame=row[9:18].reshape(3,3);f=row[18:24]
        F=[sign*math.fsum(float(frame[k,c])*float(f[k]) for k in range(3)) for c in range(3)]
        direct=[sign*math.fsum(float(frame[k,c])*float(f[k+3]) for k in range(3)) for c in range(3)]
        r=row[6:9]-preCOM[t]
        M=[math.fsum([r[1]*F[2],-r[2]*F[1],direct[0]]),math.fsum([r[2]*F[0],-r[0]*F[2],direct[1]]),math.fsum([r[0]*F[1],-r[1]*F[0],direct[2]])]
        B=[math.fsum(float(R[k,c])*float(F[k]) for k in range(3)) for c in range(3)]+[math.fsum(float(R[k,c])*float(M[k]) for k in range(3)) for c in range(3)]
        for c in range(6):acc[j][c].append(h*B[c])
        fn[(t,j)].append(max(0.,float(f[0])));generated[t,j]+=1
    J=np.array([[math.fsum(v) for v in group] for group in acc]);Fn=np.zeros((N,22))
    for (t,j),values in fn.items():Fn[t,j]=math.fsum(values)
    mean=np.array([math.fsum(float(v) for v in Fn[:,j])/N for j in range(22)])
    cov=np.array([sum(v>.1 for v in Fn[:,j])/N for j in range(22)])
    triple=sum(all(Fn[t,PAIR.index(g)]>.1 for g in PAD) for t in range(N))/N
    return J,mean,cov,(generated>0).mean(0),triple,phone_rows,dict(orders)

def run():
    report_path=P/'phone_tripod_replay_dataset_v1/completed_v1/report.json';r=read(report_path)
    assert r['schema']=='phone_tripod_replay_training_dataset:1' and r['engineering_qualified'] is True and r['passed']==r['total']==12
    assert r['canonical_pair_geom_ids']==list(PAIR) and r['designated_pad_geom_ids']==list(PAD)
    raw_review_path=P/'phone_tripod_task_acquisition_v1/saved_review_v1/report.json';raw=read(raw_review_path)
    assert pin(raw_review_path)['sha256']=='c98b7f481ddaa6e4401d9713f3c630c60727c64c33d72c43e8f3fc46018a1e97'
    assert raw['engineering_qualified'] is True and raw['passed']==raw['total']==12 and raw['producer_report']==r['collector_report']
    terminal=read(P/'phone_tripod_replay_dataset_v1/process_run1/terminal.json');assert terminal['returncode']==0 and terminal['source']==r['sources']['exporter'] and terminal['protocol']==r['protocol']
    codecpath=P/'phone_transition_dataset_v1/transition_codec_v1.py';assert pin(codecpath)['sha256']=='1bf33a7fd27ae51d38a71c1f7e095dbe3047c4828ee6e65c199117bf536f4cac'
    spec=importlib.util.spec_from_file_location('tripod_review_world_codec',codecpath);codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
    a=npz(r['all_rows']);notes=read(checked(r['sample_identities']));assert len(notes)==205
    assert np.array_equal(a['row_id_i64'],np.arange(205)) and a['x58_F64'].shape==(205,58) and a['memory83_F64'].shape==(205,83)
    assert a['obs222_FP32'].dtype==np.float32 and a['obs222_FP32'].shape==(205,222) and a['normalized_action20_F64'].shape==(205,20)
    assert a['complete_H_bool'].all() and a['physical_labels_known_bool'].all() and a['WM_eligible_bool'].all() and a['Actor_start_eligible_bool'].all()
    assert int(a['actor_absorbing_done_bool'].sum())==32 and not a['reference_timeout_bool'].any() and not a['acquisition_truncated_bool'].any() and not a['reader_done_bool'].any()
    phys=read(checked(r['physical_parameters']));g=np.array(phys['gravity_world_m_s2']);rows=[];phone_count=0;ticks=0;orders=defaultdict(int)
    for i,note in enumerate(notes):
        identity=note['source_identity'];z=npz(identity['trace']);x=a['x58_F64'][i];y=a['y58_F64'][i];h=float(a['native_dt_s_F64'][i]);N=len(z['pre_tick'])
        assert N==int(a['n_substeps_i64'][i])==int(a['executed_substeps_i64'][i]) and h*N==.02
        J,mean,cov,generated,triple,count,order=scalar_labels(z,x,h)
        close(J,a['impulse_source_BODY_moving_COM22x6_F64'][i],'scalar_signed_impulse22x6')
        close(mean,a['mean_positive_Fn22_N_F64'][i],'scalar_positiveFn22')
        close(cov,a['Fn_gt_0p1_coverage22_F64'][i],'scalar_coverage22')
        close(generated,a['generated_contact_coverage22_F64'][i],'scalar_generated_coverage22')
        close(np.array(triple),np.array(a['simultaneous_triple_coverage_F64'][i]),'scalar_triple_coverage')
        close(codec.delta52(x,y),a['target_delta52_F64'][i],'world_left_delta52')
        expected=np.r_[y[:40]-x[:40],y[40:43]-x[40:43]-x[52:55]*(h*N)-.5*g*(h*N)**2,codec.log_world(y[43:52].reshape(3,3)@x[43:52].reshape(3,3).T),J.ravel()]
        close(expected,a['target_physical_heads178_F64'][i],'physical178_unit_layout')
        rows.append(dict(row_id=i,family_id=int(a['family_id_i64'][i]),episode_id=int(a['episode_id_i64'][i]),step_k=int(a['step_k_i64'][i]),
            triple_coverage=triple,designated_marginal_coverage3=cov[[PAIR.index(g) for g in PAD]].tolist(),pair_meanFn_max_N=float(mean.max()),
            force_impulse22_vectorRMS_Ns=float(np.sqrt(np.mean(np.sum(J[:,:3]**2,axis=1)))),moment_impulse22_vectorRMS_Nms=float(np.sqrt(np.mean(np.sum(J[:,3:]**2,axis=1))))))
        phone_count+=count;ticks+=N
        for key,v in order.items():orders[key]+=v
    assert ticks==19712
    splits={}
    for name,expectedrows,expectedfamilies in [('train',153,set(range(12))),('monitor',52,set(range(12,16)))]:
        part=npz(r['partition_arrays'][name]);ids=part['row_id_i64'];assert len(ids)==expectedrows and set(part['family_id_i64'])==expectedfamilies
        for key in part:assert np.array_equal(part[key],a[key][ids],equal_nan=True),'Partition preserves original typed rows '+key
        assert all(part[r['field_map'][key]].shape[0]==expectedrows for key in r['field_map'])
        splits[name]=dict(rows=expectedrows,WM_eligible=int(part['WM_eligible_bool'].sum()),Actor_start_eligible=int(part['Actor_start_eligible_bool'].sum()),H2_start_eligible=int(part['H2_start_eligible_bool'].sum()),partial=int((~part['complete_H_bool']).sum()),absorbing_task_boundaries=int(part['actor_absorbing_done_bool'].sum()))
    alias=read(checked(r['alias_review']));assert alias['cross_split_groups']==[] and alias['exact_input_action_groups']==[]
    train=a['target_physical_heads178_F64'][np.isin(a['family_id_i64'],list(range(12)))]
    blocks=dict(q_rad=np.arange(20),qd_rad_s=np.arange(20,40),COM_ballistic_residual_m=np.arange(40,43),world_dR_rad=np.arange(43,46),
        force_impulse_Ns=46+np.array([6*j+k for j in range(22) for k in range(3)]),moment_impulse_Nms=46+np.array([6*j+k for j in range(22) for k in range(3,6)]))
    scale_readiness={name:dict(components=len(idx),finite=bool(np.isfinite(train[:,idx]).all()),population_std_min=float(train[:,idx].std(0).min()),population_std_max=float(train[:,idx].std(0).max()),nonzero_std_components=int(np.count_nonzero(train[:,idx].std(0)))) for name,idx in blocks.items()}
    assert all(v['finite'] for v in scale_readiness.values())
    assert not any(k in sys.modules for k in ['torch','mujoco','warp'])
    write(H/'all205_label_summaries.json',rows)
    checks={k:True for k in ['exact_dataset_terminal_and_raw_review_parent_pins','all205_completeH153TRAIN52MONITOR_partial0','all22_signed_J_scalar_fsum_frame_sign_lever_directonce','positiveFn_marginal_generated_and_joint_triple_labels','worldleft52_and_physical178_unit_layout','typed_memory83_obs222_action20_actualu_training_fields','first_boundary_task_failure_absorbing32_no_false_collectiontimeout','partition_rowbytes_families_and_no_crosssplit_alias','finite_TRAIN_unit_blocks_ready_for_frozen_loss_scaling','no_historicalCHECK_fresh_or_NN_native_value_calls']}
    result=dict(schema='phone_tripod_replay_dataset_saved_review:1',status='PASS_SAVED_TRIPOD_TASK_TRANSITIONS_AND_PAIR22_LABELS_NOT_POLICY_QUALIFICATION',engineering_qualified=True,
        dataset_report=pin(report_path),collector_report=r['collector_report'],raw_integrity_review=pin(raw_review_path),export_terminal=pin(P/'phone_tripod_replay_dataset_v1/process_run1/terminal.json'),
        checks=checks,passed=len(checks),total=len(checks),rows=205,split_counts=splits,partial_rows=0,partial_behavior_actual_cases_exercised=False,
        canonical_pair_geom_ids=list(PAIR),designated_pad_geom_ids=list(PAD),partition_arrays=r['partition_arrays'],field_map=r['field_map'],
        scalar_arithmetic_max_absolute_errors=ERRORS,numerical_tolerance=dict(atol=2e-12,rtol=2e-12,scope='NumPy vectorization versus scalar fsum rounding, not a physical accuracy gate'),
        labels_summary=dict(triple_coverage_mean=float(a['simultaneous_triple_coverage_F64'].mean()),triple_coverage_min=float(a['simultaneous_triple_coverage_F64'].min()),triple_coverage_max=float(a['simultaneous_triple_coverage_F64'].max()),
            nonzero_pair_slot_count=int(np.count_nonzero(np.any(abs(a['impulse_source_BODY_moving_COM22x6_F64'])>0,axis=(0,2)))),source_body_linear_residual_vectorRMS_Ns=float(np.sqrt(np.mean(np.sum(a['linear_momentum_residual_world_Ns_F64']**2,axis=1)))),source_body_angular_residual_vectorRMS_Nms=float(np.sqrt(np.mean(np.sum(a['angular_momentum_residual_world_Nms_F64']**2,axis=1))))),
        TRAIN_physical_unit_scale_readiness=scale_readiness,counts=dict(raw_transition_NPZs=205,PRE_ticks=ticks,phone_contact_rows=phone_count,geom_order_rows=dict(orders),reviewer_new_native_FK_NN_training_calls=0,oldTRAIN_CHECK_TEST_fresh_NPZ_value_reads=0),
        summaries=pin(H/'all205_label_summaries.json'),source_pins=dict(reviewer=pin(Path(__file__)),scalar_codec=pin(codecpath),source_freeze=pin(H/'SOURCE_FREEZE_v1.json')),
        scope=['Reviewer authored this dataset adapter/exporter; Root executed it. This is a source-author saved review, with alternative scalar label arithmetic, not an independent sampler/physics experiment.',
            'Full300 receipts/trajectory/EMA/obs integrity delegated to the separate pinned collector saved review; not re-audited here.',
            'Per-pair meanFn/cov and signed impulses are observed future labels. They are not deployment inputs or human force/strict contact certificates.',
            'All205 actual intervals are complete; partial0 is measured, while partial handling remains a source rule rather than exercised actual-case evidence.',
            'TRAIN-only block stds are readiness descriptors, not fitted model normalizers or new loss weights. MONITOR is known-initial-lineage monitoring, not unseen ego.',
            'Only boundaryPOST task failure is absorbing; any-native-tick descriptors did not replace MDP terminal. No policy or2s holding claim.'])
    write(H/'report.json',result);return result
if __name__=='__main__':
    assert not (H/'report.json').exists() and not (H/'failure.json').exists()
    try:
        r=run();print(json.dumps(dict(report=pin(H/'report.json'),passed=r['passed'],total=r['total'],splits=r['split_counts'],labels=r['labels_summary'])))
    except BaseException as e:
        write(H/'failure.json',dict(error=repr(e),source=pin(Path(__file__)),automatic_retry=False,new_native_NN_calls=0));raise
