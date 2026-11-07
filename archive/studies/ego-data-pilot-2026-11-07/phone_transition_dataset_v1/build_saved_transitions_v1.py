"""Build every frozen MPC H2 summary into two H=.02 saved transitions."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import transition_codec_v1 as codec
H=Path(__file__).resolve().parent
P=H.parent
OUT=H/'dataset_v1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(v):
    p=Path(v['path']);assert pin(p)==v,'Changed input '+str(p);return p
def save(p,j):
    with Path(p).open('x',encoding='utf8') as f:json.dump(j,f,indent=2,allow_nan=False)
def bits(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);return a.shape==b.shape and a.tobytes()==b.tobytes()
def close(a,b,tol=2e-12):return bool(np.max(abs(np.asarray(a)-np.asarray(b)),initial=0)<=tol)

def main():
    prereg=read(H/'preregistration_v1.json');assert prereg['builder']==pin(__file__) and prereg['codec']==pin(H/'transition_codec_v1.py')
    for p in prereg['inputs'].values():checked(p)
    fixture=read(H/'codec_fixture_v1.json');assert fixture['passed'] is True and fixture['codec_source']==prereg['codec']
    if OUT.exists():raise ValueError('Fresh saved dataset only; no overwrite')
    OUT.mkdir()
    r=read(checked(prereg['inputs']['MPC_report']));protocol=read(checked(r['protocol']))
    for v in protocol['inputs'].values():checked(v)
    review=read(checked(prereg['inputs']['MPC_saved_review']));assert review['producer_report']==prereg['inputs']['MPC_report'] and review['passed']==review['total']==13 and all(v is True for v in review['checks'].values())
    with np.load(checked(r['trace']),allow_pickle=False) as z:actual={k:z[k].copy() for k in z.files}
    ipos=np.asarray(read(checked(prereg['inputs']['COM_metadata']))['CPU_compile']['phone_asset.xml']['body_ipos_m'])
    source_q=[];source_v=[];boundaries=[];boundarytime=[]
    for i in range(7):
        q=actual['initial_qpos27'] if i==0 else actual['post_qpos'][i*128-1];v=actual['initial_qvel26'] if i==0 else actual['post_qvel'][i*128-1]
        x=codec.actual_state(q,v,ipos);source_q.append(q);source_v.append(v);boundaries.append(x);boundarytime.append(0. if i==0 else float(actual['post_time_native_s'][i*128-1]))
        assert close(x[40:43],actual['initial_COM_world3'] if i==0 else actual['post_COM'][i*128-1])
        if i>0:assert close(x[52:55],actual['post_v_COM'][i*128-1]) and close(x[55:],actual['post_omega_world'][i*128-1])
    X=[];Y=[];U=[];D=[];endpointFn=[];endpointPREtime=[];targetquats=[];clocks=[];identities=[];branchrows=[];cohorts=[];snapshotstates=[];controls=[];codecerrors=[];deltart=[];Rerrors=[];angles=[];branchidorder=[];confirmed=0
    for i in range(7):
        path=checked(prereg['inputs'][f'interval_{i:03d}']);interval=read(path);assert interval['interval']==i and interval['start_tick']==i*128
        assert len(interval['evaluations'])==23
        source_digest=interval['source_digest_before_actual'];cohort_samples=[];cohort_branches=[]
        for eidx,b in enumerate(interval['evaluations']):
            bid=b['branch_index'];branchidorder.append(bid);cohort_branches.append(bid)
            assert b['status']=='COMPLETE_H2_PLANNING_ONLY' and b['actual_planning_steps']==256 and b['complete_clone_equal'] is True and b['source_state_before']==b['source_state_after']==source_digest
            u=np.asarray(b['controls2x20'],dtype=float);assert u.shape==(2,20) and np.isfinite(u).all()
            assert bits(u[0],u[1]),'This frozen producer held the same command twice; preserve both rows'
            snapshots=b['snapshots'];assert len(snapshots)==2
            states=[codec.snapshot_state(s,ipos) for s in snapshots];snapshotstates.append(states);controls.append(u)
            branchrows.append(dict(branch_id=bid,source_boundary_cohort=i,purpose=b['purpose'],cost=b['cost'],selected=b['branch_index']==interval['selected_branch'],original_H2_horizon_safety_pass=b['original_horizon_safety_limits_pass'],H2_aggregates=b['aggregates'],aggregate_scope='Entire256step H2; never a per-H or per-step label',source_clone_receipt_before=b['source_state_before'],source_clone_receipt_after=b['source_state_after'],snapshot20_integration_sha256=snapshots[0]['integration_state_sha256'],snapshot40_integration_sha256=snapshots[1]['integration_state_sha256']))
            for h in [0,1]:
                sample_id=len(X);cohort_samples.append(sample_id);x=boundaries[i] if h==0 else states[0];y=states[h];snap=snapshots[h]
                assert snap['post_tick']==128*(h+1) and close(snap['elapsed_time_nominal_s'],.02*(h+1))
                assert close(snap['time_nominal_s'],boundarytime[i]+.02*(h+1),1e-10)
                assert close(snap['pad_force_pre_time_s'],boundarytime[i]+.02*(h+1)-.00015625,1e-10)
                d=codec.delta52(x,y);recovered=codec.apply_delta52(x,d);error=np.max(abs(recovered-y));assert error<=2e-12
                for value in [x,y]:
                    rt=codec.codec_roundtrip(value,ipos);err=float(np.max(abs(rt-value)));codecerrors.append(err);assert err<=2e-12
                    R=value[43:52].reshape(3,3);Rerrors.append(max(float(np.max(abs(R.T@R-np.eye(3)))),abs(float(np.linalg.det(R)-1))))
                deltart.append(float(error));angles.append(float(np.linalg.norm(d[43:46])))
                is_confirmed=bool(h==0 and i<6 and b['branch_index']==interval['selected_branch'])
                if is_confirmed:
                    k=(i+1)*128-1;ax=codec.actual_state(actual['post_qpos'][k],actual['post_qvel'][k],ipos);assert close(ax,y)
                    assert bits(np.asarray(snap['state']['q20']),actual['post_qpos'][k,:20]) and bits(np.asarray(snap['state']['qd20']),actual['post_qvel'][k,:20]);confirmed+=1
                input_sha=source_digest['integration_sha256'] if h==0 else snapshots[0]['integration_state_sha256']
                identities.append(dict(sample_id=sample_id,source_boundary_cohort=f'SOURCE_BOUNDARY_{i:03d}',source_boundary_index=i,branch_id=bid,H2_step=h,source_evaluation_index=eidx,input_source='actual_boundary_qpos_qvel' if h==0 else 'same_branch_20ms_snapshot',input_actual_boundary_POST_tick=i*128,input_branch_relative_POST_tick=0 if h==0 else 128,target_branch_relative_POST_tick=128*(h+1),nominal_input_global_tick=(i+h)*128,nominal_target_global_tick=(i+h+1)*128,control_duration_s=.02,control_held_substeps=128,source_interval_pin=prereg['inputs'][f'interval_{i:03d}'],source_json_pointer=f'evaluations[{eidx}].snapshots[{h}]',input_integration_sha256=input_sha,target_integration_sha256=snap['integration_state_sha256'],snapshot_integration_signature=snap['integration_state_signature'],selected_branch=b['branch_index']==interval['selected_branch'],actual_full_H1_confirmation_available=is_confirmed,target_POST_native_time_s=snap['native_time_s'],target_snapshot_force_PRE_time_s=snap['pad_force_pre_time_s'],endpoint_force_scope='Only final solver PRE tick; not target POST geometry or an interval-force trajectory'))
                X.append(x);Y.append(y);U.append(u[h]);D.append(d);endpointFn.append(snap['pad_normal_N']);endpointPREtime.append(snap['pad_force_pre_time_s']);targetquats.append(snap['state']['phone_quaternion_wxyz']);clocks.append([boundarytime[i] if h==0 else float(snapshots[0]['native_time_s']),float(snap['native_time_s'])])
        cohorts.append(dict(cohort_id=f'SOURCE_BOUNDARY_{i:03d}',source_actual_boundary_POST_tick=i*128,source_boundary_native_time_s=boundarytime[i],branch_ids=cohort_branches,sample_ids=cohort_samples,sample_count=len(cohort_samples),source_boundary_digest=source_digest))
    assert branchidorder==list(range(1,162)) and len(X)==322 and confirmed==6 and all(c['sample_count']==46 for c in cohorts)
    X=np.asarray(X);Y=np.asarray(Y);U=np.asarray(U);D=np.asarray(D);assert X.shape==Y.shape==(322,58) and U.shape==(322,20) and D.shape==(322,52)
    np.savez_compressed(OUT/'transitions322_F64.npz',input_state58=X,control20=U,next_state58=Y,physical_world_delta52=D,source_boundary_index=np.asarray([v['source_boundary_index'] for v in identities],dtype=np.int64),branch_id=np.asarray([v['branch_id'] for v in identities],dtype=np.int64),H2_step=np.asarray([v['H2_step'] for v in identities],dtype=np.int64),actual_boundary_qpos27=np.asarray(source_q),actual_boundary_qvel26=np.asarray(source_v),actual_boundary_state58=np.asarray(boundaries),branch_snapshot_state58=np.asarray(snapshotstates),branch_controls2x20=np.asarray(controls),endpoint_last_solver_PRE_pad_normal_N=np.asarray(endpointFn),endpoint_last_solver_PRE_time_s=np.asarray(endpointPREtime),target_source_quaternion_wxyz=np.asarray(targetquats),input_target_native_clock_readback_s=np.asarray(clocks),body_ipos_body_m=ipos)
    save(OUT/'sample_identities322.json',identities);save(OUT/'source_boundary_cohorts7.json',cohorts);save(OUT/'branch_H2_context161.json',branchrows)
    schema=dict(schema='phone_saved_H20ms_transition_dataset:1',state_schema=codec.state.STATE_SCHEMA,state58_fields=codec.state.STATE_FIELDS,state_dtype='float64',control_units='absolute original position-actuator commands, rad;20ms hold128steps',physical_delta52_fields={'dq20_rad':[0,20],'dqd20_rad_s':[20,40],'dp_COM_world_m':[40,43],'Log_Rnext_RcurT_world_rad':[43,46],'dv_COM_world_m_s':[46,49],'domega_world_rad_s':[49,52]},physical_delta52_is_native_WM_chart52=False,rotation_decode='Rnext=Exp(world_delta_rotation) @ Rcur; raw saved R values are not projected',actual_boundary_rotation='F64 quaternion sandwich from saved raw qpos quaternion; no normalization or qpos repair; native snapshot R preserved separately',snapshot_velocity_inverse_scope='World omega is a source capture. When codec inverse reconstructs body omega, it is numerical derived arithmetic, not saved raw native-body bytes.',force_labels='Only snapshot last solver PRE scalar pads and explicit PRE clock. No unchosen per-step contact/force trajectories. H2 extrema stay only in branch-context table.',split_policy='No split assigned. Keep all46 rows in each source-boundary cohort together, including every H2 pair. Never random-row split branches/common boundaries. Cohorts are consecutive states of one actual failure prefix, not independent trajectories.',hidden_context='Warmstart/HISTORY and other integration/solver context not present in state58. Full-clone hash receipts are source-backed, not a state58 Markov certificate.',qualification='Simulator planning exploration and failed-hold coverage only; not real ego dynamics GT, policy performance, model accuracy, independent-trajectory generalization or Markov proof.')
    save(OUT/'schema.json',schema)
    report=dict(schema='phone_saved_transition_dataset_build_report:1',status='COMPLETE_ALL322_SAVED_TRANSITIONS_NOT_DYNAMICS_OR_MARKOV_QUALIFICATION',builder=pin(__file__),codec=pin(H/'transition_codec_v1.py'),preregistration=pin(H/'preregistration_v1.json'),fixture=pin(H/'codec_fixture_v1.json'),parent_report=prereg['inputs']['MPC_report'],parent_saved_review=prereg['inputs']['MPC_saved_review'],outputs={f:pin(OUT/f) for f in ['transitions322_F64.npz','sample_identities322.json','source_boundary_cohorts7.json','branch_H2_context161.json','schema.json']},counts=dict(branches=161,samples=322,source_boundary_cohorts=7,rows_per_cohort=46,actual_confirmed_selected_H1=6,actual_selected_complete_intervals=6,actual_selected_partial_final_steps=54,new_model_FK_LP_physics_NN_training=0),roundtrip=dict(max_state58_COM_codec_abs=float(max(codecerrors)),max_Exp_Log_left_decode_state58_abs=float(max(deltart)),max_raw_rotation_orthogonality_or_det_defect=float(max(Rerrors)),max_world_rotation_step_rad=float(max(angles)),Log_halfturn_branch_margin_rad=1e-6,no_rotation_projection=True),coverage=dict(actual_trace_steps=822,actual_failure_time_s=822*.00015625,planning_branch_steps=161*256,source_boundary_time_span_s=[0.,.12],target_nominal_time_span_s=[.02,.16],input_state_feature_min=X.min(0).tolist(),input_state_feature_max=X.max(0).tolist(),control_feature_min=U.min(0).tolist(),control_feature_max=U.max(0).tolist(),H2_original_horizon_safety_pass=sum(v['original_H2_horizon_safety_pass'] for v in branchrows),all_source_branches_kept=True),checks={k:True for k in ['frozen_sources_parent_pins_and13_saved_proof','all161_cloned_source_boundaries_and_two_snapshots','actual_boundary_first_input_and_same_branch_second_input','all322_source_controls_ticks_and_snapshot_hash_joins','all322_state_COM_codec_and_world_Exp_Log_roundtrip','six_selected_H1_actual_saved_crosschecks','seven_grouped_source_cohorts_no_assigned_random_split','PRE_endpoint_force_and_H2_context_not_fabricated']},state58_Markov_qualified=False,real_ego_dynamics_ground_truth=False,training_or_policy_or_holding_qualified=False,scope=schema['qualification'])
    report['passed']=sum(report['checks'].values());report['total']=len(report['checks']);save(OUT/'report.json',report)
    print(json.dumps({'report':pin(OUT/'report.json'),'counts':report['counts'],'roundtrip':report['roundtrip']}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(H/'build_failure_v1.json',dict(error=repr(e),source=pin(__file__),no_automatic_retry=True));raise
