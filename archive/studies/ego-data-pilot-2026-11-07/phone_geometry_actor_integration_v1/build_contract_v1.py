"""Freeze SOURCE/JSON phone task bindings. No native, Torch, NN or stepping."""
from pathlib import Path
import ast,hashlib,json,math
import numpy as np

H=Path(__file__).resolve().parent;P=H.parent
M=Path(r'D:/workspace/supermimic-chord-study')
UP=Path(r'D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic')
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def rawpin(a):
    a=np.ascontiguousarray(a);return dict(dtype=a.dtype.str,shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
def main():
    preload=P/'functional_tripod_native_preload_v1/completed_run1/report.json';pr=read(preload);c=pr['candidates'][0]
    cp=Path(c['candidate']['path']);candidate=read(cp);probe_path=Path(c['final_repeat_probes'][0]['path']);probe=read(probe_path)
    assert probe_path.name=='probe_030.json' and c['native_loaded_entry'] is True and all(c['fresh_repeat_bit_equal'].values())
    assert candidate['geometry']['passed'] is True and candidate['geometry']['unknown']==[] and candidate['designated_geom_ids']==[15,28,41]
    q=np.array(probe['raw']['qpos27'],dtype=np.float64);v=np.array(probe['raw']['qvel26'],dtype=np.float64);u=np.array(probe['raw']['ctrl20'],dtype=np.float64)
    assert np.array_equal(q,np.array(candidate['qpos27_F64'])) and np.array_equal(v,np.zeros(26)) and probe['measurements']['native_loaded_entry'] is True
    gp=P/'phone_geometry_features_v1/completed_v1/compiled_constants.json';meta=read(gp)
    qr=np.array(meta['joint_range_rad']);ur=np.array(probe['PD']['ctrlrange20x2'])
    lower=np.maximum(qr[:,0],ur[:,0]);upper=np.minimum(qr[:,1],ur[:,1])
    assert np.all(u>=lower) and np.all(u<=upper)
    scale=np.minimum(.05,np.minimum(u-lower,upper-u));assert np.all(scale>=0)
    ipos=np.array(meta['phone_body_ipos']);R=np.array(probe['phone_R_body_world']);COM=np.array(probe['COM_world_m'])
    assert np.max(abs(R.T@R-np.eye(3)))<1e-10
    omega=R@v[23:26];world_v=v[20:23]+np.cross(omega,R@ipos)
    x=np.r_[q[:20],v[:20],COM,R.ravel(),world_v,omega]
    assert np.max(abs(COM-q[20:23]-R@ipos))<1e-12
    template=[]
    for row in np.array(probe['raw']['contact_rows24']):
        g1,g2=map(int,row[1:3])
        if 71 not in [g1,g2]:continue
        hand=g2 if g1==71 else g1
        if hand not in [15,28,41] or row[18]<=.1:continue
        pos=row[6:9];nw=(1. if g2==71 else -1.)*row[9:18].reshape(3,3)[0]
        nb=R.T@nw;nb=nb/np.linalg.norm(nb)
        template.append(dict(hand_geom_id=hand,source_world_contact_position_m=pos.tolist(),point_BODY_COM_m=(R.T@(pos-COM)).tolist(),
            force_on_object_unit_normal_BODY=nb.tolist(),source_decoded_Fn_N=float(row[18]),source_contact_frame_world3x3=row[9:18].reshape(3,3).tolist(),
            scope='Native solver contact position transported as a hypothetical material anchor; not certified filled-surface/nearest point or real ego contact'))
    assert set(v['hand_geom_id'] for v in template)=={15,28,41}
    basis=np.random.default_rng(0).standard_normal((512,6));basis/=np.linalg.norm(basis,axis=1,keepdims=True)
    bp=H/'support_basis512_v1.npz'
    with bp.open('xb') as f:np.savez_compressed(f,basis512_unit6_F64=basis)
    rr=P/'phone_ego_reference_v1/completed_v1/report.json';reference=read(rr)
    refpin=next(v for k,v in reference['outputs'].items() if k.startswith('exact_source__'))
    model=read(P/'functional_tripod_loaded_entry_v1/completed_run1/report.json')['model_mjb']
    sources=dict(task=pin(H/'phone_task_v1.py'),reader_replay=pin(H/'reader_replay_adapter_v1.py'),imagination=pin(H/'phone_actor_imagination_v1.py'),metadata=pin(H/'metadata_preflight_v1.py'),
        native_reader=pin(P/'phone_episode_acquisition_v1/native_phone_reader.py'),world_model=pin(P/'phone_geometry_momentum_wm_v1/train_geometry_momentum_wm.py'),
        structured_codec=pin(P/'phone_structured_transition_v1/structured_codec.py'),directional=pin(P/'phone_directional_gap_features_v1/directional_gap_provider.py'),
        compiled_FK=pin(P/'phone_geometry_features_v1/geometry_provider.py'),reference=pin(P/'phone_ego_reference_v1/reference.py'),
        Actor_Critic=pin(UP/'networks.py'),Actor_common=pin(UP/'common.py'),old_state_not_used=pin(UP/'state.py'),replay=pin(UP/'buffer.py'),upstream_runner=pin(UP/'runner.py'),upstream_pen_mdp=pin(UP/'pen.py'),
        chord_contacts=pin(M/'src/chord_study/contacts.py'),weighted_support=pin(M/'src/chord_study/contact_geometry.py'))
    bundle=dict(schema='phone_fixed_wrist_ego61_tripod_task:1',task_id='ego1856_42_102_exact_GN_probe030_functional_tripod',model=model,sources=sources,
        initial=dict(source_preload_report=pin(preload),source_probe=pin(probe_path),source_candidate=pin(cp),qpos27_F64=q.tolist(),qvel26_F64=v.tolist(),ctrl20_F64=u.tolist(),native_PRE_state58_F64=x.tolist(),
            raw_qpos_qvel_ctrl_receipt=dict(qpos27=rawpin(q),qvel26=rawpin(v),ctrl20=rawpin(u)),
            source_phase=probe['phase'],source_is_static_native_PRE=True,source_full_integration300_receipt_available=False,future_reset_actual_integration_receipt_required=True,
            source_three_pad_initial_Fn_N=probe['pad_normal_N3'],holding_qualified=False),
        joint_range20x2=qr.tolist(),actuator_ctrlrange20x2=ur.tolist(),
        action=dict(kind='bounded_symmetric_absolute_target_residual_with_EMA',normalized_dim=20,nominal_max_residual_rad=.05,alpha=.5,
            symmetric_residual_scale20_rad=scale.tolist(),target_lower20=lower.tolist(),target_upper20=upper.tolist(),
            unavailable_zero_scale_axes=np.flatnonzero(scale==0).tolist(),clamp=False,base_ctrl_is_measured_legal_preload_control=True,encoder_bias=0,noise=False,mask=False),
        reference_report=pin(rr),reference_arrays=refpin,reference_frame_rule='state boundary k -> action interval[k,k+1] -> POST reward against reference k+1, then memory advances',
        source_duration_s=reference['source_duration_s'],duration_s=2.,control_dt_s=.02,control_intervals=100,native_dt_s=[.0003125,.00015625],native_substeps=[64,128],
        task_designated_geom_ids=[15,28,41],model_canonical_pair_geom_ids=[v['other_geom_id'] for v in meta['pairs']],model_training_original_designated_ids=[15,28],
        contact_template=template,basis=pin(bp),support=dict(radius_m=.077,mu_actual_assumed=.7,mu_design=.7/math.sqrt(2),cone_edges=8,proximity_bandwidth_m=.001,basis_count=512,
            reference_kind='engineered static native-loaded tripod capability template, not ego contact/force truth',normal_assumption='Frozen source BODY normals; no closest-normal inference',
            current_point_assumption='same hand material anchors transported by current FK',observed_contact_known=False,force_truth_known=False,friction_certificate=False,
            eight_edge_proxy_is_not_native_condim3_pyramidal_four_edge_solver=True,mass_friction_and_material_parameters_measured=False),
        reward=dict(A1='2exp(-COMerror²/.01²)+8exp(-R_Froberror²/(8sin(10deg/2)²))-.1 action_rate-2 joint_limit; times.02',
            A2='A1+.02*CHORD tolerance-band unit-wrench-support match of engineered material proxy',
            actual_contact_and_force_reward_known=False,no_native_holding_success_reward_claim=True,coefficients_not_tuned_on_future_policy=True),
        actor_obs_dim=222,critic_dim=222,memory_dim=83,policy_observation='privileged simulator state; no RGB estimator/deployment certificate',
        limitations=['Human-left wrist-relative object intent borrowed for robot-right fixed wrist; no full hand/contact retarget.',
            'Current WM fitted original two-pad local states, not these new tripod-loaded dynamics: task/state/contact distribution mismatch.',
            'Current contact proxy may hallucinate capability; unknown actual contact/normal/force masks stay distinct from proxy-known.',
            'state58 omits warmstart/history: no Markov or full-MjData certificate.',
            'Old pen palm-frame/right-Exp state,15self-contact reward and old norm/weights are not compatible.'])
    write(H/'task_bundle_v1.json',bundle)
    protocol=dict(schema='phone_geometry_actor_integration_CPU_contract:1',role='source_phone_task_reference_action_replay_and_engineered_CHORD_bridge_not_trainer',
        CPU_run_authorized=False,Actor_training_authorized=False,NN_forward_authorized=False,native_execution_authorized=False,
        task_bundle=pin(H/'task_bundle_v1.json'),sources=sources,builder=pin(Path(__file__)),
        binding=dict(Actor='Explicit fresh upstream Actor(obs_dim=222); no source Actor weights/norm imported',Critic='fresh Critic(dim=222,amp=False,empirical_norm=True)',
            replay='upstream generic ReplayBuffer,holdout_fraction=0; raw sidecar mandatory before nan_to_num; episode/family split external',
            world_model='Injected frozen new222-output model, same current366 feature implementation; current-only inputs; explicit checkpoint/normalizer/physical context runtime binding required',
            native_reader='Injected original Reader; original2pad summaries preserved; new3pad raw-row regrouping is explicit',
            old_DexSuperMimicRunner_drop_in_compatible=False,new_phone_policy_update_runner_required=True),
        modes=['A1_tracking','A2_engineered_material_proxy'],future_budget='Must freeze same initialization, native sampling, replay, Actor updates, WM updates and evaluation budget for both modes before any training',
        required_runtime_bindings=dict(WM_checkpoint=None,WM_health=None,Actor_training_recipe=None,native_evaluation_protocol=None),
        preconditions=dict(manual_two_second_holding_success_required=False,checkpoint_or_norm_legacy_migration_allowed=False,future_force_labels_as_inputs_allowed=False,
            native_new_task_replay_and_WM_adaptation_before_policy_claim=True,physical_action_gradient_qualified=False),
        evaluation=dict(original_holding_truth_source=pin(P/'functional_tripod_dynamics_v1/run_functional_tripod_dynamics_v3.py'),
            actual_two_seconds_double_dt_noaid_table_nonpad_contact_and_tail_gates_unchanged=True,new_motion_tracking_errors_also_required=True,
            Reader_acquisition_safety_is_not_full_holding_evaluator=True),
        actual_execution=dict(native_model_loads=0,native_FK_forward_steps=0,NN_constructors=0,NN_forward_backward=0,training_updates=0),
        source_contract_status='CALLABLE_SOURCE_ONLY_NOT_RUNTIME_QUALIFIED')
    for f in [H/'phone_task_v1.py',H/'reader_replay_adapter_v1.py',H/'phone_actor_imagination_v1.py',H/'metadata_preflight_v1.py']:ast.parse(f.read_text())
    write(H/'protocol_CPUFalse_v1.json',protocol)
    write(H/'SOURCE_FREEZE_v1.json',dict(schema='phone_geometry_actor_integration_source_freeze:1',protocol=pin(H/'protocol_CPUFalse_v1.json'),sources=sources,builder=pin(Path(__file__))))
    print(json.dumps(dict(protocol=pin(H/'protocol_CPUFalse_v1.json'),bundle=pin(H/'task_bundle_v1.json'),template_contacts=len(template),zero_scale_axes=np.flatnonzero(scale==0).tolist(),native_NN_calls=0)))
if __name__=='__main__':main()
