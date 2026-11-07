"""Pure saved-JSON planning only: freeze48 local simulation families before native use."""
from pathlib import Path
import hashlib,json,math,struct
H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)
def raw(x):
    return b''.join(raw(v) for v in x) if isinstance(x,list) else struct.pack('<d',float(x))
def raw_sha(x):return hashlib.sha256(raw(x)).hexdigest()
def unit(role,seed,draw,attempt=0):
    token=f'phone_episode_acquisition_v1:{role}:{seed}:{attempt}:{draw}'
    z=int.from_bytes(hashlib.sha256(token.encode('ascii')).digest()[:8],'big')>>11
    return (z+.5)/2**53
source_names=['exact_source_original_hand','source1mm_opposed_side_original_hand']
profiles=['mixed20_slow','thumb_index_opposed','unipolar_q8_mixed12']
source=[]
for b in source_names:
    path=P/'surface_assembly_v11'/f'{b}_completed_v1/seed1_rank0.json'
    r=read(path);n=r['native_probe'];pd=n['contact_design']['PD_raw']
    assert r['seed']==1 and r['rank']==0 and r['candidate']['assembly_gate'] and r['final_geometry']['geometry_gate_pass']
    assert n['unknown']==[] and not any(n['warnings_number'])
    assert raw(n['raw']['qpos27'])==raw(r['candidate']['kinematics']['qpos27']) and not any(n['raw']['qvel26'])
    source.append(dict(branch=b,record=pin(path),qpos=n['raw']['qpos27'],ctrl=n['raw']['ctrl20'],
        joint_range=pd['joint_range20x2'],ctrl_range=pd['actuator_ctrlrange20x2'],model_digest=n['model_parameter_digest'],
        source_initial_pad_normal_N=[n['native_contact_aggregates'][k]['normal_force_signed_N'] for k in ['thumb_pad','index_pad']],
        source_ideal_native_support_candidate=r['ideal_native_support_candidate'],source_ctrl_origin=r['initial_native_probe_ctrl_source']))
assert raw(source[0]['joint_range'])==raw(source[1]['joint_range'])
assert raw(source[0]['ctrl_range'])==raw(source[1]['ctrl_range'])
assert source[0]['model_digest']==source[1]['model_digest']
families=[];rejections=[]
for bi,s in enumerate(source):
    for pi,profile in enumerate(profiles):
        for rep in range(8):
            fid=bi*24+pi*8+rep
            initseed=1100001+bi*100000+pi*1000+rep*10
            ctrlseed=2100002+bi*100000+pi*1000+rep*10
            split='train' if rep<6 else ('validation' if rep==6 else 'test')
            for attempt in range(64):
                dq=[.005*(2*unit('initial',initseed,j,attempt)-1) for j in range(8)]
                q=s['qpos'].copy();q[:8]=[q[j]+dq[j] for j in range(8)]
                invalid=[j for j in range(20) if not s['joint_range'][j][0]<=q[j]<=s['joint_range'][j][1]]
                if not invalid:break
                rejections.append(dict(family_id=fid,initialization_seed=initseed,attempt=attempt,
                    proposed_delta_q8_rad=dq,rejected_joint_indices=invalid,reason='OUTSIDE_ORIGINAL_HARD_JOINT_RANGE_NO_CLIP'))
            else:raise ValueError('Initial proposal exhausted; no replacement family or native execution')
            requested=[]
            for k in range(10):
                delta=[]
                for j in range(20):
                    amp=.05 if j<8 else .025
                    if pi==0:
                        z=2*unit('control',ctrlseed,(k//2)*20+j)-1
                    elif pi==1 and j<8:
                        z=(2*unit('control',ctrlseed,400+k//2)-1)*(1 if j<4 else -1)
                    elif pi==2 and j<8:
                        sign=1 if unit('control',ctrlseed,500+j)>.5 else -1
                        z=sign*unit('control',ctrlseed,600+(k//2)*8+j)
                    else:z=2*unit('control',ctrlseed,800+(k//2)*20+j)-1
                    delta.append(amp*z)
                requested.append(delta)
            alpha=[]
            for j in range(20):
                lo,hi=s['ctrl_range'][j];u=s['ctrl'][j]
                factors=[1.]
                for row in requested:
                    d=row[j]
                    if d>0:factors.append((hi-u)/d)
                    elif d<0:factors.append((lo-u)/d)
                # Uniform scaling of this actuator's entire10-interval curve; never samplewise clamp.
                a=min(factors)
                if a<1:a=max(0.,a)*(1-1e-12)
                alpha.append(a)
            actions=[[s['ctrl'][j]+alpha[j]*row[j] for j in range(20)] for row in requested]
            assert all(lo<=u<=hi for row in actions for u,(lo,hi) in zip(row,s['ctrl_range']))
            assert raw(q[8:20])==raw(s['qpos'][8:20]) and raw(q[20:27])==raw(s['qpos'][20:27])
            assert max(abs(q[j]-s['qpos'][j]) for j in range(8))<=.005
            assert all(abs(row[j]-s['ctrl'][j])<=(.05 if j<8 else .025)+1e-15 for row in actions for j in range(20))
            families.append(dict(family_id=fid,source_branch=s['branch'],source_record=s['record'],
                source_seed=1,source_rank=0,split=split,profile=profile,replicate=rep,
                initialization_seed=initseed,control_seed=ctrlseed,initial_attempt=attempt,
                initial_delta_q8_rad=dq,initial_qpos27_F64=q,initial_qvel26_F64=[0.]*26,
                base_ctrl20_F64=s['ctrl'],requested_delta_u10x20_F64=requested,
                whole_curve_scale20_F64=alpha,held_ctrl10x20_F64=actions,
                initial_qpos27_raw_F64_sha256=raw_sha(q),held_ctrl10x20_raw_F64_sha256=raw_sha(actions),
                native_dt_s=[.0003125,.00015625],substeps_per_H=[64,128]))
assert len(families)==48 and [sum(f['split']==s for f in families) for s in ['train','validation','test']]==[36,6,6]
assert len({f['initialization_seed'] for f in families})==len({f['control_seed'] for f in families})==48
assert len({f['initial_qpos27_raw_F64_sha256'] for f in families})==48
assert len({f['held_ctrl10x20_raw_F64_sha256'] for f in families})==48
plan=dict(schema='phone_episode_family_plan:1',status='FROZEN_BEFORE_NEW_NATIVE_EXECUTION',
    builder=pin(Path(__file__)),source_records=[s['record'] for s in source],families=families,
    rejected_initial_proposals=rejections,split_counts={'train':36,'validation':6,'test':6},
    H_s=.02,intervals_per_episode=10,native_dt_s=[.0003125,.00015625],episode_count=96,
    maximum_complete_H_rows=960,maximum_native_integration_steps=92160,
    actual_new_native_model_FK_forward_step_training_calls=0)
planpath=H/'family_plan_v1.json';write(planpath,plan)
coverage=[]
for s in source:
    ff=[f for f in families if f['source_branch']==s['branch']]
    coverage.append(dict(source_branch=s['branch'],source_record=s['record'],
        minimum_source_q8_hard_range_margin_rad=min(min(q-lo,hi-q) for q,(lo,hi) in zip(s['qpos'][:8],s['joint_range'][:8])),
        source_ctrl_positive_margin20_rad=[hi-u for u,(lo,hi) in zip(s['ctrl'],s['ctrl_range'])],
        source_ctrl_negative_margin20_rad=[u-lo for u,(lo,hi) in zip(s['ctrl'],s['ctrl_range'])],
        planned_delta_u_min20_rad=[min(row[j]-s['ctrl'][j] for f in ff for row in f['held_ctrl10x20_F64']) for j in range(20)],
        planned_delta_u_max20_rad=[max(row[j]-s['ctrl'][j] for f in ff for row in f['held_ctrl10x20_F64']) for j in range(20)],
        scaled_family_count20=[sum(f['whole_curve_scale20_F64'][j]<1 for f in ff) for j in range(20)]))
contract=dict(schema='phone_episode_acquisition_design:1',status='FROZEN_SOURCE_AND_PLANNED_COMMANDS_NOT_ACQUIRED',
    source=pin(Path(__file__)),family_plan=pin(planpath),
    source_inputs=dict(selected_v11_CPU_protocol=pin(P/'hand2_source_mapping_dynamics_v11/protocol_CPUTrue_v1.json'),
        selected_v11_metadata=pin(P/'hand2_source_mapping_dynamics_v11/metadata_summary_v11.json'),
        state58_codec=pin(P/'phone_state_time_contract_v1/state_contract.py'),
        physical_delta52_codec=pin(P/'phone_transition_dataset_v1/transition_codec_v1.py')),
    source_records=[s['record'] for s in source],model_parameter_digest=source[0]['model_digest'],
    actual_limits=dict(joint_range_source_key='native_probe.contact_design.PD_raw.joint_range20x2',
        control_range_source_key='native_probe.contact_design.PD_raw.actuator_ctrlrange20x2',
        both_source_records_exact_F64_bytes_equal=True,joint_range20x2_F64=source[0]['joint_range'],
        actuator_ctrlrange20x2_F64=source[0]['ctrl_range']),
    budget=dict(families=48,families_per_source=24,episodes_per_family=2,episodes=96,
        H_s=.02,max_intervals_per_episode=10,max_duration_s=.2,native_dt_s=[.0003125,.00015625],
        substeps_per_H=[64,128],max_complete_H_rows=960,max_native_integration_steps=92160,
        models_physics_acquisition_authorized_by_this_document=False,new_training_authorized=False),
    initialization=dict(active_perturbation_joint_indices=list(range(8)),max_abs_delta_rad=.005,
        fixed_initial_joint_indices=list(range(8,20)),phone_root_qpos20_27_unchanged=True,zero_qvel26=True,
        seed_roles_separate=True,max_attempts_per_family=64,range_policy='Reject entire proposed q8 if any original hard range fails; append all rejected values/reasons. Never clip; exhaustion blocks family acquisition.',
        reset='Fresh native integration context per episode, same literal initial qpos27/base_ctrl20 for bothdt; do not inherit previous family or planner warmstart/history.',
        no_contact_selection='Reset/source contact may unload, penetrate, or have other contacts. Preserve all96 episode records; never resample or drop on observed contact/holding.'),
    actions=dict(source='Literal family_plan_v1 held_ctrl10x20_F64 is authoritative; no runtime RNG or online policy.',
        primary_indices=list(range(8)),primary_max_abs_delta_rad=.05,other_indices=list(range(8,20)),other_max_abs_delta_rad=.025,
        profiles=profiles,held_duration_s=.02,slow_knot_duration_s=.04,
        scaling='For each actuator j scale its entire10-entry requested delta curve by one alpha_j=min(1, all legal positive/negative margin ratios); if reduced multiply by1-1e-12. Retain requested curves andalpha. Add to original source ctrl; no per-time clipping.',
        generator='SHA256 phone_episode_acquisition_v1:{role}:{seed}:{attempt}:{draw}; first8bytes big-endian high53bits -> (integer+.5)/2^53; all planned additions/coefficients F64.',
        controls_identical_F64_bytes_between_paired_dt=True,initialization_identical_F64_bytes_between_paired_dt=True,
        input_controls_are_position_PD_commands_not_torques=True,post_reset_other12_joint_motion_is_not_frozen=True),
    split=dict(unit='family; includes bothdt and all intervals/partial traces',counts={'train':36,'validation':6,'test':6},
        per_source_counts={'train':18,'validation':3,'test':3},rule='Each source has3profiles x8replicates; replicate0..5 TRAIN,6 VALIDATION,7 TEST.',
        frozen_before_native=True,random_row_split_forbidden=True,
        exported_arrays='Three independently pinned NPZ files TRAIN/VALIDATION/TEST; trainer opens only TRAIN+VALIDATION. TEST arrays stay sealed until explicit evaluation; hashing filebytes/header provenance is not permission to inspect values.',
        alias_policy='Before fitting, audit full integration source-after-action+target receipts, command rawbytes, H, native_dt/nsubsteps, model+state-spec/dtype. Union exact aliases across families. If any component crosses frozen partitions, retain all rows and mark partition_qualifiedFalse; no silent reassignment or deduplication.'),
    safety=dict(nonfinite=True,any_native_warning=True,COM_drift_from_episode_initial_m=.15,
        orientation_from_episode_initial_rad=math.pi/2,max_native_penetration_m=.005,
        max_joint_limit_excursion_rad=.05,threshold_operator='Strict greater than; finite/warning checked every native step.',
        holding_failure_is_not_data_filter=True,old_holding_50mm_45deg_gates_are_not_acquisition_abort_gates=True,
        partial='Abort whole episode at first safety trigger; save triggering step/rawcontacts and complete partial interval prefix. No synthetic target at H and no extending/interpolating.',
        training_row_mask='complete_H AND all raw states/controls finite AND no native warning during interval. Keep every row and mask/reason; safety failure at a full-H boundary is separately recorded, not silently excluded beyond this exact mask.'),
    reader=dict(proposed_API='NativePhoneReader(model_path, expected_model_sha256, dt_s); reset(qpos27,base_ctrl20)->snapshot+receipt; step_control(ctrl20,max_substeps=None)->interval dict',
        H_contract='Exactly64 or128 integral steps; POST source/target at boundaries, native contacts/force/energy tagged PRE tick. Last PRE time=target_time-native_dt.',
        state_schema='phone_COM_world_state58:1',state_order='q20/qd20/pCOM_world3/Rbody_to_world_rowmajor9/vCOM_world3/omega_world3',
        body_COM='pCOM=rootp+Rbody*body_ipos; vCOM=vroot+omega_world cross Rbody*body_ipos. Rbody is not inertial-axis rotation.',
        delta52_order='dq20/dqd20/dpCOM_world3/Log(Rtarget Rsource.T)_world3/dvCOM_world3/domega_world3',
        SO3_scope='Use pinned pure saved codec; retain raw native R and log/Exp roundtrip diagnostics without old native pen chart/norm/checkpoint migration.',
        no_external_aid='Fixed original MJB/material/gains/solver. No external object force, pose reset or state overwrite inside episode; only prescribed ctrl20 updates.'),
    receipts=dict(full_state='Raw F64 mjSTATE_INTEGRATION vector, actual state-spec integer, mj_stateSize, dtype/shape/rawSHA, model SHA/effective parameter digest includingdt. Save vectors, not only booleans/digests.',
        initial='Full reset integration receipt per episode plus qpos/qvel/ctrl/time and initial contact observations; do not discard initially unloaded episode.',
        per_interval='source before action, source after setting held ctrl and before first step, and target integration receipts; actual held u20 rawbytes, integer tick/time, full POST raw state, phase-specific contact trace, complete/partial/abort reasons.',
        provenance='Every episode/interval carries family/sourcebranch/source record pin, initialization/control seeds, split, profile, dt/nsubsteps and action-plan indices.',
        Markov_scope='Full receipt aids replay/alias matching. It does not prove state58 contains warmstart/history/hidden solver context or is Markov.'),
    arrays=dict(schema='phone_episode_H20ms_arrays:1',split_files=['train.npz','validation.npz','test.npz'],
        keys={'x58_F64':'[N,58] source boundary POST', 'u20_F64':'[N,20] held actual PD command',
            'y58_F64':'[N,58] target boundary POST; partial rows use NaN target and complete_HFalse',
            'target_delta52_F64':'[N,52] physical world tangent; complete rows only, partial NaN',
            'native_dt_s_F64':'[N] actual integration timestep; this plus n_substeps forms optional80-input conditioning',
            'n_substeps_i64':'[N]64/128 requested fullH', 'executed_substeps_i64':'[N] actual completed prefix',
            'complete_H_bool':'[N]', 'training_eligible_bool':'[N] exact mask above',
            'row_id_i64':'[N] global stable ID including partial records', 'family_id_i64':'[N]',
            'episode_id_i64':'[N]', 'interval_index_i64':'[N]',
            'contact_summary_F64':'[N,7]: coverage_pad2,minFn_pad2,endpoint_lastPRE_Fn_pad2,max_nonpadFn; partial exactprefix summaries descriptive and maskFalse'},
        contact_summary_definition='Each PRE step perpad Fn=sum(max(native decoded contact.force6[0],0)) over generated phone-pad pairs; coverage uses strict Fn>0.1N. min and lastPRE use those positive sums. max_nonpadFn=sum positive phone-other except designatedpads andtable; table is a separate raw/summary field. Preserve signed raw force6/contact geomids andtableFn independently; no manual efc/pyramid re-decode.',
        identifiers='sample_identities.json includes receipt hashes/pins and masks/reasons for ALL intervals, including partial; episode manifest includes all96 even zero complete rows.'),
    planned_coverage=coverage,
    coverage_assessment=dict(required='Report actual completed/partial by source,split,dt,profile; initial0/1/2loadedpads; no/one/two-pad contact occupancy, table/other contacts, command ranges/covariance/effective scales, time-to-unload/stop. No selection based on these outcomes.',
        conclusion='20 input commands vary but independent rank/excitation/contact coverage is not guaranteed. Paired dt are repeats withinfamily, not independent samples. Near-hard-bound 1mm index commands constrain positive excitation; unipolar profiles preserve permitted negative range.',
        population='Local simulated dynamics around two algorithmic mappings of the same source ego frame, under assumed phone/hand/contact parameters. Not independent ego trajectories, measured real contact dynamics GT, robust holding success, or policy/true-Jacobian qualification.'),
    planned_prechecks=dict(family_count48=True,distinct_initial_qpos_hashes48=True,distinct_control_curve_hashes48=True,
        split36_6_6=True,source_limits_match=True,initials_inside_true_joint_ranges=True,
        all_planned_controls_inside_true_ctrl_ranges=True,rejected_initial_proposal_count=len(rejections)),
    actual_new_native_model_FK_forward_step_training_calls=0)
out=H/'design_contract_v1.json';write(out,contract)
print(json.dumps({'contract':pin(out),'family_plan':pin(planpath),'families':len(families),'initial_rejections':len(rejections),'native_calls':0}))
