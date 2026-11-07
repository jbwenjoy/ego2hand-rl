"""Freeze action-probe selection/commands from saved data only; zero native/NN calls."""
from pathlib import Path
from collections import Counter
import hashlib,json,math
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent.parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def checked(r):
    assert pin(r['path'])==r;return Path(r['path'])
def load(r):
    with np.load(checked(r),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
R=P/'phone_episode_acquisition_v1/acquired_v1/report.json';r=read(R)
ids=read(checked(r['sample_identities']));episodes=read(checked(r['episodes']))
plan=read(checked(r['family_plan']));fams=plan['families']
val=load(r['outputs']['validation']);train=load(r['outputs']['train']) # TEST unopened.
vrows={int(s):i for i,s in enumerate(val['row_id_i64'])}
trainmask=train['training_eligible_bool'];train_observable59=np.c_[train['target_delta52_F64'][trainmask],
    train['contact_summary_F64'][trainmask,:2],np.log1p(train['contact_summary_F64'][trainmask,2:])]
std59=train_observable59.std(0,ddof=0)
train_input32=np.c_[train['x58_F64'][trainmask],train['u20_F64'][trainmask],train['native_dt_s_F64'][trainmask],train['n_substeps_i64'][trainmask]].astype(np.float32)
train_input64=np.c_[train['x58_F64'][trainmask],train['u20_F64'][trainmask],train['native_dt_s_F64'][trainmask],train['n_substeps_i64'][trainmask]]
train_keys32={sha(a) for a in train_input32};train_keys64={sha(a) for a in train_input64}
branches=sorted({f['source_branch'] for f in fams});selected=[];operands=[]
for branch in branches:
    fid=min(f['family_id'] for f in fams if f['source_branch']==branch and f['split']=='validation')
    for dt in [.0003125,.00015625]:
        e=next(e for e in episodes if e['family_id']==fid and e['dt_s']==dt);initial=load(e['initial'])
        ranges=np.array(r['model_audits'][str(dt)]['actuator_ctrlrange20x2'])
        for interval in [0,2,5]:
            sid=e['interval_row_ids'][interval];ident=ids[sid];trace=load(ident['trace']);i=vrows[sid]
            assert val['complete_H_bool'][i] and val['training_eligible_bool'][i]
            assert trace['source_integration_state'].shape==(300,) and trace['source_integration_state'].dtype==np.float64
            assert sha(trace['source_integration_state'])==ident['source_receipt']['sha256']
            previous=initial if interval==0 else load(ids[e['interval_row_ids'][interval-1]]['trace'])
            q=initial['qpos27'] if interval==0 else previous['post_qpos27'][-1]
            v=initial['qvel26'] if interval==0 else previous['post_qvel26'][-1]
            u=val['u20_F64'][i].copy();x=val['x58_F64'][i].copy()
            native59=np.r_[val['target_delta52_F64'][i],val['contact_summary_F64'][i,:2],np.log1p(val['contact_summary_F64'][i,2:])]
            proposals=[]
            for j in range(8):
                for sign,name in [(1,'plus'),(-1,'minus')]:
                    margin=float(ranges[j,1]-u[j] if sign>0 else u[j]-ranges[j,0]);h=min(.01,.5*margin)
                    trial=u.copy();trial[j]+=sign*h;du=float(trial[j]-u[j])
                    available=bool(margin>=0 and abs(du)>=1e-6 and np.all(trial>=ranges[:,0]) and np.all(trial<=ranges[:,1]))
                    u32=u.astype(np.float32);trial32=trial.astype(np.float32);du32=float(np.float64(trial32[j])-np.float64(u32[j]))
                    inp64=np.r_[x,trial,dt,round(.02/dt)];inp32=inp64.astype(np.float32)
                    proposals.append(dict(direction_joint=j,side=name,sign=sign,nominal_step_rad=.01,margin_to_hard_bound_rad=margin,
                        requested_side_h_rad=h,actual_signed_du_F64_rad=du,available=available,
                        unavailable_reason=None if available else 'ACTUAL_F64_STEP_BELOW_1e-6_OR_BOUND_VIOLATION',
                        actual_ctrl20_F64=trial.tolist(),ctrl_raw_F64_sha256=sha(trial),
                        base_ctrl20_FP32=u32.tolist(),actual_ctrl20_FP32=trial32.tolist(),actual_signed_du_FP32_rad=du32,
                        FP32_input_collapsed_numeric=bool(du32==0),FP32_du_minus_F64_du_rad=du32-du,
                        FP32_to_F64_step_ratio=None if du==0 else du32/du,
                        exact_TRAIN_input_F64_alias=sha(inp64) in train_keys64,exact_TRAIN_input_FP32_alias=sha(inp32) in train_keys32))
            assert not any(p['exact_TRAIN_input_F64_alias'] or p['exact_TRAIN_input_FP32_alias'] for p in proposals if p['available'])
            k=len(selected)
            selected.append(dict(state_id=k,source_branch=branch,family_id=fid,episode_id=e['episode_id'],split='validation',
                interval_index=interval,source_sample_id=sid,native_dt_s=dt,substeps=round(.02/dt),H_s=.02,
                source_trace=ident['trace'],source_before_action_receipt=ident['source_receipt'],
                original_source_after_action_receipt=ident['source_after_action_receipt'],original_target_receipt=ident['target_receipt'],
                original_episode_initial=e['initial'],original_episode_initial_receipt=e['initial_receipt'],
                source_identity=ident,baseline_ctrl20_F64=u.tolist(),baseline_ctrl_raw_F64_sha256=sha(u),
                original_hard_stop_baseline=dict(initial_qpos27_F64=initial['qpos27'].tolist(),
                    initial_COM_world3_F64=initial['state58'][40:43].tolist(),initial_quaternion_wxyz_F64=initial['qpos27'][23:27].tolist()),
                perturbations=proposals))
            operands.append(dict(source_integration=trace['source_integration_state'],source_qpos27=q,source_qvel26=v,
                source_state58=x,baseline_ctrl20=u,baseline_observable59=native59,
                original_target_integration=trace['target_integration_state'],original_target_qpos27=trace['post_qpos27'][-1],
                original_target_qvel26=trace['post_qvel26'][-1],initial_qpos27=initial['qpos27'],initial_state58=initial['state58']))
assert len(selected)==12 and Counter(s['substeps'] for s in selected)=={64:6,128:6}
assert sorted({s['family_id'] for s in selected})==[6,30]
arrays={k+'_F64':np.stack([s[k] for s in operands]) for k in operands[0]}
arrays.update(TRAIN_observable59_population_std_F64=std59,TRAIN_observable59_population_mean_F64=train_observable59.mean(0),
    TRAIN_observable59_nonconstant_bool=std59>0)
npz=H/'selected_operands_v1.npz'
with npz.open('xb') as f:np.savez_compressed(f,**arrays)
selection=H/'selection_v1.json'
write(selection,dict(schema='phone_action_probe_selection:2',status='FROZEN_BEFORE_NEW_NATIVE_OR_NN_EXECUTION',source=pin(Path(__file__)),
    states=selected,operands=pin(npz),TRAIN_partition=r['outputs']['train'],TEST_NPZ_reads=0,
    available_sides=sum(p['available'] for s in selected for p in s['perturbations']),
    unavailable_sides=sum(not p['available'] for s in selected for p in s['perturbations'])))
health=read(P/'phone_world_model_train_v1/cpu_run1/checkpoint_5000_health.json')
contract=dict(schema='phone_action_gradient_probe_design:2',status='FROZEN_SAMPLING_AND_EVALUATION_NO_NEW_NATIVE_NN_OR_TRAINING',
    source=pin(Path(__file__)),selection=pin(selection),selected_operands=pin(npz),dataset_report=pin(R),
    dataset_saved_independent=pin(P/'phone_episode_acquisition_v1/independent_dataset_review_v1/report.json'),
    family_plan=r['family_plan'],sample_identities=r['sample_identities'],episode_manifest=r['episodes'],
    fixed_checkpoint_metadata=pin(P/'phone_world_model_train_v1/cpu_run1/checkpoint_5000_health.json'),
    fixed_training_result=pin(P/'phone_world_model_train_v1/cpu_run1/report.json'),fixed_checkpoint=health['checkpoint'],
    model_parameter_audits=r['model_audits'],
    sampling=dict(rule='For each source choose minimum VAL familyID, bothdt, intervals0/2/5; no outcome/error-driven choice.',
        family_ids=[6,30],states=12,directions=list(range(8)),nominal_side_step_rad=.01,
        side_h_rule='hp=min(.01,.5*(ctrlupper-u)); hm=min(.01,.5*(u-ctrllower)); compute each independently. Actualdu is F64 endpoint subtraction. Side abs(actualdu)<1e-6 unavailable, retain reason; never clip and never shrink feasible opposite side to match.',
        baseline_per_state=1,maximum_perturbed_branches=192,maximum_total_branches=204,
        maximum_native_integration_steps=19584,actual_planned_available_sides=sum(p['available'] for s in selected for p in s['perturbations']),
        maximum_planned_native_steps=sum(s['substeps']*(1+sum(p['available'] for p in s['perturbations'])) for s in selected),
        symmetry='Primary is separate one-sided finite-action response. Do not silently call unequal hp/hm a centered derivative. No three-point interpolation is prescribed.'),
    restore_and_replay=dict(source='Raw source_integration_F64 before the original interval action, full300 mjSTATE_INTEGRATION/signature16383, actual native time bits retained.',
        preserve='Restore ALL integration state, current nonzero qvel, previouscontrol/warmstart/history, globaltick/time; do not recreate from state58, zero current velocity, or add a forward that changes context.',
        safety_reference='Every branch uses original episode initial qpos/COM/quaternion from selected_operands; current probe state is not a new initial condition.',
        original_safety=dict(COM_drift_m=.15,angle_rad=math.pi/2,penetration_m=.005,joint_excursion_rad=.05,nonfinite=True,native_warning=True),
        baseline_gate='First run all12 original-u baselines and reproduce original source/after-action/target receipts, POST qpos/qvel and complete H labels. Any failure blocks new action response qualification; preserve failure/no automatic retry.',
        baseline_numeric='Raw state/receipt/qpos/qvel/ctrl use exact F64 bytes. Native contact summaries/59 derived arithmetic use1e-10 absolute+relative solely as saved arithmetic agreement, never a dynamics accuracy gate.',
        partial='Keep all attempted branch prefixes, full raw receipts/contacts/actualtimes/termination. A partialH has NaN H target59/state58/delta52; observed prefix stays separately. No model-vs-short-prefix endpoint comparison.',
        eligible_comparison='Baseline replay passes AND perturbed branch has complete64/128 H, finite state/contacts and no warning. Preserve other branches with masks/reasons, no success-only reporting.'),
    observations=dict(observable59='physical_world_delta52 + padcoverage2 + log1p(force5/1N)',
        delta52='q20 rad/qd20 rad_s/pCOM_world3 m/Log(Rtarget Rsource.T)_world3 rad/vCOM_world3 m_s/omega_world3 rad_s',
        force5_order='pad_min_normal2,pad_lastPRE_normal2,max_phone_nonpad_normal1; all positive native-force sums; force_ref=1N',
        coverage='Strict per-PRE-tick pad positive-normal sum>0.1N; coverage is a fraction, not contact force or cachedself15.',
        time='POST H endpoints; each force label aggregates full64/128 solver-PRE ticks; lastPRE=H-native_dt.',
        extra='Save predicted and native full body-to-world R9 and COM3/velocity/omega. Do not collapse orientation to one yaw, project native matrices or substitute inertial axes.'),
    comparisons=dict(primary='For every available side: native_s=(Ynative_side-Ynative_base)/du_F64; model_s_same_step=(Ymodel_side-Ymodel_base)/du_F64. Same state and actualF64 action endpoints; native F64, model input/output FP32 then castF64 for subtraction.',
        model_AD='Parameters/frozen buffers fixed; action-only AD at original state/action FP32. g_j=dYmodel59/du_j in physical output units; includes normalization chain, sigmoid coverage and physical logforce output.',
        own_model_AD_check='ModelFD=(Ymodel_side-Ymodel_base)/du_FP32actual; compare g_j to this finite step, not a claimed exact derivative of physics. du_FP32 is cast endpoint subtraction in F64. If0, undefined/null+mask, retain values.',
        quantization='Save full endpoint/baseFP32 command bytes, signeddu_F64/signeddu_FP32, ratio anddifference. Any rounding/collapse is reported; no hidden physical-vs-model precision equivalence.',
        original_unit_blocks={'hand_q20':[0,20],'hand_qd20':[20,40],'COM3':[40,43],'world_rotvec3':[43,46],
            'vCOM3':[46,49],'omega_world3':[49,52],'coverage2':[52,54],'logforce5':[54,59]},
        statistics='All12states/8directions/2sides retained. Perblock/perfeature signed bias, MAE, componentRMSE, vectorRMS; native/model secant norms and norm error. Cosine only if both norms>0, otherwise null/mask. No mixed-unit59 Euclidean score as sole conclusion.',
        TRAIN_normalization='Additional perfeature errors divided by population std of59 from original684 eligible TRAIN rows only; frozen in operands. std==0 gives null/mask for normalized value; all raw-unit values remain. Never refit using VAL/native newbranch/TEST.',
        contact_changes='Save baseline/side per-PRE tick contact geom/active signatures and loaded/unloaded switches. Crossing manifolds yields finite-action secants; learned AD is not true contact derivative/Jacobian.',
        aggregate_cohorts='ALL/sourcebranch/dt/sourceinterval/direction/side; this is exploratory VAL local analysis with repeated sources, not independent trials/CI or unseen real-ego generalization.'),
    failure_gates=dict(engineering_only=['source/model/checkpoint/schema/pin mismatch','restore/baseline replay mismatch','unauthorized ctrl clamp or out-of-range command',
        'changed model/material/gains/solver or original safety baseline','incorrect step/time/phase/targetmask','unexpected FW/AD/step budgets','nonfinite model prediction or AD'],
        native_partial_or_warning='Preserved outcome with comparison maskFalse, no fabricated H target; no resampling substitute.',
        derivative_or_accuracy='No pass threshold, sign filter, noise floor, weight tuning, checkpoint selection, retraining, policy improvement, holding success or true-Jacobian qualification is defined.'),
    fixed_scope=dict(new_training_or_optimizer_updates=0,TEST_NPZ_reads=0,actual_new_native_model_FW_AD_calls_in_design=0,
        context='Two mapped poses from oneego frame under assumed simulator parameters; validation used during training, so this is not a new blind confirmation. No J/K truth or independent real contact-force measurement.'))
out=H/'design_contract_v1.json';write(out,contract)
print(json.dumps({'design_contract':pin(out),'selection':pin(selection),'operands':pin(npz),
    'available_sides':contract['sampling']['actual_planned_available_sides'],'max_native_steps':contract['sampling']['maximum_planned_native_steps'],
    'new_native_NN_training_calls':0,'TEST_NPZ_reads':0}))
