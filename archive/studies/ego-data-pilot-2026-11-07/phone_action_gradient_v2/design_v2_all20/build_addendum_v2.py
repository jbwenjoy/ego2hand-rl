"""Saved JSON-only all20 extension; original8 definitions and operands remain frozen."""
from pathlib import Path
import copy,hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent.parent;V1=H.parent/'design_v1'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def json_sha(x):return hashlib.sha256(json.dumps(x,separators=(',',':'),allow_nan=False).encode()).hexdigest()
old=read(V1/'selection_v1.json');contract=read(V1/'design_contract_v1.json')
assert pin(V1/'selection_v1.json')['sha256']=='7c0b01967860bde659cd185a6b1a6cf617c256ef407f6e6bded31c143cd7e6a3'
assert pin(V1/'design_contract_v1.json')['sha256']=='a761f42f259a63e7705a1360c1481a3b707c1f4be50d80b8907f7ffdb37fbae0'
assert pin(V1/'selected_operands_v1.npz')==old['operands'] # Hash only; no data/model calls.
r=read(contract['dataset_report']['path']);assert pin(contract['dataset_report']['path'])==contract['dataset_report']
with np.load(r['outputs']['train']['path'],allow_pickle=False) as z:
    assert pin(r['outputs']['train']['path'])==r['outputs']['train'];m=z['training_eligible_bool']
    x64=np.c_[z['x58_F64'][m],z['u20_F64'][m],z['native_dt_s_F64'][m],z['n_substeps_i64'][m]]
    trainkeys64={sha(a) for a in x64};trainkeys32={sha(a) for a in x64.astype(np.float32)}
# Source states are stored as literal trace arrays/metadata in v1. Reuse the frozen operand package.
with np.load(V1/'selected_operands_v1.npz',allow_pickle=False) as z:states58=z['source_state58_F64'].copy()
states=copy.deepcopy(old['states']);preservation=[]
for s,original in zip(states,old['states']):
    u=np.asarray(s['baseline_ctrl20_F64'],dtype=np.float64)
    ranges=np.asarray(r['model_audits'][str(s['native_dt_s'])]['actuator_ctrlrange20x2'],dtype=np.float64)
    for j in range(8,20):
        for sign,name in [(1,'plus'),(-1,'minus')]:
            margin=float(ranges[j,1]-u[j] if sign>0 else u[j]-ranges[j,0]);h=min(.01,.5*margin)
            trial=u.copy();trial[j]+=sign*h;du=float(trial[j]-u[j]);keep=np.arange(20)!=j
            assert u[keep].tobytes()==trial[keep].tobytes(),'Other19 commands must remain exactbytes'
            available=bool(margin>=0 and abs(du)>=1e-6 and np.all(trial>=ranges[:,0]) and np.all(trial<=ranges[:,1]))
            u32=u.astype(np.float32);trial32=trial.astype(np.float32);du32=float(np.float64(trial32[j])-np.float64(u32[j]))
            inp64=np.r_[states58[s['state_id']],trial,s['native_dt_s'],s['substeps']];inp32=inp64.astype(np.float32)
            s['perturbations'].append(dict(direction_joint=j,side=name,sign=sign,nominal_step_rad=.01,margin_to_hard_bound_rad=margin,
                requested_side_h_rad=h,actual_signed_du_F64_rad=du,available=available,
                unavailable_reason=None if available else 'ACTUAL_F64_STEP_BELOW_1e-6_OR_BOUND_VIOLATION',
                actual_ctrl20_F64=trial.tolist(),ctrl_raw_F64_sha256=sha(trial),
                base_ctrl20_FP32=u32.tolist(),actual_ctrl20_FP32=trial32.tolist(),actual_signed_du_FP32_rad=du32,
                FP32_input_collapsed_numeric=bool(du32==0),FP32_du_minus_F64_du_rad=du32-du,
                FP32_to_F64_step_ratio=None if du==0 else du32/du,
                exact_TRAIN_input_F64_alias=sha(inp64) in trainkeys64,exact_TRAIN_input_FP32_alias=sha(inp32) in trainkeys32))
    assert len(s['perturbations'])==40 and s['perturbations'][:16]==original['perturbations']
    assert {k:v for k,v in s.items() if k!='perturbations'}=={k:v for k,v in original.items() if k!='perturbations'}
    assert not any(p['exact_TRAIN_input_F64_alias'] or p['exact_TRAIN_input_FP32_alias'] for p in s['perturbations'][16:] if p['available'])
    preservation.append(dict(state_id=s['state_id'],original8_objects_identical=True,
        original8_canonical_JSON_sha256=json_sha(original['perturbations']),copied8_canonical_JSON_sha256=json_sha(s['perturbations'][:16]),
        unchanged_state_baseline_receipts_safety_and_pins=True,added12_nondirection19_ctrl_F64_bytes_identical=True))
available8=sum(p['available'] for s in states for p in s['perturbations'][:16]);available12=sum(p['available'] for s in states for p in s['perturbations'][16:])
assert available8==188 and available12==288
maximum_steps=sum(s['substeps']*(1+sum(p['available'] for p in s['perturbations'])) for s in states)
assert maximum_steps==46848
selection=dict(schema=old['schema'],status='FROZEN_ALL20_EXTENSION_BEFORE_NEW_NATIVE_OR_NN_EXECUTION',
    source=pin(Path(__file__)),parent_selection=pin(V1/'selection_v1.json'),parent_design_contract=pin(V1/'design_contract_v1.json'),
    states=states,operands=old['operands'],TRAIN_partition=old['TRAIN_partition'],TEST_NPZ_reads=0,
    available_sides=available8+available12,unavailable_sides=480-available8-available12,
    direction_cohorts=dict(primary_original8=list(range(8)),secondary_added12=list(range(8,20)),all20=list(range(20))))
sp=H/'selection_v2_all20.json';write(sp,selection)
addendum=dict(schema='phone_action_gradient_probe_all20_addendum:2',status='FROZEN_BEFORE_NEW_PHYSICS_OR_MODEL_AD',
    source=pin(Path(__file__)),parent_design_contract=pin(V1/'design_contract_v1.json'),parent_selection=pin(V1/'selection_v1.json'),
    reused_operands=old['operands'],selection=pin(sp),preservation=preservation,
    sampling=dict(states=12,baseline_per_state=1,original8_available_sides=188,added12_available_sides=288,
        total_available_sides=476,total_unavailable_sides=4,total_branches=488,max_native_steps=46848,
        rule='Append j8..19 +/- independently min(.01,.5actual hard-control margin), signed actual F64 endpoint subtraction; absdu<1e-6 unavailable. No clip, other19 commands exactly baseline bits.',
        unchanged='All12 source states, full300 before-action receipts, original episode safety baseline, original8 perturbations, nominal step, horizon, actualbounds and fixed model unchanged.'),
    reporting=dict(primary_original8='Preserve original8 results/cohorts as primary.',secondary_added12='Report added12 as a prespecified secondary control for spurious NN gradients in other fingers.',
        all20='Report all20 separately; an8-axis check is never described as full20D validation.',
        metrics='All parent59 original-unit/TRAiN-normalized blocks and masks, native-vs-model same-step secants, own-model AD/FP32 FD, full orientation and contact signatures unchanged.',
        timing='These cohorts and actions are fixed before any new physics/modelAD results; no result-dependent subset/checkpoint/threshold selection.'),
    failure_and_qualification='Parent engineering/baseline reproduction/safety/partial masks unchanged. Low secant error or all20 coverage is not true-physical-Jacobian, holding, policy or real-ego qualification.',
    no_new_learning=True,actual_new_native_model_FW_AD_training_calls=0,TEST_NPZ_reads=0)
ap=H/'design_addendum_v2_all20.json';write(ap,addendum)
print(json.dumps({'addendum':pin(ap),'selection':pin(sp),'available8':available8,'available12':available12,
    'branches':488,'max_native_steps':maximum_steps,'new_native_NN_training_calls':0}))
