"""Evaluate pre-data conditional formulas on saved converter evidence only.

No production imports, native conversion, MJB, model, objective, or GPU. The
envelopes below implement frozen a9505c formulas in ordinary float64. They are
conditional diagnostics, not outward-rounded certified bounds or new gates.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'outputs/full151105_converter_forensic_CPU_v1'
OUTPUT = ROOT / 'outputs/full151105_converter_conditioning_saved_v1'
FREEZE = ROOT / 'outputs/full151105_converter_arithmetic_source_review_v1/FREEZE.json'
FREEZE_SHA = '2e1b9af72d694fc11772395460869c5eb26af7b65965b1d223b841a2d83f689e'
GROUPS = {'position': (40,43,'m'), 'rotation': (43,52,'dimensionless'), 'linear_velocity': (52,55,'m/s'), 'angular_velocity': (55,58,'rad/s')}
U = 2.0**-24


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pin(path):
    return {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def checked(value):
    path = Path(value['path'])
    require(pin(path)['sha256'] == value['sha256'], 'Changed input/source: ' + str(path))
    return path


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def gamma(n):
    return n*U/(1-n*U)


def mv(a, b):
    # Match the native product + sum definition in a secondary F64 evaluation.
    return (a*b[..., None, :]).sum(axis=-1)


def mm(a, b):
    return (a[..., :, :, None]*b[..., None, :, :]).sum(axis=-2)


def multiply(a, b):
    x, ex = a
    y, ey = b
    return x*y, abs(x)*ey+abs(y)*ex+ex*ey+U*(abs(x)+ex)*(abs(y)+ey)


def add(a, b, sign=1):
    x, ex = a
    y, ey = b
    return x+sign*y, ex+ey+U*(abs(x)+abs(y)+ex+ey)


def rotation_envelope(h, eh):
    w, x, y, z = [(h[:,i], eh[:,i]) for i in range(4)]
    const = lambda value: (value, 0.)
    two = lambda value: multiply(const(2.), value)
    diag = lambda a,b: add(const(1.), two(add(multiply(a,a),multiply(b,b))), -1)
    off = lambda a,b,c,d,sign: two(add(multiply(a,b),multiply(c,d),sign))
    entries = [diag(y,z),off(x,y,w,z,-1),off(x,z,w,y,1),
               off(x,y,w,z,1),diag(x,z),off(y,z,w,x,-1),
               off(x,z,w,y,-1),off(y,z,w,x,1),diag(x,y)]
    return np.stack([a for a,b in entries],-1).reshape(-1,3,3), np.stack([b for a,b in entries],-1).reshape(-1,3,3)


def evaluate(raw):
    q = np.concatenate((raw['pre_qpos'],raw['post20_qpos'])).astype(np.float64)
    v = np.concatenate((raw['pre_qvel'],raw['post20_qvel'])).astype(np.float64)
    oq, ov = int(raw['object_qpos_start']), int(raw['object_dof_start'])
    palm = raw['palm_pos_all2260'].astype(np.float64)
    a = raw['palm_rot_all2260'].astype(np.float64).swapaxes(-1,-2)
    quat = q[:,oq+3:oq+7]
    norm = np.sqrt((quat*quat).sum(axis=-1,keepdims=True))
    require(np.all(norm > 1e-12), 'Normalize clamp may be active')
    h = quat/norm
    eta = max((1+U)*np.sqrt(1+gamma(7))-1,1-(1-U)*np.sqrt(1-gamma(7)))
    eh = abs(h)*((1+U)/(1-eta)-1)
    r, er = rotation_envelope(h,eh)
    position = q[:,oq:oq+3]
    dp = position-palm
    worldv, bodyw = v[:,ov:ov+3], v[:,ov+3:ov+6]
    worldw = mv(r,bodyw)
    abs_a = abs(a)
    ez = U*(abs(position)+abs(palm))
    ep = gamma(5)*mv(abs_a,abs(dp))+(1+gamma(5))*mv(abs_a,ez)
    ev = gamma(5)*mv(abs_a,abs(worldv))
    erout = gamma(5)*mm(abs_a,abs(r))+(1+gamma(5))*mm(abs_a,er)
    et = gamma(5)*mv(abs(r),abs(bodyw))+(1+gamma(5))*mv(er,abs(bodyw))
    ew = gamma(5)*mv(abs_a,abs(worldw))+(1+gamma(5))*mv(abs_a,et)
    shadow = np.concatenate((q[:,raw['hand_qpos']],v[:,raw['hand_dof']],mv(a,dp),mm(a,r).reshape(-1,9),mv(a,worldv),mv(a,worldw)),-1)
    envelope = np.concatenate((np.zeros((len(q),40)),ep,erout.reshape(-1,9),ev,ew),-1)
    operands = np.concatenate((abs(shadow[:,:40]),mv(abs_a,abs(dp)),mm(abs_a,abs(r)).reshape(-1,9),mv(abs_a,abs(worldv)),mv(abs_a,mv(abs(r),abs(bodyw)))),-1)
    output_zero = shadow == 0
    ratio = np.full_like(shadow,np.nan)
    np.divide(operands,abs(shadow),out=ratio,where=~output_zero)
    return {'shadow58_F64':shadow,'one_side_conditional_envelope58_F64':envelope,'operand_sums58_F64':operands,
            'conditioning_ratio58_defined_F64':ratio,'conditioning_ratio58_zero_output':output_zero,
            'position_coordinate_operand_scale_F64':mv(abs_a,abs(position)+abs(palm)),
            'normalized_quaternion_F64':h,'normalized_quaternion_conditional_error_F64':eh,
            'object_rotation_F64':r,'object_rotation_conditional_error_F64':er,'object_quaternion_norm_F64':norm[:,0],
            'body_angular_velocity_F64':bodyw,'world_angular_velocity_F64':worldw,
            'omega_rotation_polynomial_error_contribution_F64':(1+gamma(5))*mv(abs_a,(1+gamma(5))*mv(er,abs(bodyw)))}


def max_value(array):
    return float(np.max(array))


def distribution(array):
    a = np.asarray(array)
    return {'min':float(np.min(a)),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(np.max(a))}


def compare_summary(actual,reference,envelope):
    residual = abs(actual-reference)
    out = {}
    for name,(s,e,unit) in GROUPS.items():
        delta, bound = residual[:,s:e], envelope[:,s:e]
        ratio = np.zeros_like(delta)
        np.divide(delta,bound,out=ratio,where=bound>0)
        ratio[(bound==0)&(delta>0)] = np.inf
        out[name] = {'units':unit,'elements':delta.size,'max_abs_difference':max_value(delta),'evaluated_envelope_min_margin':float(np.min(bound-delta)),
                     'elements_above_evaluated_envelope':int((delta>bound).sum()),'max_residual_over_evaluated_envelope':float(np.max(ratio)),
                     'conditional_only_not_a_qualification':True}
    return out


def write_json(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')


def main():
    require(not OUTPUT.exists(),'Fresh output required')
    require(pin(FREEZE)['sha256']==FREEZE_SHA,'Pre-data formula freeze changed')
    frozen = read(FREEZE)
    original_pins = {str(FREEZE):pin(FREEZE)}
    for group in ('artifact_pins','source_pins'):
        for item in frozen[group].values():
            original_pins[str(checked(item))] = item
    inputs = read(INPUT/'output_pins.json')
    for key in ['raw_conversion_all2260.npz','conditioning_secondary_all2260.npz','endpoint452_old_rule.npz','leading160_old_rule.npz','endpoint_identities.json','anchor_identities.json','summary.json','report.json','solver_position1808_old_rule.npz','solver_rotation1808_old_rule.npz']:
        item=inputs[key]
        original_pins[str(checked(item))]=item
    raw=load(INPUT/'raw_conversion_all2260.npz')
    conditions=load(INPUT/'conditioning_secondary_all2260.npz')
    endpoint=load(INPUT/'endpoint452_old_rule.npz')
    leading=load(INPUT/'leading160_old_rule.npz')
    labels=read(INPUT/'endpoint_identities.json')
    anchor_labels=read(INPUT/'anchor_identities.json')
    computed=evaluate(raw)
    shadow=computed['shadow58_F64']
    one=computed['one_side_conditional_envelope58_F64']
    converted=raw['converted_native58'].astype(np.float64)
    refs=raw['post20_reference'].astype(np.float64)
    require(shadow.shape==(2260,58) and len(labels)==452 and len(anchor_labels)==160,'Exact row scope required')
    require(np.array_equal(raw['converted_native58'][1808:],endpoint['actual']) and np.array_equal(raw['post20_reference'],endpoint['reference']),'Saved endpoint identity mismatch')
    require(np.array_equal(raw['converted_native58'][raw['anchor_pre_indices']],leading['actual']),'Saved anchor identity mismatch')
    failures=np.argwhere(~endpoint['old_flags'])
    checks={'formula_freeze_and_ten_source_pins_match':True,'all2260_rows_finite':bool(np.isfinite(shadow).all() and np.isfinite(one).all()),
            'all452_endpoint_authorities_exact':True,'all160_anchor_authorities_exact':True,
            'no_q_qd_disagreement_all2260':bool(np.array_equal(converted[:,:40],shadow[:,:40])),
            'F64_shadow_agrees_with_producer_secondary_to_1e14_relative_diagnostic':bool(np.allclose(shadow,conditions['F64_shadow_native_definition_NOT_ACCEPTED_STATE'],rtol=1e-14,atol=1e-14)),
            'old_failure_elements_five_omega_only':bool(len(failures)==5 and np.all(failures[:,1]>=55)),
            'Torch_Warp_MuJoCo_not_imported':not any(k in sys.modules for k in ['torch','warp','mujoco'])}
    # The 1e-14 comparison is solely independent F64 expression consistency; it
    # cannot qualify the converter, replace old8eps, or change any endpoint flag.
    require(all(checks.values()),str(checks))
    OUTPUT.mkdir()
    row_arrays = dict(computed)
    row_arrays.update({'CPU_minus_F64shadow_all2260':converted-shadow,'GPU_endpoint_minus_F64shadow452':refs-shadow[1808:],
                       'CPU_GPU_endpoint_pair_envelope452_F64':2*one[1808:],'CPU_GPU_endpoint_pair_margin452_F64':2*one[1808:]-abs(endpoint['difference_F64']),
                       'CPU_F64shadow_margin_all2260_F64':one-abs(converted-shadow),
                       'old_endpoint_flags452x58':endpoint['old_flags'],'old_endpoint_bounds452x58':endpoint['old_bound_F64'],
                       'old_endpoint_difference452x58':endpoint['difference_F64']})
    np.savez_compressed(OUTPUT/'conditional_diagnostics_all2260.npz',**row_arrays)
    fail_rows=[]
    for row,feature in failures:
        i=int(row)+1808;feature=int(feature);row=int(row)
        fail_rows.append({'endpoint_row':row,'all_row':i,'feature':feature,'component':feature-55,'identity':labels[row],
                         'CPU':float(converted[i,feature]),'GPU':float(refs[row,feature]),'F64_shadow':float(shadow[i,feature]),
                         'CPU_minus_GPU':float(converted[i,feature]-refs[row,feature]),'old_bound':float(endpoint['old_bound_F64'][row,feature]),
                         'old_margin':float(endpoint['old_bound_F64'][row,feature]-abs(converted[i,feature]-refs[row,feature])),
                         'CPU_minus_F64':float(converted[i,feature]-shadow[i,feature]),'GPU_minus_F64':float(refs[row,feature]-shadow[i,feature]),
                         'conditional_one_side_envelope':float(one[i,feature]),'conditional_pair_envelope':float(2*one[i,feature]),
                         'conditional_pair_margin':float(2*one[i,feature]-abs(converted[i,feature]-refs[row,feature])),
                         'conditional_pair_residual_ratio':float(abs(converted[i,feature]-refs[row,feature])/(2*one[i,feature])),
                         'operand_abs_sum':float(computed['operand_sums58_F64'][i,feature]),
                         'conditioning_ratio':float(computed['conditioning_ratio58_defined_F64'][i,feature]),
                         'body_omega_absmax':float(abs(computed['body_angular_velocity_F64'][i]).max()),
                         'quat_norm':float(computed['object_quaternion_norm_F64'][i]),'units':'rad/s'})
    write_json(OUTPUT/'five_original_failed_components.json',fail_rows)
    with (OUTPUT/'all2260_component_conditioning.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['all_row','source','endpoint_row','group','feature','units','CPU','F64_shadow','CPU_minus_F64','conditional_one_side_envelope','conditional_one_side_margin','absolute_operand_sum','conditioning_ratio_if_nonzero_output','zero_output','GPU_endpoint_or_blank','CPU_minus_GPU_or_blank','old_bound_or_blank','old_flag_or_blank','conditional_pair_margin_or_blank'])
        for i in range(2260):
            for name,(s,e,unit) in GROUPS.items():
                for j in range(s,e):
                    end=i-1808
                    extra=[float(refs[end,j]),float(converted[i,j]-refs[end,j]),float(endpoint['old_bound_F64'][end,j]),bool(endpoint['old_flags'][end,j]),float(2*one[i,j]-abs(converted[i,j]-refs[end,j]))] if end>=0 else ['']*5
                    writer.writerow([i,'post20_endpoint' if i>=1808 else 'solver_pre',end if end>=0 else '',name,j,unit,converted[i,j],shadow[i,j],converted[i,j]-shadow[i,j],one[i,j],one[i,j]-abs(converted[i,j]-shadow[i,j]),computed['operand_sums58_F64'][i,j],'' if computed['conditioning_ratio58_zero_output'][i,j] else computed['conditioning_ratio58_defined_F64'][i,j],bool(computed['conditioning_ratio58_zero_output'][i,j]),*extra])
    anchor_indices=raw['anchor_pre_indices']
    comparisons={'CPU_vs_F64_all2260':compare_summary(converted,shadow,one),
                 'GPU_vs_F64_endpoints452':compare_summary(refs,shadow[1808:],one[1808:]),
                 'CPU_vs_GPU_endpoints452':compare_summary(converted[1808:],refs,2*one[1808:]),
                 'CPU_vs_GPU_leading160':compare_summary(leading['actual'].astype(float),leading['reference'].astype(float),2*one[anchor_indices])}
    conditioning={}
    for name,(s,e,unit) in GROUPS.items():
        ratio=computed['conditioning_ratio58_defined_F64'][:,s:e]
        finite=ratio[np.isfinite(ratio)]
        conditioning[name]={'units':unit,'absolute_operand_sums':distribution(computed['operand_sums58_F64'][:,s:e]),'output_absolute_values':distribution(abs(shadow[:,s:e])),
                            'zero_outputs':int(computed['conditioning_ratio58_zero_output'][:,s:e].sum()),'finite_conditioning_ratios':distribution(finite),'conditional_one_side_envelopes':distribution(one[:,s:e])}
    normalized_cpu=conditions['normalized_quaternion_CPU_FP32'].astype(float)
    object_cpu=conditions['object_rotation_CPU_FP32'].astype(float)
    diagnostics={'CPU_normalized_quaternion_minus_F64_max':max_value(abs(normalized_cpu-computed['normalized_quaternion_F64'])),
                 'CPU_normalized_quaternion_components_outside_provisional_norm_model':int((abs(normalized_cpu-computed['normalized_quaternion_F64'])>computed['normalized_quaternion_conditional_error_F64']).sum()),
                 'CPU_object_rotation_minus_F64_max':max_value(abs(object_cpu-computed['object_rotation_F64'])),
                 'CPU_object_rotation_components_outside_provisional_polynomial_model':int((abs(object_cpu-computed['object_rotation_F64'])>computed['object_rotation_conditional_error_F64']).sum()),
                 'raw_quaternion_norm':distribution(computed['object_quaternion_norm_F64']),
                 'body_omega_absmax':max_value(abs(computed['body_angular_velocity_F64'])),
                 'producer_shadow_vs_independent_F64_max':max_value(abs(shadow-conditions['F64_shadow_native_definition_NOT_ACCEPTED_STATE'])),
                 'palm_rotation_distinct_bitpatterns':len({x.tobytes() for x in raw['palm_rot_all2260']}),
                 'palm_position_distinct_bitpatterns':len({x.tobytes() for x in raw['palm_pos_all2260']})}
    for value in original_pins.values():checked(value)
    selfpin=pin(__file__)
    source_freeze=read(OUTPUT.parent/'full151105_converter_conditioning_saved_source_freeze_v1/FREEZE.json')
    require(source_freeze['source']==selfpin,'Analysis source changed since freeze')
    report={'schema':'full151105_converter_predata_conditioning_saved_analysis:1','status':'COMPLETED_CONDITIONAL_SAVED_DIAGNOSTICS_OLD_GATE_FAIL_RETAINED',
            'source':selfpin,'source_freeze':pin(OUTPUT.parent/'full151105_converter_conditioning_saved_source_freeze_v1/FREEZE.json'),
            'input_pins':original_pins,'formula_contract':frozen['artifact_pins']['conditional_equations'],
            'rows':{'all':2260,'solver_pre':1808,'post20_endpoint':452,'leading_anchors':160,'old_failed_elements':5,'old_failed_rows':5},
            'checks':checks,'passed':sum(checks.values()),'total':len(checks),'comparisons':comparisons,'conditioning_all2260':conditioning,'auxiliary_diagnostics':diagnostics,
            'failed_components':fail_rows,
            'arithmetic_model_assumptions':'Same exact FP32 operands and ordinary finite normal-operation rounding for normalization/reductions/sqrt/division. Kernel/FTZ/dispatch assumptions remain UNKNOWN. Evaluator is ordinaryF64, not outward-rounded interval certification.',
            'interpretation':'Compatibility within the frozen conditional envelope is evidence consistent with conditioning-sensitive arithmetic, not unique causation, semantic equivalence proof, a new acceptance gate, or restore/proxy/physical-gradient eligibility. All original flags/residuals remain saved unchanged.',
            'source_runtime_boundary':'Native mm/mv are elementwise multiply+sum, not GEMM. Source TF32 assignment does not explain these differences by itself; no actual historical GPU dispatch/precision trace is established.',
            'qualification':{'original8eps':'FAIL','new_operative_gate':False,'dense_scoring_eligibility':False,'semantic_equivalence_certified':False,'physical_gradient_qualified':False},
            'execution':{'saved_arrays_only':True,'native_state_conversions':0,'MJB_loads':0,'objectives_models_WM_Actor_Env_physics_GPU':0,'Torch_imported':False,'Warp_imported':False},
            'pending':['Actual native norm/reduction kernel rounding proof or independent operational acceptance protocol','No GPU normalized-quaternion intermediate was captured; endpoint references cannot uniquely identify which arithmetic stage differs','Broad frame/phase/address mismatch is not indicated by this sparse pattern but cannot be uniquely excluded from numerical agreement alone']}
    write_json(OUTPUT/'report.json',report)
    write_json(OUTPUT/'output_pins.json',{p.name:pin(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()})
    print(json.dumps({'status':report['status'],'report':pin(OUTPUT/'report.json'),'comparisons':comparisons,'failed_components':fail_rows,'auxiliary_diagnostics':diagnostics},indent=2))


if __name__=='__main__':
    main()
