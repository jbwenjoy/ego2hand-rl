"""Read-only arithmetic audit; retain all three failed admission reports."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from supermimic_research.strict_json import write_json


def main():
    start=time.perf_counter(); out=ROOT/'runs/pen_stable_friction_arithmetic_saved_review_v1'
    if out.exists(): raise ValueError('Fresh independent arithmetic review required')
    inputs=[]; rows=[]; checks={}; arrays={}
    for version in [1,2,3]:
        pin=file_pin(ROOT/f'runs/pen_stable_friction_backend_qualification_v{version}/report.json'); inputs.append(pin)
        q=json.loads(verify_file_pin(pin).read_text()); inputs.extend(q['inputs'].values()); inputs.extend([q['outputs'],q['source_receipt'],q['factory_manifest']])
        verify_source_snapshot(json.loads(verify_file_pin(q['source_receipt']).read_text()),live=True)
        z=dict(np.load(verify_file_pin(q['outputs']))); arrays[version]=z
        batch=dict(np.load(verify_file_pin(q['inputs']['fixed_inputs']))); x=batch['inputs']; target=batch['FP64_delta']
        j,v,d,f,a=x.astype(np.float64).T; radius=f/d; advanced=a*v; y=j+advanced
        h=((-radius<j)&(j<radius)&(-radius<y)&(y<radius))|((j<=-radius)&(y<=-radius))|((j>=radius)&(y>=radius))
        recorded=z['handled']; item=dict(version=version,producer=pin,recorded_checks=q['checks'],recorded_passed=q['passed'],recorded_total=q['total'],
            rows=len(x),fallback=int((~h).sum()),FP64_same_regime_matches_recorded=bool(np.array_equal(h,recorded)),comparisons={})
        for label in ['single','three']:
            old,new=z['original_'+label],z['candidate_'+label]
            if label=='three': old,new=old[:,0],new[:,0]
            neq=old.view(np.uint32)!=new.view(np.uint32); fallback=~recorded
            item['comparisons'][label]=dict(gradient_changed_rows=int(neq[:,1].sum()),hessian_changed_rows=int(neq[:,2].sum()),
                fallback_cost_changed_rows=int(neq[fallback,0].sum()),fallback_max_abs_delta_change=float(abs(old[fallback,0].astype(float)-new[fallback,0].astype(float)).max()),
                fallback_sign_disagreements=int(np.sum((old[fallback,0]<0)!=(new[fallback,0]<0))),
                original_max_abs_error=float(abs(old[:,0].astype(float)-target).max()),candidate_max_abs_error=float(abs(new[:,0].astype(float)-target).max()))
            checks[f'v{version}_{label}_slope_gate_recomputed']=(not neq[:,1:].any())==q['checks'][label.replace('three','triple')+'_derivative_hessian_bytes']
            checks[f'v{version}_{label}_fallback_gate_recomputed']=(not neq[fallback].any())==q['checks'][label.replace('three','triple')+'_fallback_bytes']
        item['original_single_vs_original_three_fallback_cost_changed_rows']=int(np.sum(z['original_single'][~recorded,0].view(np.uint32)!=z['original_three'][~recorded,0,0].view(np.uint32)))
        item['v3_candidate_vs_original_three_fallback_bytes_equal']=z['candidate_single'][~recorded].tobytes()==z['original_three'][~recorded,0].tobytes() if version==3 else None
        checks[f'v{version}_all_rows_finite']=len(x)==139160 and all(np.isfinite(v).all() for v in z.values())
        checks[f'v{version}_failure_preserved']=q['passed']==sum(q['checks'].values())<q['total']==len(q['checks'])
        checks[f'v{version}_zero_native_and_updates']=q['actual_native_integrations']==q['optimizer_steps']==q['normalizer_fits']==0
        rows.append(item)
    dp=file_pin(ROOT/'runs/pen_stable_friction_compile_layout_v1/report.json'); inputs.append(dp); diag=json.loads(verify_file_pin(dp).read_text())
    inputs.extend([diag[k] for k in ['source','generated','fixed_inputs','outputs']]); separate=dict(np.load(verify_file_pin(diag['outputs']))); h=arrays[2]['handled']
    for label in ['single','triple']:
        old,new=separate['original_'+label],separate['candidate_'+label]
        expected=dict(derivative_hessian_bytes_equal=old[:,1:].tobytes()==new[:,1:].tobytes(),fallback_bytes_equal=old[~h].tobytes()==new[~h].tobytes(),
            fallback_cost_differences=int(np.sum(old[~h,0].view(np.uint32)!=new[~h,0].view(np.uint32))),
            derivative_differences=int(np.sum(old[:,1:].view(np.uint32)!=new[:,1:].view(np.uint32))))
        checks[label+'_separate_diagnostic_recomputed']=expected==diag['comparisons'][label]
    checks['diagnostic_zero_native']=diag['actual_native_integrations']==diag['optimizer_steps']==0 and diag['kernel_launches']==4
    for pin in inputs: verify_file_pin(pin)
    out.mkdir(); report=dict(schema='pen_stable_friction_failed_arithmetic_review:1',source=file_pin(__file__),inputs=inputs,rows=rows,
        separate_kernel_comparisons=diag['comparisons'],checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-start,
        scope='Green review means all failed qualification evidence is faithfully retained/recomputed. It does not qualify any candidate. '
              'All three original qualification reports remain failed. No gate is relaxed, no simulation or training was run. Compiler-causality remains an inference,not uniquely established.')
    write_json(out/'report.json',report)
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[report['passed'],report['total']],rows=rows)),flush=True)
    if not all(checks.values()): raise ValueError('Failed evidence review mismatch')


if __name__=='__main__': main()
