"""Source derivation only. Never calls a model, preflight or simulation."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent
old=H.parent/'hand2_constant_u_baseline_v1/run_cpu_constant_u_baseline_v1.py'
text=old.read_text();selector=(H/'selector_source_v11.txt').read_text()
assert hashlib.sha256(old.read_bytes()).hexdigest()=='1750796f7c21d39f706c912f0cafbe9e8c6c871b5fdb65840227cc5fa5ae04d2'
replacements={
 'hand2_phone_matched_constant_u_baseline_CPU_protocol:1':'hand2_source_mapping_constant_u_diagnostic_protocol:11',
 'hand2_phone_matched_constant_u_baseline_CPU_result:1':'hand2_source_mapping_constant_u_diagnostic_result:11',
 'COMPLETED_MATCHED_CONSTANT_U_BASELINE_BOTH_DT':'COMPLETED_SOURCE_MAPPING_CONSTANT_U_DYNAMIC_DIAGNOSTIC',
 'BLOCKED_FIXED_BASELINE_EVIDENCE_NO_MODEL_OR_STEPS':'BLOCKED_NO_SOURCE_MAPPING_DYNAMIC_CANDIDATE_NO_MODEL_OR_STEPS',
 'Exact finite original rank0 pose and zero speed':'Exact finite pinned source-mapping pose and zero speed',
 'Registered constant-u baseline: fixedrank0, all nonacceleration entry/repeat/audit/22+189 gates; three static accelerations descriptive only':'Pinned source mapping: assembly/fullgeometry/known/loaded/legal control; strictLP and zeroacc not admission; initial3N descriptive',
 'Fixed original seed0rank0 constant-u baseline, both fixed dt, unchanged original2s case gates. ':'Two pinned source-normal mapping branches, same original model, both dt and unchanged2s gates. ',
 'New designated-pad initial compression up to0.1mm is a hypothesis, not measured compliance; no old point LP or positive-gap condition. ':'Source designated-pad initial compression up to0.1mm remains a hypothesis. Strict ideal LP is not necessary for soft-contact dynamic diagnosis. ',
 'No object aid, pose reset, materials/gains/solver tuning, sample selection or retries. ':'No object aid, pose reset or materials/gains/solver tuning. Preserved failed source flags; no faithful ego, learning or isolated geometry causal claim. '
}
new=text[:text.index('NATIVE_GATE_KEYS =')]+selector+'\n'+text[text.index('def dt_pair_summary'):]
for a,b in replacements.items():
 assert a in new,('Missing declared literal replacement',a)
 new=new.replace(a,b)
ast.parse(new)
def function(s,name):return next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
reversed_text=new
for a,b in replacements.items():reversed_text=reversed_text.replace(b,a)
assert ast.dump(function(reversed_text,'main'),include_attributes=False)==ast.dump(function(text,'main'),include_attributes=False)
target=H/'run_source_mapping_dynamics_v11.py'
with target.open('x',encoding='utf8') as f:f.write(new)
pin=lambda p:dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
diff=H/'source_derivation_v11.diff'
with diff.open('x') as f:f.write(''.join(difflib.unified_diff(text.splitlines(True),new.splitlines(True),fromfile=str(old),tofile=str(target))))
report=dict(schema='source_mapping_dynamics_runner_source_derivation:11',parent=pin(old),runner=pin(target),selector=pin(H/'selector_source_v11.txt'),builder=pin(Path(__file__)),diff=pin(diff),enumerated_literal_changes=replacements,whole_main_AST_exact_after_reversing_listed_literals=True,new_native_or_physics_calls=0,scope='Only selector/schema/descriptive literals change. Original model loading, NPZ raw binding, native tick loop, PRE/POST metrics, gates, aborts and dt-pair code preserved.')
with (H/'source_derivation_v11.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({'runner':pin(target),'derivation':pin(H/'source_derivation_v11.json')}))
