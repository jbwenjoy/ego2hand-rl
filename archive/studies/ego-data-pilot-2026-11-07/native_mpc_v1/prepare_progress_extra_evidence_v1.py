"""Index saved-only reviews and figures for the current goal handoff."""
from pathlib import Path
import json
from rollout_probe_v1 import pin

r=Path(__file__).resolve().parent
p=r.parent
files={
 'MPC_saved_arithmetic':r/'saved_recheck_v1/report.json',
 'MPC_saved_counts_provenance':r/'saved_count_review_v1/report.json',
 'MPC_saved_failure_mechanism':r/'failure_mechanism_saved_v1/report.json',
 'MPC_matched_prefix_figure':r/'saved_recheck_v1/constant_u_vs_MPC_actual_prefix.png',
 'fixed_contact_upper_bound_description':p/'native_equilibrium_v6/wrench_upper_bound_saved_v1/actual_v1/description.md',
 'fixed_contact_upper_bound_source_review':r/'saved_count_review_v1/wrench_upper_bound_source_review_v1.json'}
for key in ('MPC_saved_arithmetic','MPC_saved_counts_provenance'):
    report=json.loads(files[key].read_text(encoding='utf8'))
    if report['passed']!=report['total']:raise ValueError('Incomplete saved audit '+key)
target=r/'final_extra_evidence_v1.json'
with target.open('x',encoding='utf8') as f:json.dump({k:pin(v) for k,v in files.items()},f,indent=2)
print(json.dumps({'index':pin(target),'pins':len(files),'new_model_physics_training_calls':0}))
