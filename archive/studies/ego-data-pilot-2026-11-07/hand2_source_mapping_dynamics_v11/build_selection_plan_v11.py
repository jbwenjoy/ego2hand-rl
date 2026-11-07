"""Read-only source/JSON pins and AST literals; creates a CPUFalse plan, no models."""
from pathlib import Path
import ast,hashlib,json
H=Path(__file__).resolve().parent
P=H.parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
runner=H/'run_source_mapping_dynamics_v11.py'
tree=ast.parse(runner.read_text());constants={}
for n in tree.body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['SCHEMA','DTS','DURATION','GATES','PAIR_GATES']:
        constants[n.targets[0].id]=ast.literal_eval(n.value)
assert len(constants)==5
binding=P/'hand2_holding_v1/compiled_nominal_v2/binding.json';b=read(binding)
sources={name:pin(H/file) for name,file in {
    'runner':'run_source_mapping_dynamics_v11.py','selector':'selector_source_v11.txt',
    'source_builder':'build_runner_source_v11.py','source_derivation':'source_derivation_v11.json',
    'state_preparation':'prepare_selected_states_v11.py','plan_builder':'build_selection_plan_v11.py',
    'physics_preservation_checker':'check_source_preservation_v11.py',
    'physics_preservation_receipt':'source_preservation_v11.json'}.items()}
sources.update(old_holding_runner=pin(P/'hand2_holding_v1/run_cpu_holding_v1.py'),
    constant_u_parent=pin(P/'hand2_constant_u_baseline_v1/run_cpu_constant_u_baseline_v1.py'),scene=b['source'])
V=P/'surface_assembly_v11'
plan=dict(schema=constants['SCHEMA'],CPU_run_authorized=False,training_or_GPU_authorized=False,
    output=str(H/'cpu_nominal_v1'),dt_s=constants['DTS'],duration_s=constants['DURATION'],
    gates=constants['GATES'],dt_pair_gates=constants['PAIR_GATES'],
    inputs=dict(model_binding=pin(binding),model_mjb=b['model'],
        exact_source_protocol=pin(V/'protocol_exact_source_original_hand_run_v1.json'),
        exact_source_report=pin(V/'exact_source_original_hand_completed_v1/report.json'),
        sensitivity_protocol=pin(V/'protocol_source1mm_opposed_side_original_hand_run_v1.json'),
        sensitivity_report=pin(V/'source1mm_opposed_side_original_hand_completed_v1/report.json')),
    source_pins=sources,selections=[],
    process_terminal_receipts={name:pin(V/dirname/'terminal.json') for name,dirname in {
        'exact_source':'process_exact_source_v1','source1mm':'process_source1mm_v1'}.items()},
    selection_rule='One rank0 seed per mapping branch with minimum saved current_native_LP.scaled_loss among known assembly/fullgeometry/loaded/legal-control evidence; tie seed. Declared before dynamics.',
    control_rule='If original holding_diagnostic_eligible and complete strict-control two-fresh evidence use fresh command0; otherwise original native_probe/raw. Preserve both fresh evidence and all false ideal flags.',
    budget=dict(mapping_branches=2,dt_cases_per_branch=2,maximum_cases=4,maximum_integration_steps=38400,
        maximum_MJB_reads=1,maximum_MjData=4,maximum_initial_forward=4,new_training_GPU=0),
    scope='Exploratory full-retarget-pipeline constant-u diagnostic, not isolated geometry causal effect, policy benefit, measured mass/friction calibration, or faithful ego success. Original2s individual and dt-pair gates decide nominal holding. Strict ideal point LP and zero acceleration are not necessary admission gates; initial3N descriptive only.')
out=H/'selection_plan_v11.json'
with out.open('x') as f:json.dump(plan,f,indent=2,allow_nan=False)
print(json.dumps({'selection_plan':pin(out),'actual_model_load_forward_step_calls':0}))
