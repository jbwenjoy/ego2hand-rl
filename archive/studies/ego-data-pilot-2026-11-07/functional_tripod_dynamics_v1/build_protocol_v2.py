"""Build an unauthorized source/metadata protocol from the completed tripod geometry report."""
from pathlib import Path
import argparse, ast, hashlib, json

H=Path(__file__).resolve().parent
P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,value):
    with Path(p).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
def constants():
    # AST literal metadata only; do not import the runtime or a physics backend.
    tree=ast.parse((H/'run_functional_tripod_dynamics_v2.py').read_text())
    found={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:found[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    old=found['OLD_GATES'];found['GATES']={**{k:v for k,v in old.items() if k!='tail_dual_pad_force_coverage'},'tail_triple_pad_force_coverage':.95}
    return found
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate-report',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    k=constants();r=json.loads(a.candidate_report.read_text(encoding='utf8'))
    if r['schema']!=k['CANDIDATE_REPORT_SCHEMA']:raise ValueError('New candidate report schema')
    if not a.output.is_absolute() or a.output.exists() or a.output.parent.resolve()!=H.resolve():raise ValueError('Fresh new namespace output')
    c=dict(schema=k['SCHEMA'],CPU_run_authorized=False,training_or_GPU_authorized=False,
        output=str(a.output),dt_s=k['DTS'],duration_s=k['DURATION'],gates=k['GATES'],old_two_pad_gates=k['OLD_GATES'],
        dt_pair_gates=k['PAIR_GATES'],control_routes=k['CONTROL_ROUTES'],designated_geom_ids=k['DESIGNATED'],
        inputs=dict(model_mjb=pin(P/'hand2_holding_v1/compiled_nominal_v2/model.mjb'),
                    model_binding=pin(P/'hand2_holding_v1/compiled_nominal_v2/binding.json'),candidate_report=pin(a.candidate_report)),
        source_pins=dict(runner=pin(H/'run_functional_tripod_dynamics_v2.py'),protocol_builder=pin(__file__),
            parent_two_pad_runner=pin(P/'hand2_source_mapping_dynamics_v11/run_source_mapping_dynamics_v11.py'),
            original_holding_runner=pin(P/'hand2_holding_v1/run_cpu_holding_v1.py')),
        budget=dict(max_candidates=4,control_routes=3,dt_cases=2,maximum_cases=24,maximum_integration_steps=230400),
        preload_rule=dict(desired_normal_N_each=[0,1,3],no_preload='ctrl=q when legal',
            inverse_PD='tau_act=bias-passive+sum(J_hand.T@F_on_object); u=(tau-offset)/gain; never clamp',
            geometry_anchor_FK_world_tolerance_m=1e-10,geometry_normal_FK_tolerance=1e-10,
            tangential_load='No extra feedforward tangential load in this bounded normal-preload ablation. Native friction must supply actual support; no static acceleration gate.'),
        automatic_retry=False,source_stage='UNFROZEN_REVIEW_REQUIRED_NOT_EXECUTION_AUTHORIZATION',
        scope='All predeclared geometry candidates and all three available controls/both dt retained. No labels-only pass; initial actual native Fn and all2s dynamic gates decide nominal outcome. New designated41 changes the functional experiment; old2pad result remains separately reported.')
    write(a.protocol,c);print(json.dumps(dict(protocol=pin(a.protocol),CPU_run_authorized=False)))
if __name__=='__main__':main()
