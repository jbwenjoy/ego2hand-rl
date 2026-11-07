"""AST/literal/source-preservation check only; no runtime/NumPy/MuJoCo import."""
from pathlib import Path
import ast,hashlib,json

H=Path(__file__).resolve().parent
P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def literals(tree):
    out={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:out[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    return out
def calls(tree,name):
    return [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr==name]
old=P/'hand2_source_mapping_dynamics_v11/run_source_mapping_dynamics_v11.py';new=H/'run_functional_tripod_dynamics_v1.py'
ot=ast.parse(old.read_text());nt=ast.parse(new.read_text());oc=literals(ot);nc=literals(nt)
case=next(n for n in nt.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_case')
loop=next(n for n in case.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='tick')
checks=dict(
    original_all_two_pad_gate_values_retained=oc['GATES']==nc['OLD_GATES'],
    original_duration_and_both_dt_and_pair_thresholds_retained=all(oc[k]==nc[k] for k in ['DTS','DURATION','PAIR_GATES']),
    all_three_controls_and_true_geometry_ids_predeclared=nc['CONTROL_ROUTES']==['no_preload_ctrl_equal_q','normal_1N_each_pad','normal_3N_each_pad'] and nc['DESIGNATED']==[15,28,41],
    one_native_step_site_inside_original_tick_loop=len(calls(nt,'mj_step'))==1 and len(calls(loop,'mj_step'))==1,
    no_forward_inside_native_tick_loop=len(calls(loop,'mj_forward'))==0,
    preparation_and_each_case_initial_forward_only=len(calls(nt,'mj_forward'))==2,
    no_neural_optimizer_or_native_distance_calls=not any(calls(nt,x) for x in ['mj_geomDistance','backward','optimizer','mj_step1','mj_step2']),
    real20joint_source_does_not_lock_finger3='qrest' not in new.read_text() and "q=np.asarray(c['qpos27_F64'],dtype=np.float64)" in new.read_text() and 'q[:20]<m.jnt_range[:20,0]' in new.read_text(),
    old2pad_reclassifies_third_force='oldnonpad=r[\'pre_phone_nonpad_normal_N\']+r[\'pre_pad_normal_N\'][:,2]' in new.read_text(),
    controller_inverse_sign_is_explicit='base=d.qfrc_bias[:20]-d.qfrc_passive[:20]' in new.read_text() and "base+np.einsum('pik,pi->k',J,f)" in new.read_text(),
    geometry_anchor_and_normal_same_q_FK_checks_present='Same-q hand anchor/normal FK mismatch' in new.read_text(),
    full_model_audit_excludes_only_explicit_dt='model_before!=model_after' in new.read_text() and "name=='timestep'" in new.read_text())
report=dict(schema='functional_tripod_dynamic_source_preservation:1',status='PASS_AST_SOURCE_ONLY_NOT_PHYSICS_EXECUTION' if all(checks.values()) else 'FAILED_SOURCE_CONTRACT',
    passed=sum(checks.values()),total=len(checks),checks=checks,parent=pin(old),runner=pin(new),checker=pin(__file__),
    changed_physics_equations=False,new_native_executions=0,
    declared_changes=['New three-pad candidate/geometry metadata adapter and no static-qacc admission.',
        'Predeclared ctrl=q/1N/3N normal-preload inverse-PD preparation: one fresh forward and three physical hand-point Jacobians per geometry candidate.',
        'Contact24 decoding unchanged; designated set grows15/28 to15/28/41, with old2pad views derived by counting41 as old nonpad.',
        'Original step/finite/no-aid/warning/loss abort phases,2s tail and thresholds preserved semantically; loop factored with additive parameter/margin traces.',
        'Full-model start/end digest excludes timestep only; original material/gain/solver arrays also saved.',
        'This is not byte-identical whole-loop provenance: extracted contact helper and extra measurements are explicitly new source and require peer review.'])
with (H/'source_preservation_v1.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(report))
if not all(checks.values()):raise SystemExit(1)
