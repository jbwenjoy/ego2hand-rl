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
old=P/'hand2_source_mapping_dynamics_v11/run_source_mapping_dynamics_v11.py';new=H/'run_functional_tripod_dynamics_v3.py'
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
# One metadata-only synthetic regression: four retained records, two rejected
# empty-anchor failures, two complete metadata proposals. No NPZ values loaded.
import importlib.util,sys
F=H/'metadata_rejection_fixture_v3';F.mkdir()
def write_fixture(p,j):
    with p.open('x',encoding='utf8') as f:json.dump(j,f,indent=2,allow_nan=False)
state=F/'synthetic_state_not_loaded.npz';state.write_bytes(b'SYNTHETIC_METADATA_HASH_ONLY_NOT_AN_ARRAY')
entries=[]
for i in range(4):
    passed=i>=2
    pads=[dict(hand_geom_id=g,hand_body_id=j+1,hand_anchor_world_m=[0.,0.,0.],geom_local_hand_anchor_m=[0.,0.,0.],
        force_on_object_unit_normal_world=[1.,0.,0.],geom_local_force_on_object_normal=[1.,0.,0.]) for j,g in enumerate([15,28,41])] if passed else []
    c=dict(schema='functional_tripod_geometric_reference_candidate:1',candidate_id='synthetic_'+str(i),
        qpos27_F64=[0.]*23+[1.,0.,0.,0.],qvel26_F64=[0.]*26,designated_geom_ids=[15,28,41],pad_bindings=pads,
        geometry=dict(passed=passed,phone_pair_count=22 if passed else 0,self_pair_count=189 if passed else 0,
            table_complete=passed,unknown=[] if passed else [dict(configuration_error='synthetic retained failure')]))
    cp=F/('candidate_'+str(i)+'.json');write_fixture(cp,c);entries.append(dict(candidate=pin(cp),state_npz=pin(state)))
binding=P/'hand2_holding_v1/compiled_nominal_v2/binding.json';model=P/'hand2_holding_v1/compiled_nominal_v2/model.mjb'
rp=F/'candidate_report.json';write_fixture(rp,dict(schema='functional_tripod_geometric_reference_result:1',model_mjb=pin(model),candidates=entries))
proto=dict(schema=nc['SCHEMA'],CPU_run_authorized=False,training_or_GPU_authorized=False,
    dt_s=nc['DTS'],duration_s=nc['DURATION'],gates={**{k:v for k,v in nc['OLD_GATES'].items() if k!='tail_dual_pad_force_coverage'},'tail_triple_pad_force_coverage':.95},
    old_two_pad_gates=nc['OLD_GATES'],dt_pair_gates=nc['PAIR_GATES'],control_routes=nc['CONTROL_ROUTES'],designated_geom_ids=nc['DESIGNATED'],
    output=str(H/'synthetic_never_execute_v3'),inputs=dict(model_mjb=pin(model),model_binding=pin(binding),candidate_report=pin(rp)),
    source_pins=dict(runner=pin(new)),budget=dict(max_candidates=4,control_routes=3,dt_cases=2,maximum_cases=24,maximum_integration_steps=230400))
pp=F/'protocol.json';write_fixture(pp,proto)
sp=importlib.util.spec_from_file_location('metadata_only_tripod_v2',new);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
_,_,_,chosen,screen=mod.metadata_preflight(pp)
checks['four_metadata_records_two_empty_anchor_failures_retained_two_enrolled']=len(screen)==4 and len(chosen)==2 and [x['eligible_geometry'] for x in screen]==[False,False,True,True]
checks['failed_state_pins_and_unknown_reasons_are_not_dropped']=all(x['source_state_npz']==pin(state) for x in screen) and all(x['reasons'] and x['geometry_unknown'] for x in screen[:2])
# A geometrically enrolled but structurally incomplete anchor payload must still
# fail closed. This separate negative fixture does not modify the positive one.
bad=json.loads((F/'candidate_2.json').read_text());bad['pad_bindings']=[]
bp=F/'bad_qualified_candidate.json';write_fixture(bp,bad)
br=F/'bad_candidate_report.json';write_fixture(br,dict(schema='functional_tripod_geometric_reference_result:1',model_mjb=pin(model),candidates=[dict(candidate=pin(bp),state_npz=pin(state))]))
negative=dict(proto,inputs=dict(proto['inputs'],candidate_report=pin(br)));npth=F/'negative_protocol.json';write_fixture(npth,negative)
try:mod.metadata_preflight(npth);negative_rejected=False
except ValueError as e:negative_rejected='Exact three pad bindings' in str(e)
checks['qualified_missing_three_pad_bindings_still_rejected']=negative_rejected
checks['source_metadata_proof_loaded_no_native_numpy_torch_backend']=not any(k in sys.modules for k in ['mujoco','numpy','torch'])
report=dict(schema='functional_tripod_dynamic_source_preservation:1',status='PASS_AST_SOURCE_ONLY_NOT_PHYSICS_EXECUTION' if all(checks.values()) else 'FAILED_SOURCE_CONTRACT',
    passed=sum(checks.values()),total=len(checks),checks=checks,parent=pin(old),runner=pin(new),checker=pin(__file__),
    changed_physics_equations=False,new_native_executions=0,
    declared_changes=['New three-pad candidate/geometry metadata adapter and no static-qacc admission.',
        'Predeclared ctrl=q/1N/3N normal-preload inverse-PD preparation: one fresh forward and three physical hand-point Jacobians per geometry candidate.',
        'Contact24 decoding unchanged; designated set grows15/28 to15/28/41, with old2pad views derived by counting41 as old nonpad.',
        'Original step/finite/no-aid/warning/loss abort phases,2s tail and thresholds preserved semantically; loop factored with additive parameter/margin traces.',
        'Full-model start/end digest excludes timestep only; original material/gain/solver arrays also saved.',
        'This is not byte-identical whole-loop provenance: extracted contact helper and extra measurements are explicitly new source and require peer review.'])
with (H/'source_preservation_v3.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(report))
if not all(checks.values()):raise SystemExit(1)
