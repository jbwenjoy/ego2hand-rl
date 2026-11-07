"""Source-only integration correction receipt; zero payload/model imports."""
import ast,difflib,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES={
 'core_v2':('src/full151_gradient/boundary_component_final_evaluation105_v2.py','5eea0973150225a3b7cc659f8fbc11562000d98e8db9d98d9c9eafd6105a75e8'),
 'CLI_v2':('scripts/evaluate_full151_boundary_components_final_cpu_v2.py','bb963fcd89a9e61cd22894af4dd653f68f4e54f87bd98093bbb42c5cc01b4511'),
 'checker_v2':('scripts/check_full151_boundary_components_final_cpu_v2.py','60e00c1afb58e00ccae9b2e79d76ee02cdf0ae2f54ecf21e89195b5f0d09aeba'),
 'core_v1_preserved':('src/full151_gradient/boundary_component_final_evaluation105_v1.py','8c652a01128117f7ced84d3ae09e8aad1aafd41aa66caf51c8a408d3e6e75cf8'),
 'prior_source_review':('outputs/full151105_boundary_component_final_evaluation_source_independent_v1/report.json','75791470bc5aea4326882b624463deaad30e0beca3f846fa4655642dd2bbc3d0'),
 'READY_v2':('outputs/full151105_boundary_component_final_evaluation_preparation_CPU_v2/READY.json','5c65d3fb070b7fa133d716f7e69c14fbc06753e8dd369e76607b7c0e45872c86'),
 'actual_training_protocol':('protocols/full151105_boundary_impulse_matched_training_CPU_run_v1.json','9f1a74a553f23d9ecd097a7e94a5381220860434a4701b995af936323fd0ccbe'),
 'statistics':('scripts/describe_boundary_component_predictions_saved_cpu_v1.py','7487e16b819271fd3b6d3d6562d4802fd97216afa4e3f666f705948f88cce036'),
 'stats_pure_READY':('outputs/full151105_boundary_component_statistics_contract_preparation_v1/READY.json','6e9cabefc49dd6d3d847d13eb500771d15fa1ccdb52e54b92f9780af5ce3ea42')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def require(x,s):
    if not x:raise ValueError(s)
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def functions(text):return {v.name:ast.dump(v,include_attributes=False) for v in ast.parse(text).body if isinstance(v,ast.FunctionDef)}

def main():
    out=ROOT/'outputs/full151105_boundary_component_final_evaluation_source_independent_v2';out.mkdir(exist_ok=False);pins={}
    for k,(name,h) in FILES.items():
        p=ROOT/name;require(sha(p)==h,'Current source '+k);pins[k]=pin(p);(out/(k+p.suffix)).write_bytes(p.read_bytes())
    pins['reviewer']=pin(__file__);(out/'executed_reviewer.py').write_bytes(Path(__file__).read_bytes())
    old=(ROOT/FILES['core_v1_preserved'][0]).read_text();new=(ROOT/FILES['core_v2'][0]).read_text();a,b=functions(old),functions(new)
    require(set(a)==set(b),'Same full core function inventory')
    unchanged={k:a[k]==b[k] for k in a if k!='load_final_into'}
    reverted=new.replace('boundary_component_final_evaluation105_v2.py','boundary_component_final_evaluation105_v1.py').replace('evaluate_full151_boundary_components_final_cpu_v2.py','evaluate_full151_boundary_components_final_cpu_v1.py').replace('check_full151_boundary_components_final_cpu_v2.py','check_full151_boundary_components_final_cpu_v1.py')
    new_mapping="        protocol_key='inputs' if key=='input_pins' else key\n        require(set(payload[key])==set(training[protocol_key]) and all(same(v,training[protocol_key][k]) for k,v in payload[key].items()),'Exact source/fit cohort pin map')"
    old_mapping="        require(set(payload[key])==set(training[key]) and all(same(v,training[key][k]) for k,v in payload[key].items()),'Exact source/fit cohort pin map')"
    require(reverted.count(new_mapping)==1,'One explicit namespace correction');reverted=reverted.replace(new_mapping,old_mapping)
    training=read(ROOT/FILES['actual_training_protocol'][0]);ready=read(ROOT/FILES['READY_v2'][0]);pp=ready['cpu_report'];require(sha(pp['path'])==pp['sha256'],'Actual new proof hash');proof=read(pp['path'])
    for v in ready['source_pins'].values():require(sha(v['path'])==v['sha256'],'Current proof source')
    failure_path=ROOT/'outputs/full151105_boundary_component_final_evaluation_CPU_v1/failure.json';failure=read(failure_path);counts=failure['counts']
    mapping=proof['actual_training_protocol_metadata_ABI'];checker=(ROOT/FILES['checker_v2'][0]).read_text()
    checks={
      'reverse_only_OWNroutes_and_one_input_field_mapping_text_exact_v1':reverted==old and all(unchanged.values()),
      'actual_protocol_uses_inputs_not_payload_input_pins_and_caller_passes_real_protocol':training['schema']=='full151105_boundary_impulse_matched_training_CPU_protocol:1' and 'inputs' in training and 'input_pins' not in training and "'training':train" in new and "m['training'],torch)" in new,
      'new_toy_schema_is_actual_protocol_shaped_and_separates_checkpoint_alias':proof['passed']==proof['total']==len(proof['checks'])==3 and all(v is True for v in proof['checks'].values()) and mapping['checkpoint_input_field']=='input_pins' and mapping['protocol_input_field']=='inputs' and mapping['top_level_keys']==sorted(training) and mapping['pin']==pin(ROOT/FILES['actual_training_protocol'][0]) and "'input_pins':training['inputs']" in checker,
      'old_actual_failure_zero_FW_preserved_not_prediction_evaluation':counts['own_trusted_torch_loads']==3 and counts['native_WM_constructors']==1 and counts['raw_deepcopies']==2 and all(counts[k]==0 for k in ('native_FW_attempted','native_FW_returned','aux_head_FW','predicted_rows')) and 'input_pins' in failure['error'],
      'scientific_equations_precision_budget_outputs_and_statistics_unchanged':all(unchanged.values()) and "'native_FW_attempted':3,'native_FW_returned':3,'aux_head_FW':2,'predicted_rows':456" in new and sha(ROOT/FILES['statistics'][0])==FILES['statistics'][1],
      'reviewer_source_JSON_AST_only_no_actual_arrays_payloads_models':not any(k.split('.')[0] in ('numpy','torch','mujoco','warp','dex_supermimic','mjlab') for k in sys.modules)}
    require(all(checks.values()),'Source integration review failed')
    (out/'v1_to_v2_core.diff').write_text('\n'.join(difflib.unified_diff(old.splitlines(),new.splitlines(),fromfile='preserved_core_v1',tofile='core_v2')),encoding='utf-8')
    defect={'schema':'boundary_final_evaluation_v1_integration_defect:1','old_failure':pin(failure_path),'old_source_review':pins['prior_source_review'],'cause':'Checkpoint payload input_pins was incorrectly indexed using the same name in the actual training protocol, whose field is inputs. Original Toy invented input_pins on the protocol and masked this mismatch. Root, producer owner and independent source reviewer all missed this integration defect.','effect':'Three own loads, constructor1/copies2, then failure before every native/head FW. Prepared inputs are not model predictions.','correction':'New v2 explicitly maps payload input_pins to protocol inputs, while source_pins remains source_pins; exact samefile/SHA map validation retained. All other function ASTs/scientific equations and budgets unchanged.','scope':'New real-protocol-keyset-shaped Toy3 evidence is delegated, not rerun here. No actual v2 execution/accuracy/physical qualification from this receipt.'}
    write(out/'v1_integration_defect_addendum.json',defect)
    report={'schema':'full151105_boundary_component_final_evaluation_source_independent:2','status':'NO_KNOWN_SOURCE_BLOCKER_AFTER_EXPLICIT_PROTOCOL_FIELD_CORRECTION','source_review_qualified':True,'checks':checks,'passed':len(checks),'total':len(checks),'reviewed_source_pins':pins,'unchanged_function_AST':unchanged,'producer_cpu_proof':pp,'defect_addendum':pin(out/'v1_integration_defect_addendum.json'),'known_source_blockers':[],'actual_evaluation_accuracy_or_physical_qualified':False,'reviewer_actual_arrays_weights_Torch_models_FW_AD_MJB_Env_physics_GPU':0,'limits':read(ROOT/FILES['prior_source_review'][0])['limits']}
    for v in pins.values():require(sha(v['path'])==v['sha256'],'Sources unchanged')
    write(out/'report.json',report);print(json.dumps({'report':pin(out/'report.json'),'defect':pin(out/'v1_integration_defect_addendum.json'),'passed':len(checks),'total':len(checks)},indent=2))
if __name__=='__main__':main()
