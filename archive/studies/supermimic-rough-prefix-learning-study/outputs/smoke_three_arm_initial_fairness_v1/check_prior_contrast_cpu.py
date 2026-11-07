"""Saved-only P/A2 binding/formula contrast; no environment or device imports."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
cfg={m:read(ROOT/f'configs/{n}-engineering_smoke-v2.json') for m,n in [('A1L','a1l'),('P_ONLY','p_only'),('A2','a2')]}
meta={m:read(ROOT/f'outputs/smoke_{n}_v2/contact_objective_metadata.json') for m,n in [('P_ONLY','p_only'),('A2','a2')]}
checks={}
checks['A1L_no_prior_binding_or_metadata']=cfg['A1L']['contact_binding'] is None and not (ROOT/'outputs/smoke_a1l_v2/contact_objective_metadata.json').exists()
checks['P_A2_same_contact_spec_pin']=cfg['P_ONLY']['contact_binding']==cfg['A2']['contact_binding']
checks['P_A2_same_all_execution_target_source_pins']=cfg['P_ONLY']['source_pins']==cfg['A2']['source_pins']
checks['P_A2_configuration_only_method_output_differ']={k for k in cfg['A2'] if cfg['A2'][k]!=cfg['P_ONLY'][k]}=={'method','output'}
spec=read(cfg['A2']['contact_binding']['path']);checks['spec_pin_exact']=pin(cfg['A2']['contact_binding']['path'])==cfg['A2']['contact_binding']
checks['same_actual_bound_targets_model_context_code']=all(meta[m].get(k)==v for m in meta for k,v in spec['expected_hashes'].items())
checks['same_geometry_basis_penetration']=all(meta['P_ONLY'][k]==meta['A2'][k] for k in ('geometry','basis','penetration_prior','phase_weights','known_semantics','known_frame_count','unknown_frame_count'))
checks['P_only_direct_shared_penalty']=meta['P_ONLY']['formula']=='-.1*penetration_prior' and meta['P_ONLY']['support_coefficient']==0. and meta['P_ONLY']['penetration_coefficient']==.1 and meta['P_ONLY']['base_A2_scoring']==meta['A2']['scoring'] and meta['P_ONLY']['implementation']=='A2 full forward retained; support output is not used in the reward graph'
checks['A2_same_P_adds_union_support']=meta['A2']['formula']=='phase_weight * support_match - .1*penetration_prior' and meta['A2']['scoring']['support_coefficient']==1. and meta['P_ONLY']['scoring']['support_coefficient']==0.
checks['both_external_dt_and_failed_qualification_retained']=all(v['dt_applied_inside_module'] is False and v['caller_dt_s']==.02 and v['human_contact_observation_known'] is False and v['target_defined_from_policy_contacts'] is False and v['physical_reference_qualified'] is False and v['local_proxy_qualification']=='FAILED_RETAINED_EXPLORATORY_ONLY' and v['compiled_precision_qualification']=='FAILED_RETAINED_NOT_REVALIDATED' for v in meta.values())
native=ROOT/'src/rough_prefix/native_v2.py';tree=ast.parse(native.read_text(encoding='utf-8-sig'));prior=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='imagination_prior');shaped=next(n for n in prior.body if isinstance(n,ast.FunctionDef) and n.name=='shaped');ret=next(n for n in shaped.body if isinstance(n,ast.Return))
checks['source_imagination_dt_exactly_once']=ast.unparse(ret.value.elts[0])=='reward + 0.02 * bonus' and sum(isinstance(n,ast.BinOp) and isinstance(n.op,ast.Mult) and ast.unparse(n)=='0.02 * bonus' for n in ast.walk(prior))==1
checks['audited_native_source_pin_exact']=all(any(v==pin(native) for v in c['source_pins'].values()) for c in cfg.values())
result={'schema':'rough_prefix_three_arm_prior_contrast_saved_review:1','status':'PASS_FROZEN_PRIOR_CONTRAST' if all(checks.values()) else 'FAIL','checks':checks,'passed':sum(checks.values()),'total':len(checks),'source':pin(__file__),'spec':cfg['A2']['contact_binding'],'metadata':{m:pin(ROOT/f'outputs/smoke_{n}_v2/contact_objective_metadata.json') for m,n in [('P_ONLY','p_only'),('A2','a2')]},'scope':'Saved-input/source/formula contrast only. Prior applied only in imagination; this doesnot infer physical/task benefit. Constructor inputs, initialparameter bytes and actual8 equality are separately recorded in report.json. No H0/T0 policy contacts define this target.','GPU_used':False,'environment_created':False,'physics_steps':0}
out=Path(__file__).with_name('prior_contrast_bindings.json')
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':result['status'],'passed':result['passed'],'total':result['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(out)},indent=2))
raise SystemExit(int(not all(checks.values())))
