"""Freeze the first MPC feasibility pilot, without importing a native backend."""
from pathlib import Path
import json
from run_mpc_feasibility_v1 import CONFIG,GATES,pin,write

root=Path(__file__).resolve().parent
p=root.parent
fixture=json.loads((root/'two_branch_contract_v1/report.json').read_text(encoding='utf8'))
fp=json.loads(Path(fixture['protocol']['path']).read_text(encoding='utf8'))
binding=json.loads((p/'hand2_holding_v1/compiled_nominal_v2/binding.json').read_text(encoding='utf8'))
protocol={'schema':'native_true_model_MPC_feasibility_protocol:1',
 'CPU_run_authorized':True,'training_or_GPU_authorized':False,'automatic_retry':False,
 'config':CONFIG,'gates':GATES,'output_path':str(root/'feasibility_fine_v1'),
 'inputs':{
  'controller_source':pin(root/'run_mpc_feasibility_v1.py'),
  'branch_source':fp['source_pins']['evaluator'],
  'branch_fixture_report':pin(root/'two_branch_contract_v1/report.json'),
  'candidate_state':fp['inputs']['initial_state'],
  'native_search_report':fp['inputs']['native_search_report'],
  'model_binding':fp['inputs']['model_binding'],'model_mjb':binding['model'],
  'matched_constant_u_report':pin(p/'hand2_constant_u_baseline_v1/cpu_v1/report.json')},
 'initialization':'Exact fixture seed0rank0 qpos/zeroqvel/ctrl, original MJB, one initial forward only. Full integration-state boundary required equal fixture source_state_before.',
 'reference':'Fixed original COM/R/q20 for all100 intervals. No rolling target or object pose reset.',
 'planning':'H2 true-model clones;8 original thumb/index actuators; rest12 fixed. All FD/line steps counted separately. Original horizon safety limits required for execution.',
 'actual':'Only first20ms of selected branch executed, exactly128 native steps; complete intervals must match selected prediction integration state bits. Original2s case and earlyabort limits unchanged.',
 'admission':'Fixture-bound fixed seed0rank0 with all nonacceleration geometry/load/budget/repeat/model gates. Three original static acceleration gates descriptive only, old records unchanged.',
 'qualification':'Single fine-dt bounded oracle only. No two-dt, calibrated dynamics, ego tracking, learned policy or method benefit claim.'}
write(root/'mpc_feasibility_protocol_v1.json',protocol)
print(json.dumps({'status':'FROZEN_NO_MODEL_OR_STEPS','protocol':pin(root/'mpc_feasibility_protocol_v1.json'),'controller':protocol['inputs']['controller_source']}))
