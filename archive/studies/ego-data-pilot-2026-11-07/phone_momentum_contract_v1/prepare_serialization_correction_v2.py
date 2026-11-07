"""Retain failed run1, correct only JSON NumPy-bool serialization in a new source."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
source=ROOT/'verify_contract_v1.py';s=source.read_text(encoding='utf8')
needle="    report=dict(schema='phone_momentum_saved_TRAIN_and_analytic_contract_result:1'"
assert s.count(needle)==1
s=s.replace(needle,"    checks={key:bool(value) for key,value in checks.items()}  # JSON scalar normalization only\n"+needle)
with (ROOT/'verify_contract_v2.py').open('x',encoding='utf8') as f:f.write(s)
launcher=(ROOT/'execute_once_v1.py').read_text(encoding='utf8').replace('verify_contract_v1.py','verify_contract_v2.py').replace('protocol_run1.json','protocol_run2.json').replace('process_run1','process_run2').replace('completed_run1','completed_run2')
with (ROOT/'execute_once_v2.py').open('x',encoding='utf8') as f:f.write(launcher)
receipt=dict(schema='momentum_contract_execution_serialization_correction:1',
    reason='Run1 numerical calculations and one input AD completed; report serialization failed on numpy.bool_. No physics or NN models ran.',
    change='Normalize completed check scalars to built-in bool before report serialization; formulas, fixtures, thresholds, TRAIN rows unchanged.',
    failed_source=pin(source),failed_launcher=pin(ROOT/'execute_once_v1.py'),failed_protocol=pin(ROOT/'protocol_run1.json'),
    failed_terminal=pin(ROOT/'process_run1/terminal.json'),failed_partial_report=pin(ROOT/'completed_run1/report.json'),
    failed_completed_numeric_output=pin(ROOT/'completed_run1/TRAIN684_momentum_closure.npz'),
    corrected_source=pin(ROOT/'verify_contract_v2.py'),corrected_launcher=pin(ROOT/'execute_once_v2.py'),core_unchanged=pin(ROOT/'momentum_contract.py'),
    old_files_modified=False,failed_input_AD_Jacobians=1,failed_NN_updates=0,failed_native_calls=0)
with (ROOT/'serialization_correction_v1.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2,allow_nan=False)
print(json.dumps({'correction':pin(ROOT/'serialization_correction_v1.json')}))
