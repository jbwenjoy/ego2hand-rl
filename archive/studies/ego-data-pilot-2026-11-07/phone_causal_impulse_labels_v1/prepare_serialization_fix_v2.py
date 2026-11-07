"""Keep first numeric export; correct only report NumPy-bool serialization."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
source=H/'export_saved_labels_v1.py';s=source.read_text(encoding='utf8')
needle="    report=dict(schema='causal_phone22_saved_impulse_export_result:1'"
assert s.count(needle)==1
s=s.replace(needle,"    checks={key:bool(value) for key,value in checks.items()}  # JSON scalar conversion only\n"+needle)
s=s.replace("out=H/'completed_v1';protocol=H/'protocol_v1.json'","out=H/'completed_v2';protocol=H/'protocol_v2.json'")
with (H/'export_saved_labels_v2.py').open('x',encoding='utf8') as f:f.write(s)
receipt=dict(schema='causal_impulse_export_serialization_correction:2',
    reason='All2942 saved branches and numeric outputs were processed; JSON report failed on numpy.bool_ in an analytic check.',
    changes='Convert completed check scalars to Python bool; all formulas/inputs/rows/thresholds unchanged, newsource/newoutput. No native or learning rerun.',
    failed_observed_process_exit_code=1,old_source=pin(source),old_protocol=pin(H/'protocol_v1.json'),old_partial_report=pin(H/'completed_v1/report.json'),
    old_numeric_output=pin(H/'completed_v1/all2952_pair22_physical_labels.npz'),old_identities=pin(H/'completed_v1/row_identities.json'),new_source=pin(H/'export_saved_labels_v2.py'),
    expected_repeated_saved_trace_reads=2942,old_native_NN_calls=0)
with (H/'serialization_correction_v2.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2,allow_nan=False)
print(json.dumps({'correction':pin(H/'serialization_correction_v2.json')}))
