"""Independent GPU EFC -> local force -> world/BODY wrench/generalized audit."""
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .warp_packet_decode import decode_saved_case
from supermimic_research.artifacts import file_pin,verify_file_pin


def review_saved_warp_contacts(report_path,output,expected_cases):
    args=SimpleNamespace(report=Path(report_path),output=Path(output))
    if args.output.exists():raise ValueError('Fresh receipt review')
    r=json.loads(args.report.read_text());args.output.mkdir();cases=[]
    actual={(c.get('name',c['controller']),c['dt']) for c in r['cases']}
    if actual!=set(expected_cases) or len(r['cases'])!=len(expected_cases):raise ValueError('Explicit unique expected cohort')
    for pin in r['sources']:verify_file_pin(pin)
    for c in r['cases']:
        if c.get('contact_label_schema')!='GPU_constraint_indexed_no_adhesion:1':raise ValueError('Explicit corrected label schema')
        z=np.load(verify_file_pin(c['trace']));m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])));p=np.load(verify_file_pin(c['GPU_physical_context']))
        metrics,arrays=decode_saved_case(c,z,m,p)
        out=args.output/f'{c.get("name",c["controller"])}_{c["dt"]}.npz'
        np.savez_compressed(out,**arrays)
        cases.append(dict(task=c['task'],controller=c['controller'],name=c.get('name',c['controller']),dt=c['dt'],**metrics,wrenches=file_pin(out)))
    checks=dict(expected_cases=len(cases)==len(expected_cases),all_receipt_checks=all(all(c['checks'].values()) for c in cases))
    result=dict(schema='corrected_warp_contact_receipts_review:2',source=file_pin(__file__),decoder_source=file_pin(Path(__file__).with_name('warp_packet_decode.py')),run=file_pin(args.report),checks=checks,cases=cases,
        passed=sum(sum(c['checks'].values()) for c in cases),total=sum(len(c['checks']) for c in cases),neural_updates=0,native_integration_steps=0,
        scope='Actual saved GPU constraint indices/forces, copied-host adhesion difference, moving-COM wrenches, generalized force/torque and free-object linear impulse. Not task success, measured material calibration, or WM gradient validation.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(result))
    if not all(checks.values()):raise ValueError('Preserve failed corrected contact review')
    return result
