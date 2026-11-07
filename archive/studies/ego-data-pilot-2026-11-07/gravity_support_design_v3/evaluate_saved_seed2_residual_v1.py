"""One authorized new LP over existing saved seed2 operands, no model/FK."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from support_residual_lp_v3 import support_residual_lp

HERE=Path(__file__).resolve().parent
P=HERE.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

raw=P/'hand2_holding_v1/support_preparation_v1/actual_v1/support_operands_and_controls.npz'
parent=P/'hand2_holding_v1/support_preparation_v1/actual_v1/report.json'
source=HERE/'support_residual_lp_v3.py'
assert pin(source)['sha256']=='371fd53ba989345164f14f664954768c552c43e30aaf68fd62fd010e26d22237'
prior=json.loads(parent.read_text())
assert pin(raw)==prior['outputs'][raw.name]
with np.load(raw,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
r=support_residual_lp(a['object_contact_points_world'],a['normals_force_on_object'],
    a['object_COM_world'],.2,.7,a['J_hand3x20'],a['qfrc_bias20'],a['torque_lower'],a['torque_upper'],normal_cap_N=3.)
assert not any(k in sys.modules for k in ['mujoco','torch','warp','mujoco_warp'])
out={'schema':'saved_seed2_support_residual_diagnostic:1',
    'status':'COMPLETED_SAVED_OPERAND_DIAGNOSTIC_NOT_NEW_POSE_QUALIFICATION',
    'source_pins':{'caller':pin(Path(__file__)),'residual_LP':pin(source),'parent_support_report':pin(parent),'saved_operands':pin(raw)},
    'frozen_thresholds_unchanged':True,'no_new_pose_or_normal_or_contact_selection':True,
    'counts':{'residual_function_calls':1,'actual_LP_solves':r['LP_calls'],'model_loads':0,'FK_calls':0,'integration_steps':0},
    'result':r,'scope':'Prior nominal0.2kg/mu0.7 seed2 nearest-pair/J/bias/bounds reused byte-for-byte from saved preparation. Not a new candidate, calibration, strict static support or holding result.'}
dest=HERE/'actual_saved_seed2_residual_v1.json'
with dest.open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
summary={'report':pin(dest),'feasible_force_balance':r['feasible_force_balance'],'moment_infinity_Nm':r['moment_infinity_Nm'],'normal_force_N':r.get('normal_force_N'),'proposal_gate':r['proposal_gate'],'LP_solves':r['LP_calls']}
if r.get('torque_slack_lower_Nm') is not None:summary['minimum_torque_slack_Nm']=min(*r['torque_slack_lower_Nm'],*r['torque_slack_upper_Nm'])
if r['force_recovery']:summary['force_recovery']=r['force_recovery']
print(json.dumps(summary))
