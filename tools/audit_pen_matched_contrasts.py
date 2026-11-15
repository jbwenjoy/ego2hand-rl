"""Finite action effects versus observed repeats, with frozen diagnostic data roles."""
import argparse
import json
import sys
import tomllib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import write
from supermimic_research.learning.pen_native_response import repeated_contrast_spread


def main(a):
    if a.output.exists():raise ValueError('Fresh diagnostic contrast audit')
    r=json.loads(a.report.read_text());cfg=tomllib.loads(verify_file_pin(r['config']).read_text());cp=verify_file_pin(r['config'])
    if r['data_role']!='DIAGNOSTIC_NEVER_FIT':raise ValueError('No relabeling this cohort for fitting')
    gates=tomllib.loads((cp.parent/cfg['gates']).read_text());old=json.loads(verify_file_pin(r['original_run']).read_text());rows=[];checks=[]
    scales=np.r_[np.full(3,gates['object_position_m']),np.full(3,gates['object_rotation_rad']),np.full(3,gates['object_velocity_m_s']),
        np.full(3,gates['object_omega_rad_s']),np.full(3,gates['EFC_force_N']),np.full(3,gates['EFC_torque_Nm'])]
    for anchor in r['anchors']:
        mode,k=anchor['mode'],anchor['k'];c=next(c for c in old['cases'] if c['name']==mode and c['dt']==anchor['dt']);qa,da=c['model_layout']['object_qposadr'],c['model_layout']['object_dofadr']
        source=np.load(verify_file_pin(c['planning'][k]['source']));initial=Rotation.from_quat(source['qpos'][0,qa+3:qa+7][[1,2,3,0]])
        for amplitude in [1,4]:
            records=[e for e in r['candidate_replays'] if e['mode']==mode and e['k']==k and e['amplitude']==amplitude]
            records.sort(key=lambda e:e['repeat']);packets=[dict(np.load(verify_file_pin(e['trace']))) for e in records];Y=[]
            for d in packets:
                q=d['Q'][:,-1];v=d['V'][:,-1];R=Rotation.from_quat(q[:,qa+3:qa+7][:,[1,2,3,0]])
                Y.append(np.c_[q[:,qa:qa+3],(R*initial.inv()).as_rotvec(),v[:,da:da+3],np.einsum('nij,nj->ni',R.as_matrix(),v[:,da+3:da+6]),d['W'].mean(1)])
            Y=np.array(Y)
            for axis in range(20):
                plus,minus=axis+1,axis+21
                pd=packets[0]['PD_targets'];half=(pd[plus]-pd[minus])/2
                same_tables=all(np.array_equal(d['PD_targets'],pd) for d in packets)
                symmetric=np.max(abs(pd[plus]+pd[minus]-2*pd[0]))<2e-7
                pair_checks=dict(three_repeats=len(packets)==3,exact_PD_per_repeat=same_tables,symmetric_PD=symmetric,
                    nonzero_target_span=float(np.linalg.norm(half))>0,finite_response=bool(np.isfinite(Y[:,[plus,minus]]).all()))
                checks.append(pair_checks)
                rows.append(dict(mode=mode,k=k,amplitude=amplitude,axis=axis,PD_half_span_rad=half.tolist(),checks={key:bool(val) for key,val in pair_checks.items()},
                    object12=repeated_contrast_spread(Y[:,plus,:12],Y[:,minus,:12],scales[:12]),
                    mean_world_wrench6=repeated_contrast_spread(Y[:,plus,12:],Y[:,minus,12:],scales[12:]),
                    combined18=repeated_contrast_spread(Y[:,plus],Y[:,minus],scales)))
    passed=sum(sum(c.values()) for c in checks);total=sum(len(c) for c in checks)
    a.output.mkdir();result=dict(schema='pen_matched_finite_contrasts:1',run=file_pin(a.report),source=file_pin(__file__),
        helper=file_pin(ROOT/'src/supermimic_research/learning/pen_native_response.py'),gates=file_pin(cp.parent/cfg['gates']),fixed_scales=scales.tolist(),
        input_role=r['data_role'],pairs=rows,passed=passed,total=total,native_integrations=0,neural_updates=0,
        counts={key:[sum(p[key]['contrast_exceeds_observed_spread'] for p in rows),len(rows)] for key in ['object12','mean_world_wrench6','combined18']},
        scope='Same observed sources,20 matched plus/minus actuator directions at each amplitude,three repeats. Object12 uses WORLD left rotation delta, WORLD linear and endpoint-rotated angular velocities; world-COM mean wrench6. Fixed original replay gates only normalize signal/envelope. Observed three-repeat envelope is not a confidence bound, derivative/JVP qualification, learning data promotion or whole-task feasibility proof. Diagnostic data never fit.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),checks=[passed,total],counts=result['counts'])))
    if passed!=total:raise ValueError('Retain contrast interpretation failure')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
