"""Finite-table endpoint coverage is not whole-task or global feasibility."""
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


def main(a):
    if a.output.exists():raise ValueError('Fresh endpoint coverage audit')
    rows=[];cases=[];angles=[];positions=[];safe_rows=[];joint_rows=[]
    for path in a.reports:
        r=json.loads(path.read_text());base=tomllib.loads(verify_file_pin(r['base_config']).read_text())
        for c in r['cases']:
            ref=np.load(verify_file_pin(c['initialization']['nominal_mean_wrench']));qa=c['model_layout']['object_qposadr'];first=len(rows)
            for decision in c['planning']:
                k=decision['k'];d=np.load(verify_file_pin(decision['prediction']));s=np.load(verify_file_pin(decision['source']))
                endpoint=d['Q'][:,-1,qa:qa+7].astype(float);target=Rotation.from_quat(ref['quaternion_wxyz'][k+1][[1,2,3,0]])
                angle=(Rotation.from_quat(endpoint[:,[4,5,6,3]])*target.inv()).magnitude()
                distance=np.linalg.norm(endpoint[:,:3]-ref['COM_world_m'][k+1],axis=1)
                safe=d['score_feasible'].astype(bool);joint=safe&(distance<=base['COM_max_m'])&(angle<=base['angle_max_rad'])
                q0=s['qpos'][0,qa+3:qa+7];source_error=(Rotation.from_quat(q0[[1,2,3,0]])*target.inv()).magnitude()
                checks=dict(fixed57=endpoint.shape==(57,7),finite_endpoints=bool(np.isfinite(endpoint).all()),valid_unit_quaternions=bool(np.max(abs(np.linalg.norm(endpoint[:,3:],axis=1)-1))<1e-6),
                    selected_finite=bool(d['score_finite'][decision['selected']]),correct_reference_clock=decision['phase']==k+1)
                rows.append(dict(run=file_pin(path),orientation=r['orientation_objective'],candidate_offset_multiplier=r.get('candidate_offset_multiplier',1),mode=c['name'],dt=c['dt'],k=k,checks=checks,
                    safe_candidates=int(safe.sum()),endpoint_joint_pass=int(joint.sum()),selected_endpoint_joint_pass=bool(joint[decision['selected']]),
                    min_safe_COM_m=float(distance[safe].min()) if safe.any() else None,
                    min_safe_full_angle_deg=float(np.rad2deg(angle[safe].min())) if safe.any() else None,
                    safe_full_angle_span_deg=float(np.rad2deg(np.ptp(angle[safe]))) if safe.any() else None,
                    source_to_next_reference_deg=float(np.rad2deg(source_error)),zero_offset_full_angle_deg=float(np.rad2deg(angle[0])),
                    best_safe_improvement_over_zero_deg=float(np.rad2deg(angle[0]-angle[safe].min())) if safe.any() else None,
                    prediction=decision['prediction'],source=decision['source']))
                angles.append(angle);positions.append(distance);safe_rows.append(safe);joint_rows.append(joint)
            segment=rows[first:];holes=[d for d in segment if not d['endpoint_joint_pass']]
            cases.append(dict(run=file_pin(path),orientation=r['orientation_objective'],candidate_offset_multiplier=r.get('candidate_offset_multiplier',1),mode=c['name'],dt=c['dt'],rows=[first,len(rows)],
                decisions=len(segment),endpoint_joint_available=sum(bool(d['endpoint_joint_pass']) for d in segment),
                first_no_joint_candidate=dict(k=holes[0]['k'],source_time_s=holes[0]['k']*.02,endpoint_time_s=(holes[0]['k']+1)*.02) if holes else None,
                max_min_safe_full_angle_deg=max(d['min_safe_full_angle_deg'] for d in segment if d['min_safe_full_angle_deg'] is not None),
                median_best_improvement_over_zero_deg=float(np.median([d['best_safe_improvement_over_zero_deg'] for d in segment if d['best_safe_improvement_over_zero_deg'] is not None])),
                COM_endpoint_gate_m=base['COM_max_m'],full_orientation_endpoint_gate_rad=base['angle_max_rad']))
    a.output.mkdir();np.savez_compressed(a.output/'all57.npz',full_angle_rad=angles,COM_error_m=positions,geometry_safe=safe_rows,joint_endpoint_pass=joint_rows)
    passed=sum(sum(d['checks'].values()) for d in rows);total=sum(len(d['checks']) for d in rows)
    write(a.output/'report.json',dict(schema='native_pen_finite_candidate_endpoint_coverage:1',source=file_pin(__file__),runs=[file_pin(p) for p in a.reports],cases=cases,decisions=rows,all57=file_pin(a.output/'all57.npz'),
        passed=passed,total=total,native_integrations=0,neural_updates=0,
        scope='Every recorded57-action20ms endpoint against original10mm COM and10deg full-orientation max gates plus existing geometry/finite/capacity gates. No RMS/full-duration or continuous-time guarantee. Absence in this finite table is not global robot/reference infeasibility; saved unselected force responses are not independently fidelity-qualified. Zero-offset candidate retains original EMA/reference control, not zero actuation.'))
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),checks=[passed,total],cases=cases)))
    if passed!=total:raise ValueError('Endpoint interpretation check failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
