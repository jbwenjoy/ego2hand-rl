"""Descriptive actual tracking/nominal force tradeoffs and sampled authority."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation,Slerp

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    r=json.loads(a.report.read_text());review=json.loads(a.review.read_text())
    if verify_file_pin(review['run'])!=a.report.resolve() or a.output.exists():raise ValueError('Exact review and fresh diagnostics')
    efc=json.loads(verify_file_pin(review['EFC']).read_text());cases=[]
    for c in r['cases']:
        z=np.load(verify_file_pin(c['trace']));ref=np.load(verify_file_pin(c['initialization']['nominal_mean_wrench']));table=np.load(verify_file_pin(c['fixed_model_arrays']))
        e=next(e for e in efc['cases'] if e['name']==c['name'] and e['dt']==c['dt']);w=np.load(verify_file_pin(e['wrenches']))['world_COM_wrench']
        n=round(.02/c['dt']);qa,da=c['model_layout']['object_qposadr'],c['model_layout']['object_dofadr'];mass=float(table['body_mass'][0,c['model_layout']['object_body']])
        rows=[]
        for d in c['planning']:
            k=d['k'];packet=np.load(verify_file_pin(d['prediction']));source=np.load(verify_file_pin(d['source']));actual_mean=w[k*n:(k+1)*n].mean(0)
            error=z['post_COM'][(k+1)*n-1]-ref['COM_world_m'][k+1]
            refR=Rotation.from_quat(ref['quaternion_wxyz'][k+1][[1,2,3,0]]).as_matrix();actualR=Rotation.from_quat(z['post_quaternion_wxyz'][(k+1)*n-1][[1,2,3,0]]).as_matrix()
            axial=float(error@refR[:,0]);transverse=float(np.linalg.norm(error-axial*refR[:,0]));shaft=float(np.rad2deg(np.arccos(np.clip(actualR[:,0]@refR[:,0],-1,1))))
            position_candidates=packet['Q'][:,-1,qa:qa+3];diameter=float(np.linalg.norm(np.ptp(position_candidates,axis=0)))
            sampled_best=float(np.linalg.norm(position_candidates-ref['COM_world_m'][k+1],axis=1).min())
            velocity_error=ref['velocity_world'][k]-source['qvel'][0,da:da+3]
            rows.append(dict(k=k,time_s=(k+1)*.02,COM_error_m=float(np.linalg.norm(error)),axial_error_m=axial,transverse_error_m=transverse,shaft_error_deg=shaft,
                mean_world_wrench=actual_mean.tolist(),nominal_world_wrench=ref['wrench_world'][k].tolist(),
                nominal_force_error_N=float(np.linalg.norm(actual_mean[:3]-ref['wrench_world'][k,:3])),nominal_torque_error_Nm=float(np.linalg.norm(actual_mean[3:]-ref['wrench_world'][k,3:])),
                sampled_COM_envelope_diagonal_m=diameter,sampled_best_terminal_COM_error_m=sampled_best,
                source_velocity_error_m_s=float(np.linalg.norm(velocity_error)),missing_velocity_correction_force_N=float(np.linalg.norm(mass*velocity_error/.02))))
        cases.append(dict(mode=c['name'],dt=c['dt'],rows=rows,actual_qualified=c['summary']['qualified']))
    common=[]
    for dt in [.005,.0025]:
        pair=[c for c in cases if c['dt']==dt];K=min(len(c['rows']) for c in pair)
        common.append(dict(dt=dt,controls=K,physical_s=K*.02,metrics={c['mode']:{key:float(np.mean([v[key] for v in c['rows'][:K]])) for key in
            ['COM_error_m','nominal_force_error_N','nominal_torque_error_Nm','source_velocity_error_m_s','missing_velocity_correction_force_N']} for c in pair}))
    a.output.mkdir();result=dict(schema='pen_native_search_tracking_diagnostics:1',run=file_pin(a.report),review=file_pin(a.review),source=file_pin(__file__),cases=cases,common_prefix=common,
        neural_updates=0,native_integrations=0,scope='All recorded actual errors and sampled57-candidate endpoint envelopes. Sampled minima are not global reachability bounds. Nominal feedforward force follows reference acceleration and does not itself correct current position/velocity error; suggested feedback interpretation is a hypothesis, not a proven cause. All numerical/task failures remain.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),common_prefix=common,
        last=[dict(mode=c['mode'],dt=c['dt'],**c['rows'][-1]) for c in cases])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
