"""Explain recorded impulse residual terms without changing the original gate."""
import argparse
import json
import sys
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    if a.output.exists():raise ValueError('Fresh residual decomposition')
    if len(a.reports)!=len(a.EFC_reviews):raise ValueError('One independent force review per original report')
    a.output.mkdir();cases=[]
    for path,review_path in zip(a.reports,a.EFC_reviews):
        run=json.loads(path.read_text());review=json.loads(review_path.read_text())
        if verify_file_pin(review['run'])!=path.resolve():raise ValueError('Force review binding')
        for c in run['cases']:
            z=np.load(verify_file_pin(c['trace']));p=np.load(verify_file_pin(c['GPU_physical_context']))
            model=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
            e=next(e for e in review['cases'] if e['name']==c['name'] and e['dt']==c['dt']);receipt=np.load(verify_file_pin(e['wrenches']))
            da=c['model_layout']['object_dofadr'];sl=slice(da,da+3);mass=float(p['body_mass'][0,c['model_layout']['object_body']]);dt=c['dt']
            dv=z['post_qvel'][:,sl]-z['pre_qvel'][:,sl];acc=z['post_qacc'][:,sl]
            force=z['post_qfrc_constraint'][:,sl]-z['post_qfrc_bias'][:,sl]+z['post_qfrc_passive'][:,sl]+z['post_qfrc_actuator'][:,sl]
            external=receipt['world_COM_wrench'][:,:3]+mass*model.opt.gravity
            original=mass*dv-dt*external
            integration=mass*(dv-dt*acc);recorded_balance=dt*(mass*acc-force);force_conversion=dt*(force-external)
            checks=dict(original_residual_exact=np.array_equal(original,receipt['linear_impulse_residual_N_s']),
                decomposition_closes=bool(np.max(abs(original-integration-recorded_balance-force_conversion))<1e-12),
                zero_COM=not np.any(c['model_layout']['COM_BODY']),zero_translation_armature=not np.any(p['dof_armature'][0,sl]),
                zero_translation_damping=not np.any(p['dof_damping'][0,sl]),no_applied_force=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']))
            peak=int(np.linalg.norm(original,axis=1).argmax())
            output=a.output/f"x{run.get('candidate_offset_multiplier',1)}_{c['name']}_{dt}.npz"
            np.savez_compressed(output,original_impulse_residual_N_s=original,integration_residual_N_s=integration,
                recorded_force_balance_residual_N_s=recorded_balance,force_conversion_residual_N_s=force_conversion,
                acceleration_force_residual_N=mass*acc-force,velocity_update_residual_m_s=dv-dt*acc,solver_niter=z['post_solver_niter'])
            cases.append(dict(run=file_pin(path),EFC_review=file_pin(review_path),trace=c['trace'],wrenches=e['wrenches'],mode=c['name'],dt=dt,
                multiplier=run.get('candidate_offset_multiplier',1),checks={k:bool(v) for k,v in checks.items()},terms=file_pin(output),
                original_impulse_gate_N_s=5e-7,original_impulse_gate_passed=e['checks']['linear_impulse'],
                peak=dict(tick=peak,control_k=peak//round(.02/dt),nominal_source_time_s=peak*dt,
                    original_residual_N_s=float(np.linalg.norm(original[peak])),integration_residual_N_s=float(np.linalg.norm(integration[peak])),
                    recorded_force_balance_residual_N_s=float(np.linalg.norm(recorded_balance[peak])),
                    force_conversion_residual_N_s=float(np.linalg.norm(force_conversion[peak])),
                    acceleration_force_residual_N=float(np.linalg.norm(mass*acc[peak]-force[peak])),solver_niter=int(z['post_solver_niter'][peak]))))
    passed=sum(sum(c['checks'].values()) for c in cases);total=sum(len(c['checks']) for c in cases)
    write(a.output/'report.json',dict(schema='pen_recorded_impulse_decomposition:1',source=file_pin(__file__),cases=cases,passed=passed,total=total,
        native_integrations=0,neural_updates=0,scope='Algebraic decomposition of original saved impulse residual into recorded velocity-update, acceleration/force balance and force-to-contact/gravity terms. Original host-gravity audit convention retained. No gate relaxation, rerun, solver-root-cause proof or fidelity promotion. Zero-COM/armature/damping assumptions checked.'))
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),checks=[passed,total],failed_original_cases=[c for c in cases if not c['original_impulse_gate_passed']])))
    if passed!=total:raise ValueError('Retain failed decomposition interpretation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,nargs='+',required=True);p.add_argument('--EFC-reviews',dest='EFC_reviews',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
