"""Original moving-reference BODY wrench targets and frozen-contact diagnostic."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.controlled_snapshot import native_controlled_snapshot
from supermimic_research.mechanics.reference_contact_wrench import reference_contact_wrench
from supermimic_research.mechanics.support import static_feasibility
from supermimic_research.mechanics.wrench_separation import separating_witness


def main(args):
    if args.output.exists():raise ValueError('Fresh original-clock inverse dynamics')
    p=json.loads(args.projection.read_text());m,d,z,_=native_controlled_snapshot(p['case']);body=p['case']['model_layout']['object_body'];principal=Rotation.from_quat(m.body_iquat[body][[1,2,3,0]]).as_matrix();I=principal@np.diag(m.body_inertia[body])@principal.T
    result=reference_contact_wrench(z['COM_world_m'],z['quaternion_wxyz'],m.body_mass[body],I,m.opt.gravity,.02)
    basis=np.load(args.basis);target=result['wrench_BODY_COM'][0];lp=static_feasibility(basis['wrench_BODY'],target,radius=.125)
    certificate=separating_witness(basis['wrench_BODY'],target,radius_m=.125) if not lp['success'] else None
    args.output.mkdir();arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
    if lp['success']:arrays['first_dynamic_ray_coefficients_N']=lp['witness']['ray_coefficients_N']
    np.savez_compressed(args.output/'targets.npz',**arrays,inertia_BODY=I,mass_kg=m.body_mass[body],gravity_world=m.opt.gravity)
    report=dict(schema='nominal_controlled_pen_reference_contact_wrench:1',projection=file_pin(args.projection),basis=file_pin(args.basis),
        sources=[file_pin(__file__),file_pin(ROOT/'src/supermimic_research/mechanics/reference_contact_wrench.py'),file_pin(ROOT/'src/supermimic_research/mechanics/support.py'),file_pin(ROOT/'src/supermimic_research/mechanics/wrench_separation.py')],
        targets=file_pin(args.output/'targets.npz'),valid_frames=len(result['frame_indices']),original_frames=len(z['time_s']),
        estimator=result['estimator'],mass_kg=float(m.body_mass[body]),inertia_BODY=I.tolist(),
        first_dynamic_target_BODY=target.tolist(),first_static_target_BODY=basis['target_BODY'].tolist(),
        first_dynamic_wrench_cone_feasible=lp['success'],first_dynamic_separating_certificate=certificate,
        initial_velocity_estimator_difference_m_s=(result['velocity_world'][0]-z['initial_qvel'][p['case']['model_layout']['object_dofadr']:p['case']['model_layout']['object_dofadr']+3]).tolist(),
        neural_updates=0,native_integration_steps=0,
        scope='Nominal simulator inertia/gravity and finite-difference reference acceleration, not measured source forces or calibrated ego dynamics. Frozen first-frame cone feasibility is necessary only; moving-hand torque bounds/contact evolution/derivative uncertainty remain.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--projection',type=Path,required=True);p.add_argument('--basis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
