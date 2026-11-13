"""One full-wrench LP witness and actual native compliance readback."""
from pathlib import Path
import argparse,json,sys,tomllib
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.controlled_snapshot import native_controlled_snapshot
from supermimic_research.mechanics.indexed_contact_support import indexed_static_support
from supermimic_research.mechanics.compound_contacts import CompoundContacts


def main(args):
    if args.output.exists():raise ValueError('Fresh preload output')
    cfg=tomllib.loads(args.config.read_text());pp=(args.config.parent/cfg['projection']).resolve();p=json.loads(pp.read_text());proposal=np.load(verify_file_pin(p['proposal']));m,d,source,index=native_controlled_snapshot(p['case'])
    args.output.mkdir();(args.output/'intent.json').write_text(json.dumps(dict(config=file_pin(args.config),projection=file_pin(pp),neural_updates=0,scope='Full6D static LP witness; native force readback and later dynamics separate.'),indent=2)+'\n')
    d.qpos[:]=proposal['qpos'];d.qvel[:]=0;d.ctrl[index['actuators']]=proposal['q20'];d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d)
    body=p['case']['model_layout']['object_body'];parts=[g for g in range(m.ngeom) if m.geom_bodyid[g]==body and (m.geom_contype[g] or m.geom_conaffinity[g])]
    allowed=[m.geom(f'robot/right_finger{i}_{suffix}').id for i in [1,2,3] for suffix in ['link4_col','tip_sensor_col']]
    j=index['joints'];a=index['actuators'];rad=cfg['preload_target_radius_rad']
    lower=np.maximum(proposal['q20']-rad,np.maximum(m.jnt_range[j,0],m.actuator_ctrlrange[a,0]));upper=np.minimum(proposal['q20']+rad,np.minimum(m.jnt_range[j,1],m.actuator_ctrlrange[a,1]))
    answer=indexed_static_support(m,d,body,parts,allowed,index['dofs'],a,lower,upper)
    readback=None
    if answer['lp']['success']:
        ctrl=answer['control'];d.ctrl[a]=ctrl;mujoco.mj_forward(m,d)
        plane=next(g for g in range(m.ngeom) if int(m.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_PLANE));hand=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('robot/') and (m.geom_contype[g] or m.geom_conaffinity[g])]
        s=CompoundContacts(m,parts,body,{1:hand},plane).read(d,raw=True);net=np.r_[s['force_world_N']+m.body_mass[body]*m.opt.gravity,s['torque_about_COM_world_Nm']]
        readback=dict(net_object_world_wrench=net.tolist(),object_qacc=d.qacc[p['case']['model_layout']['object_dofadr']:p['case']['model_layout']['object_dofadr']+6].tolist(),
            max_hand_qacc_rad_s2=float(abs(d.qacc[index['dofs']]).max()),normal_force_total_N=float(s['normal_by_finger_N'][0]),
            max_penetration_m=s['max_native_penetration_m'],source_object_and_wrist_fixed=True,
            scope='Actual native soft-contact solve at LP control; freely chosen LP coefficients were never applied.')
        np.savez_compressed(args.output/'preload.npz',control20=ctrl,controller_delta20=ctrl-source['initial_qpos'][index['qpos']],
            projected_qpos=proposal['qpos'],geometry_delta20=proposal['delta20'],wrench_BODY=answer['wrench_BODY'],target_BODY=answer['target_BODY'],
            required_torque20=answer['lp']['witness']['required_torque20_Nm'],ray_coefficients=answer['lp']['witness']['ray_coefficients_N'])
    witness=answer['lp']['witness']
    result=dict(schema='controlled_pen_fixed_geometry_preload_LP:1',config=file_pin(args.config),projection=file_pin(pp),
        sources=[file_pin(__file__),file_pin(ROOT/'src/supermimic_research/mechanics/indexed_contact_support.py'),file_pin(ROOT/'src/supermimic_research/mechanics/support.py')],
        success=answer['lp']['success'],status=answer['lp']['status'],message=answer['lp'].get('message'),full_wrench_rank=answer['full_wrench_rank'],contacts=answer['contacts'],
        force_residual_N=None if witness is None else witness['force_residual_N'].tolist(),moment_residual_Nm=None if witness is None else witness['moment_residual_Nm'].tolist(),
        minimum_torque_margin_Nm=None if witness is None else witness['minimum_torque_margin_Nm'],native_readback=readback,
        preload=None if readback is None else file_pin(args.output/'preload.npz'),neural_updates=0,native_integration_steps=0,
        scope='Necessary frozen geometry/torque feasibility only, not compliance equilibrium, dynamic grasp, expert reference or task success.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
