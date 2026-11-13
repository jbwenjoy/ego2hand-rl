"""Nested LP diagnostics: identify failed constraint layer, never task success."""
from pathlib import Path
import argparse,json,sys,tomllib
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.controlled_snapshot import native_controlled_snapshot
from supermimic_research.mechanics.indexed_contact_support import indexed_static_support
from supermimic_research.mechanics.support import static_feasibility


def main(args):
    if args.output.exists():raise ValueError('Fresh nested support diagnosis')
    parent=json.loads(args.preload.read_text());config=tomllib.loads(verify_file_pin(parent['config']).read_text());p=json.loads(verify_file_pin(parent['projection']).read_text());z=np.load(verify_file_pin(p['proposal']));m,d,_,index=native_controlled_snapshot(p['case'])
    d.qpos[:]=z['qpos'];d.qvel[:]=0;d.ctrl[index['actuators']]=z['q20'];d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d)
    body=p['case']['model_layout']['object_body'];parts=[g for g in range(m.ngeom) if m.geom_bodyid[g]==body and (m.geom_contype[g] or m.geom_conaffinity[g])];allowed=[m.geom(f'robot/right_finger{i}_{s}').id for i in [1,2,3] for s in ['link4_col','tip_sensor_col']]
    j=index['joints'];a=index['actuators'];rad=config['preload_target_radius_rad'];lo=np.maximum(z['q20']-rad,np.maximum(m.jnt_range[j,0],m.actuator_ctrlrange[a,0]));hi=np.minimum(z['q20']+rad,np.minimum(m.jnt_range[j,1],m.actuator_ctrlrange[a,1]))
    answer=indexed_static_support(m,d,body,parts,allowed,index['dofs'],a,lo,hi)
    if answer['lp']['success']!=parent['success']:raise ValueError('Repeated frozen LP result mismatch')
    full=static_feasibility(answer['wrench_BODY'],answer['target_BODY'],radius=.125)
    force=static_feasibility(answer['wrench_BODY'],answer['target_BODY'],force_only=True,radius=.125)
    args.output.mkdir();np.savez_compressed(args.output/'basis.npz',wrench_BODY=answer['wrench_BODY'],target_BODY=answer['target_BODY'],hand_loads=answer['loads20'],torque_lower=answer['torque_lower'],torque_upper=answer['torque_upper'],torque_offset=answer['offset'])
    result=dict(schema='pen_nested_frozen_support_diagnosis:1',source=file_pin(__file__),preload=file_pin(args.preload),basis=file_pin(args.output/'basis.npz'),
        bounded_full_wrench_success=answer['lp']['success'],unbounded_actuator_full_wrench_success=full['success'],force_only_success=force['success'],
        force_only_moment_residual_Nm=None if force['witness'] is None else force['witness']['moment_residual_Nm'].tolist(),
        full_wrench_rank=answer['full_wrench_rank'],native_contacts=answer['contacts'],neural_updates=0,native_integration_steps=0,
        scope='Counterfactual removal of actuator and then moment constraints identifies why original LP failed. Force-only is NOT qualification or an alternative support target.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preload',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
