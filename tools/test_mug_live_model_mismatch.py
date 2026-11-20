"""Test live feedback under mass/friction mismatch using an unchanged nominal predictor."""
import argparse
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from evaluate_mug_matched_generation import physics,file_pin,verify_file_pin,write_json


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    source=file_pin(root/'mug_matched_generation_v1/cross/report.json')
    cross=physics.read_json(source)
    frozen=physics.read_json(cross['freeze'])
    config=physics.read_json(frozen['intent'])
    inputs=config['inputs']
    path=verify_file_pin(inputs['model'])
    bind=physics.read_json(inputs['binding'])
    ref=physics.load_arrays(inputs['reference'])
    base=physics.load_arrays(inputs['base'])['ctrl']
    out.mkdir(exist_ok=False,parents=True)
    write_json(out/'intent.json',dict(source=source,code=file_pin(__file__),seeds=config['seeds'],scope=
        'Nominal planner in all three evaluator worlds. Each control copies the measured integration state into a private nominal model, evaluates all archived 12 x 3 command branches and applies only the first command. Exact nominal identity check; compare with frozen nominal commands on each evaluator. Full simulator state is privileged.'))
    rows=[]
    for record in frozen['rows']:
        if record['model']!='nominal':
            continue
        grid=physics.load_arrays(record['grid'])['ctrl']
        gold=physics.load_arrays(record['trace'])
        for env,(mass,mu) in config['evaluators'].items():
            plant=mujoco.MjModel.from_binary_path(str(path))
            m=mujoco.MjModel.from_binary_path(str(path))
            body,qa=bind['mug_body_id'],bind['mug_free_qpos']
            mug=np.isin(np.arange(m.ngeom),bind['mug_part_geom_ids'])
            hand=np.isin(np.arange(m.ngeom),[m.geom(n).id for n in bind['hand_collision_names']])
            a,b=plant.pair_geom1,plant.pair_geom2
            pairs=(mug[a]&hand[b])|(hand[a]&mug[b])
            assert pairs.sum()==mug.sum()*hand.sum()
            plant.pair_friction[pairs,:2]*=mu
            actual=mujoco.MjData(plant)
            if mass!=1.:
                plant.body_mass[body]*=mass
                plant.body_inertia[body]*=mass
                mujoco.mj_setConst(plant,actual)
            actual.qpos[:],actual.qvel[:],actual.ctrl[:]=ref['initial_qpos27'],ref['initial_qvel26'],base[0]
            mujoco.mj_forward(plant,actual)
            sim=mujoco.MjData(m)
            signature=mujoco.mjtState.mjSTATE_INTEGRATION
            snapshot=np.empty(mujoco.mj_stateSize(m,signature))
            table=m.geom('mug_task_table').id
            force=np.zeros(6)
            desired=Rotation.from_quat(ref['quaternion_wxyz'][:,[1,2,3,0]])
            controls=[];ids=[];qpos=[]
            start=time.perf_counter()
            for phase,candidates in enumerate(grid):
                assert not actual.xfrc_applied.any() and not actual.qfrc_applied.any()
                mujoco.mj_getState(plant,actual,snapshot,signature)
                costs=[]
                for candidate in candidates:
                    mujoco.mj_setState(m,sim,snapshot,signature)
                    cost=0.
                    for j,ctrl in enumerate(candidate):
                        sim.ctrl[:]=ctrl
                        hit=False
                        for _ in range(64):
                            mujoco.mj_step(m,sim)
                            g,h=sim.contact.geom.T
                            for ic in np.flatnonzero(((g==table)&mug[h])|((h==table)&mug[g])):
                                mujoco.mj_contactForce(m,sim,ic,force)
                                hit |= force[0]>.1
                        mujoco.mj_kinematics(m,sim)
                        rot=Rotation.from_quat(sim.qpos[qa+np.array([4,5,6,3])])
                        com=sim.qpos[qa:qa+3]+rot.apply(ref['COM_BODY_m'])
                        idx=phase+j+1
                        mm=1000*np.linalg.norm(com-ref['COM_world_m'][idx])
                        deg=np.rad2deg((rot*desired[idx].inv()).magnitude())
                        cost+=(mm/10)**2+(deg/10)**2+100*hit+.01*np.mean(((ctrl-base[idx])/.12)**2)
                    costs.append(cost)
                choice=int(np.argmin(costs))
                command=candidates[choice,0]
                actual.ctrl[:]=command
                mujoco.mj_step(plant,actual,nstep=64)
                ids.append(choice);controls.append(command.copy());qpos.append(actual.qpos.copy())
            elapsed=time.perf_counter()-start
            detail,trace=physics.replay(path,bind,ref,base[0],controls,mass,mu)
            np.testing.assert_array_equal(trace['qpos'],qpos)
            if env=='nominal':
                np.testing.assert_array_equal(ids,gold['chosen'])
                np.testing.assert_array_equal(qpos,gold['qpos'])
            prior=next(r for r in cross['rows'] if r['generator']=='nominal' and r['evaluator']==env and r['seed']==record['seed'])
            artifact=out/f"{record['seed']}_{env}.npz"
            np.savez_compressed(artifact,ctrl=controls,chosen=ids,**{k:v for k,v in trace.items() if k!='ctrl'})
            rows.append(dict(seed=record['seed'],env=env,feedback=detail,frozen_s=prior['valid_s'],
                             different_choices=int(np.sum(ids!=gold['chosen'])),planning_ticks=len(grid)*12*3*64,
                             actual_ticks=len(grid)*64,wall_s=elapsed,trace=file_pin(artifact)))
            write_json(out/'progress.json',rows)
            print(record['seed'],env,detail['valid_s'],flush=True)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    run(p.parse_args().out.resolve())
