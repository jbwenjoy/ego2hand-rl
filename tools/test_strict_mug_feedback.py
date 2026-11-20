"""Robust closed-loop tracking of two frozen strict trajectories without access to plant parameters."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from repair_mug_path_geometry import StrictCheck,LIMITS,h3
from supermimic_research.mechanics.mug_step_response import MugStepChart
from supermimic_research.mechanics.mug_local_feedback import tangent_errors


VARIANTS={'nominal':(1,1,1,0),'mass105':(1.05,1,1,0),'mass095':(.95,1,1,0),
    'mu095':(1,.95,1,0),'mu105':(1,1.05,1,0),'heavy_low':(1.15,.8,1,0),
    'time075':(1,1,.75,0),'time125':(1,1,1.25,0),'start_x_plus_mm':(1,1,1,.001),
    'start_x_minus_mm':(1,1,1,-.001),'cross_a':(1.025,.975,1,-.0005),'cross_b':(.975,1.025,1,.0005)}
ARMS=('open','hand_pd','kinematic','native_gain','shared_gain')


def plant(asset,bind,mass,mu,tau):
    model=mujoco.MjModel.from_binary_path(str(asset));body=bind['mug_body_id']
    hand=np.isin(np.arange(model.ngeom),[model.geom(n).id for n in bind['hand_collision_names']])
    mug=model.geom_bodyid==body;a,b=model.pair_geom1,model.pair_geom2
    mask=(hand[a]&mug[b])|(mug[a]&hand[b])
    assert mask.sum()==len(bind['hand_collision_names'])*len(bind['mug_part_geom_ids'])
    model.pair_friction[mask,:2]*=mu;model.pair_solref[mask,0]*=tau
    if mass!=1.:
        model.body_mass[body]*=mass;model.body_inertia[body]*=mass;mujoco.mj_setConst(model,mujoco.MjData(model))
    return model


def correct(name,chart,live,reference,dx,fitted,i):
    if name=='open' or np.max(abs(dx))<1e-10:return np.zeros(chart.model.nu)
    if name in ('native_gain','shared_gain'):
        return -(fitted['K_native' if name=='native_gain' else 'K_shared'][i]@dx)
    m=chart.model;dq=-.20*dx[chart.dofs]-.004*dx[m.nv+chart.dofs]
    if name=='hand_pd':return dq
    p,R,v,wb=chart.motion(live);pref,Rref,vref,wbref=chart.motion(reference)
    d=chart.data;d.qpos[:]=live[1:1+m.nq];d.qvel[:]=live[1+m.nq:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    linear=.5*(pref-p)+.01*(vref-v)
    angular=.5*(Rref*R.inv()).as_rotvec()+.01*(Rref.apply(wbref)-R.apply(wb))
    offsets=linear+np.cross(angular,d.site_xpos[chart.sites]-p)
    matrices=[]
    for site in chart.sites:
        jp=np.zeros((3,m.nv));mujoco.mj_jacSite(m,d,jp,None,site);matrices.append(jp[:,chart.dofs])
    J=np.concatenate(matrices)
    return dq+np.linalg.solve(J.T@J+.007**2*np.eye(m.nu),J.T@offsets.ravel())


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    fit=h3.file_pin(root/'mug_nominal_tracking_gains_v1/report.json')
    models=h3.physics.read_json(fit);cfg=h3.physics.read_json(models['intent'])
    old=h3.physics.load_arrays(cfg['inputs']['reference']);bind=h3.physics.read_json(cfg['inputs']['binding'])
    asset=h3.verify_file_pin(cfg['inputs']['model']);nominal=mujoco.MjModel.from_binary_path(str(asset))
    chart=MugStepChart(nominal,bind)
    previous=h3.file_pin(root/'mug_strict_stress_v1/report.json');prior=h3.physics.read_json(previous)
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(fit=fit,baseline=previous,variants=VARIANTS,arms=ARMS,cap_rad=.08,repetitions=1,
        code=[h3.file_pin(__file__),h3.file_pin(Path(__file__).with_name('repair_mug_path_geometry.py'))],
        scope='Use only measured live qpos/qvel plus known nominal trajectory and phase in feedback. Neither actual perturbed inertia/friction nor future realized state enters a decision. Five fixed controllers; clip final PD and bound correction to 0.08 rad per joint. Strict same six gates. Frozen before trials. Primary: retain both full nominal paths and positive median paired gain across eleven nonnominal scenarios for each clip. Cross_a/b are two new combinations. No target retiming or object actuation.'))
    results=[]
    for clip,folder in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
        fitted=h3.physics.load_arrays(models['clips'][str(clip)]['data'])
        item,=h3.physics.read_json(h3.file_pin(root/folder/'report.json'))['rows'];ref=h3.physics.load_arrays(item['reference'])
        trace=h3.physics.load_arrays(h3.physics.read_json(cfg['paths'][str(clip)])['trace'])
        T=ref['body']
        for condition,(mass,mu,tau,offset) in VARIANTS.items():
            m=plant(asset,bind,mass,mu,tau)
            sig=mujoco.mjtState.mjSTATE_FULLPHYSICS
            def state(d):
                s=np.empty(mujoco.mj_stateSize(m,sig));mujoco.mj_getState(m,d,s,sig);return s
            P=SimpleNamespace(m=m,binding=bind,mug=m.geom_bodyid==bind['mug_body_id'],table=m.geom('mug_task_table').id,
                chart=MugStepChart(m,bind),state=state,pos=T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'],rot=Rotation.from_matrix(T[:,:3,:3]))
            audit=StrictCheck(P)
            def initial():
                d=mujoco.MjData(m);d.qpos[:]=old['initial_qpos27'];d.qvel[:]=old['initial_qvel26'];d.ctrl[:]=old['feedforward_PD_target20'][0]
                qa=chart.qa
                d.qpos[qa:qa+3]+=Rotation.from_quat(d.qpos[qa+np.array([4,5,6,3])]).apply([offset,0,0])
                mujoco.mj_forward(m,d);return d
            for arm in ARMS:
                d=initial();qs=[];commands=[];metrics=[];saturated=0;wall=0.
                for i,cmd0 in enumerate(fitted['cmd']):
                    start=perf_counter();s=state(d);refstate=fitted['state'][i];dx=tangent_errors(nominal,refstate,s[None])[0]
                    change=correct(arm,chart,s,refstate,dx,fitted,i)
                    assert np.isfinite(change).all()
                    cmd=np.clip(cmd0+np.clip(change,-.08,.08),*nominal.actuator_ctrlrange.T)
                    wall+=perf_counter()-start;saturated+=int(np.max(abs(change))>.08)
                    measure=audit.step(d,i,cmd);metrics.append(measure);qs.append(d.qpos.copy());commands.append(cmd)
                    if np.any(measure>LIMITS):break
                accepted=len(qs)-int(np.any(metrics[-1]>LIMITS))
                replay=initial()
                for i,cmd in enumerate(commands):
                    np.testing.assert_array_equal(audit.step(replay,i,cmd),metrics[i]);np.testing.assert_array_equal(replay.qpos,qs[i])
                if condition=='nominal':
                    np.testing.assert_array_equal(qs,trace['qpos']);np.testing.assert_array_equal(commands,fitted['cmd'])
                if arm=='open' and condition not in ('cross_a','cross_b'):
                    gold=next(r for r in prior['rows'] if r['clip']==clip and r['variant']==condition)
                    assert accepted==round(gold['strict_s']/.02)
                artifact=out/f'{clip}_{condition}_{arm}.npz'
                np.savez_compressed(artifact,qpos=qs,ctrl=commands,metrics=metrics)
                results.append(dict(clip=clip,condition=condition,arm=arm,passing=accepted,valid_s=.02*accepted,
                    full=accepted==len(fitted['cmd']),limit_hits=saturated,final_metrics=metrics[-1].tolist(),
                    decision_wall_s=wall,n_decisions=len(qs),trace=h3.file_pin(artifact)))
            h3.write_json(out/'progress.json',results)
            print(clip,condition,[r['passing'] for r in results[-len(ARMS):]],flush=True)
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
