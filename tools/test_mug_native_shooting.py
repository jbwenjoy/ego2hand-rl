"""Matched native shooting diagnostic on the 2 s annotated mug reference."""
import json,sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from evaluate_mug_live_wrench import ROOT,verify_file_pin,file_pin,write_json,mean_reference_wrench


def run(out):
    out.mkdir(exist_ok=False)
    root=ROOT/'runs';pin=file_pin(root/'mug_temporal_feedforward_v1/report.json')
    read=lambda p:json.loads(verify_file_pin(p).read_bytes())
    ff=read(pin);traj=read(read(ff['intent'])['parent']);ref=read(read(traj['dynamic'])['intent'])
    binding=read(ref['binding']);ref=read(ref['reference'])
    with np.load(verify_file_pin(ref['output'])) as f:z=dict(f)
    with np.load(verify_file_pin(ff['results']['interval_lp']['trace'])) as f:ctrl=f['ctrl'];gold=f['qpos']
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(ref['model'])));b=binding['mug_body_id'];qa=binding['mug_free_qpos']
    hand={m.geom(n).id for n in binding['hand_collision_names']};sub=round(.02/m.opt.timestep)
    assert m.nu==20 and sub==64 and not m.neq and not np.any(m.body_gravcomp)
    R=Rotation.from_quat(m.body_iquat[b,[1,2,3,0]]).as_matrix();I=R@np.diag(m.body_inertia[b])@R.T
    target=mean_reference_wrench(z['COM_world_m'],z['quaternion_wxyz'],m.body_mass[b],I,m.opt.gravity)
    rref=Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]])
    seeds=[7411,7412,7413]
    write_json(out/'intent.json',dict(parent=pin,seeds=seeds,horizon_controls=3,candidates='zero, incumbent, and 4 antithetic Gaussian 20-joint directions added to incumbent; component RMS .025 rad, cumulative component bound .12 rad. Same perturbations per seed/phase.',score='Mean normalized squared position (1cm), angle (10deg), plus .01*normalized squared residual and optional .2*bounded support cost. Support cost is mean source BODY wrench squared error normalized by mg and mg*0.05m, divided by 1+error. Measured contact excludes table.',stop='First control endpoint exceeding 1cm or 10deg or table normal >0.1N in executed window. Max 100 controls. Only executed first action evaluated. Perfect simulator state and dynamics, no learned dynamics.'))
    def pose(d):
        r=Rotation.from_quat(d.qpos[qa+np.array([4,5,6,3])]);return d.qpos[qa:qa+3]+r.apply(z['COM_BODY_m']),r
    def interval(d,i,delta):
        cmd=np.clip(ctrl[i+1]+delta,*m.actuator_ctrlrange.T);d.ctrl[:]=cmd
        assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
        load=np.zeros(6);table=False;r0=pose(d)[1];hand_ticks=0
        for _ in range(sub):
            mujoco.mj_step(m,d);normal=0.
            for ic,c in enumerate(d.contact):
                g,h=map(int,c.geom);on=m.geom_bodyid[[g,h]]==b
                if on.sum()!=1:continue
                other=h if on[0] else g;f=np.zeros(6);mujoco.mj_contactForce(m,d,ic,f)
                if other not in hand:
                    table|=bool(other==table_id and f[0]>.1);continue
                s=1 if on[1] else -1;frame=c.frame.reshape(3,3)
                force=s*frame.T@f[:3];normal+=max(0.,f[0])
                load+=np.r_[force,s*frame.T@f[3:]+np.cross(c.pos-d.xipos[b],force)]
            hand_ticks+=int(normal>.1)
        load/=sub
        mujoco.mj_kinematics(m,d);p,r=pose(d)
        mm=1000*np.linalg.norm(p-z['COM_world_m'][i+1]);deg=np.rad2deg((r*rref[i+1].inv()).magnitude())
        weight=m.body_mass[b]*np.linalg.norm(m.opt.gravity);err=0.
        if target['valid_interval'][i]:
            diff=load-target['wrench_world'][i]
            err=np.sum((r0.inv().apply(diff[:3])/weight)**2)+np.sum((r0.inv().apply(diff[3:])/(weight*.05))**2)
        return dict(mm=float(mm),deg=float(deg),table=bool(table),support_cost=float(err/(1+err)),load=load.tolist(),hand_ticks=hand_ticks)
    table_id=m.geom('mug_task_table').id
    rows=[]
    for seed in seeds:
        for mode in (['replay','tracking','support'] if seed==seeds[0] else ['tracking','support']):
            rng=np.random.default_rng(seed);d=mujoco.MjData(m);clone=mujoco.MjData(m)
            d.qpos[:]=z['initial_qpos27'];d.qvel[:]=z['initial_qvel26'];d.ctrl[:]=ctrl[0];mujoco.mj_forward(m,d)
            incumbent=np.zeros(20);actual=[];planning=0;qs=[];commands=[]
            for i in range(100):
                directions=rng.normal(size=(4,20));directions*=.025/np.sqrt(np.mean(directions**2,axis=1,keepdims=True))
                grid=np.clip(np.vstack([np.zeros(20),incumbent,incumbent+directions,incumbent-directions]),-.12,.12)
                if mode=='replay':idx=0;costs=[]
                else:
                    costs=[];first=[]
                    for u in grid:
                        mujoco.mj_copyData(clone,m,d);score=0.
                        for j in range(i,min(100,i+3)):
                            a=interval(clone,j,u);planning+=sub
                            if j==i:first.append(clone.qpos.copy())
                            score+=(a['mm']/10)**2+(a['deg']/10)**2+100*a['table']+.01*np.mean((u/.12)**2)+(.2*a['support_cost'] if mode=='support' else 0)
                        costs.append(float(score))
                    idx=int(np.argmin(costs))
                incumbent=grid[idx].copy();a=interval(d,i,incumbent)
                np.testing.assert_array_equal(d.qpos,gold[i] if mode=='replay' else first[idx])
                qs.append(d.qpos.copy());commands.append(d.ctrl.copy());actual.append(dict(phase=i+1,choice=idx,costs=costs,**a))
                if a['mm']>10 or a['deg']>10 or a['table']:break
            f=out/f'{mode}_{seed}.npz';np.savez_compressed(f,qpos=qs,ctrl=commands)
            valid=sum(v['mm']<=10 and v['deg']<=10 and not v['table'] for v in actual)
            rows.append(dict(mode=mode,seed=seed,valid_s=.02*valid,full=valid==100,actual=actual,trace=file_pin(f),planning_native=planning,executed_native=len(actual)*sub))
            write_json(out/'progress.json',dict(completed=len(rows),rows=rows))
            print(seed,mode,valid*.02,len(actual),flush=True)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
