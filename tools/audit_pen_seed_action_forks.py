"""One-action swaps between two previously inspected proposal seeds on matched CPU states."""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
from audit_pen_filter_action_coverage import original_raw,shooting,GATES
from evaluate_pen_dev_clip import read,file_pin,verify_file_pin,write_json
from supermimic_research.mechanics.pen_task_contracts import pose_components


def reset(m,inputs,t):
    d=mujoco.MjData(m)
    d.qpos[:],d.qvel[:],d.qacc_warmstart[:]=(inputs[k][t] for k in ('qpos','qvel','warmstart'))
    d.mocap_pos[:],d.mocap_quat[:]=(inputs[k][t] for k in ('mocap_pos','mocap_quat'))
    d.time=.02*t
    return d


def select(m,inputs,t,ref,q,sample,omega):
    d=reset(m,inputs,t)
    state=np.empty(mujoco.mj_stateSize(m,shooting.STATE))
    mujoco.mj_getState(m,d,state,shooting.STATE)
    actor=inputs['actor_raw'][t].clip(-1,1)
    raw=np.vstack([original_raw(actor,inputs['previous_raw'][t],list(range(12,16))),
                   np.clip(actor+.35*sample[t],-1,1),np.clip(actor-.35*sample[t],-1,1)]).astype(np.float32)
    ema=inputs['previous_ema'][t];targets=[]
    for phase in (t+1,t+2):
        ema=.5*ema+.25*raw
        targets.append(np.clip(q[phase]+ema-inputs['encoder_bias'][t],inputs['lower'][t],inputs['upper'][t]))
    rows=shooting.score_candidates(m,state,np.stack(targets,axis=1),ref['COM'][t+1:t+3],
        ref['quat_wxyz'][t+1:t+3],body=m.body('object/cube').id,qa=0,da=0,
        ref_v=np.diff(ref['COM'][t:t+3],axis=0)/.02,ref_omega=omega[t:t+2])
    rank=[]
    for j,r in enumerate(rows):
        if not r['finite']:continue
        ratios=np.array([[1000*i['max_object_penetration_m'],1000*i['max_self_penetration_m'],
            i['max_joint_excursion_rad'],i['max_nonhand_normal_N'],i['mm'],i['shaft_deg']] for i in r['intervals']])/GATES
        rank.append((max(1.,ratios[0].max()),max(1.,ratios.max()),
            r['pose_cost']+.05*np.mean((raw[j]-actor)**2)+.2*r['support_cost'],j))
    j=min(rank)[-1]
    return j,raw[j],rows[j]


def branch(m,inputs,t,forced,ref,q,steps):
    d=reset(m,inputs,t)
    ema=inputs['previous_ema'][t].copy()
    body=m.body('object/cube').id
    pen=m.geom_bodyid==body
    groups=np.array([next((digit for digit in range(1,6) if m.geom(i).name.startswith(f'robot/right_finger{digit}_')),0)
                     for i in range(m.ngeom)])
    result=[]
    local=np.zeros(6)
    for k in range(t,t+steps):
        raw=forced if k==t else inputs['selected_raw'][k]
        ema=.5*ema+.25*raw
        d.ctrl[:]=np.clip(q[k+1]+ema-inputs['encoder_bias'][k],inputs['lower'][k],inputs['upper'][k])
        normal=np.zeros(6);peak=0.
        for _ in range(8):
            mujoco.mj_step(m,d)
            for ic,c in enumerate(d.contact):
                g,h=c.geom
                if pen[g]==pen[h]:continue
                peak=max(peak,-1000*c.dist)
                other=h if pen[g] else g
                if c.efc_address>=0 and m.geom(int(other)).name.startswith('robot/'):
                    mujoco.mj_contactForce(m,d,ic,local)
                    normal[groups[other]]+=max(0.,local[0])/8
        ang=pose_components(d.qpos[3:7],ref['quat_wxyz'][k+1])
        result.append(dict(phase=k+1,COM_mm=1000*np.linalg.norm(d.qpos[:3]-ref['COM'][k+1]),
            shaft_deg=np.rad2deg(ang['shaft_angle_rad']),twist_deg=np.rad2deg(ang['reference_axis_twist_rad']),
            peak_mm=peak,hand_normal_N=normal.tolist(),qpos=d.qpos.tolist()))
    return result


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    src=file_pin(root/'pen_receding_seeds_v1/report.json')
    prior=read(src); base=read(read(read(prior['intent'])['source'])['intent'])
    contract=read(read(base['baseline'])['intent'])
    cfg=read(contract['configs'])['clips']['clip_003']
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(contract['anchors']['model'])))
    with np.load(verify_file_pin(cfg['item']['source'])) as f:q=f['reference_qpos'].astype(np.float32)
    with np.load(verify_file_pin(cfg['item']['series'])) as f:ref={k:f[k] for k in ('COM','quat_wxyz')}
    R=Rotation.from_quat(ref['quat_wxyz'][:,[1,2,3,0]])
    omega=(R[1:]*R[:-1].inv()).as_rotvec()/.02
    dirs={}
    for seed in (112026,112027):
        samples=qmc.Sobol(20,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(8*len(q)))))
        dirs[seed]=(2*samples[:8*len(q)]-1).reshape(-1,8,20)
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(source=src,code=file_pin(__file__),scope='On both WW archives, query both candidate pools at each identical prestate until selected RAW first differs by more than 1e-6. Swap only this action, use the source continuation RAW and each branch own EMA, up to 16 controls. Open-loop subsequent RAW. Measure endpoints even after violations. Hand normal order palm, digits 1 to 5.'))
    results=[]
    for case in prior['rows']:
        if case['mode']!='pose_support' or case['clip']!='clip_003':continue
        original=case['seed']; s=case['case']
        with np.load(verify_file_pin(s['inputs'])) as f:inputs=dict(f)
        decisions=read(s['decisions'])
        for t,d in enumerate(decisions):
            picks={seed:select(model,inputs,t,ref,q,sample,omega) for seed,sample in dirs.items()}
            assert picks[original][0]==int(d['choice'])
            np.testing.assert_array_equal(picks[original][1],inputs['selected_raw'][t])
            if np.max(abs(picks[112026][1]-picks[112027][1]))>1e-6:break
        else:raise ValueError('No differing action')
        steps=min(16,len(decisions)-t)
        forks={str(seed):dict(id=p[0],raw=p[1].tolist(),forecast_support=p[2]['support_cost'],
            forecast_pose=p[2]['pose_cost'],path=branch(model,inputs,t,p[1],ref,q,steps)) for seed,p in picks.items()}
        with np.load(verify_file_pin(read(s['native_case'])['trace'])) as f:
            actual=f['post_qpos'][8*t+7:8*(t+steps):8]
        shadow=np.array([v['qpos'] for v in forks[str(original)]['path']])
        com_gap=1000*np.linalg.norm(shadow[:,:3]-actual[:,:3],axis=1)
        ang_gap=np.rad2deg(pose_components(shadow[:,3:7],actual[:,3:7])['full_angle_rad'])
        results.append(dict(source_seed=original,phase=t+1,steps=steps,forks=forks,
            source_CPU_Warp_COM_gap_mm=com_gap.tolist(),source_CPU_Warp_angle_gap_deg=ang_gap.tolist()))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),results=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    run(p.parse_args().out.resolve())
