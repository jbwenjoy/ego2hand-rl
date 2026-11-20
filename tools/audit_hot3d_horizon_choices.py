"""Re-evaluate the same shortlists with native continuations on the live H3 routes."""
import argparse
from pathlib import Path
import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from test_hot3d_model_horizon import (physics,file_pin,verify_file_pin,write_json,kinematics,propose,mean_reference_wrench,
    features,continuations,decode,MugHorizonHead,MugStepChart,order_horizon)


def run(parent,out):
    root=Path(__file__).resolve().parents[1]/'runs';source=file_pin(parent/'report.json')
    trials=physics.read_json(source);spec=physics.read_json(trials['intent'])
    model=physics.read_json(spec['fit']);cfg=physics.read_json(model['intent']);inputs=cfg['inputs']
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['model'])))
    chart=MugStepChart(m,physics.read_json(inputs['binding']));body=chart.body;old=physics.load_arrays(inputs['reference'])
    Q=Rotation.from_quat(m.body_iquat[body,[1,2,3,0]]).as_matrix();I=Q@np.diag(m.body_inertia[body])@Q.T
    ck=torch.load(verify_file_pin(model['model']),map_location='cpu',weights_only=True)
    torch.set_num_threads(1);net=MugHorizonHead(ck['n']);net.load_state_dict(ck['state']);net.eval()
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(source=source,code=file_pin(__file__),
        scope='Audit all executed H3 controller states including failures. Physically evaluate the exact same 8 fixed three-command continuations; actual future closed-loop replans are not fed into the audit. Future eligibility concerns COM/rotation gates only. Statistics are conditional on the visited H3 states.'))
    sig=mujoco.mjtState.mjSTATE_FULLPHYSICS;workers=[mujoco.MjData(m) for _ in range(4)]
    rows=[]
    with Rollout(nthread=4) as engine:
        for name in ('mug_annotated_windows_original_v1','mug_unilateral_interval_v1'):
            item,=physics.read_json(file_pin(root/name/'report.json'))['rows']
            ref=physics.load_arrays(item['reference']);T=ref['body'];base=old['feedforward_PD_target20'][0]
            ik=np.clip(ref['q'][1:]+base-old['initial_qpos27'][:20],*m.actuator_ctrlrange.T)
            pos=T[:,:3,3]+T[:,:3,:3]@chart.com;rot=Rotation.from_matrix(T[:,:3,:3])
            loads=mean_reference_wrench(pos,rot.as_quat()[:,[3,0,1,2]],m.body_mass[body],I,m.opt.gravity)
            for run in trials['rows']:
                if run['mode']!='h3' or run['clip']!=item['clip']:continue
                arc=physics.load_arrays(run['trace'])
                d=mujoco.MjData(m);d.qpos[:]=old['initial_qpos27'];d.qvel[:]=old['initial_qvel26'];d.ctrl[:]=base
                mujoco.mj_forward(m,d)
                previous=base;res=np.zeros(20);seed=run['seed']
                for i,log in enumerate(run['records']):
                    pre=np.empty(mujoco.mj_stateSize(m,sig));mujoco.mj_getState(m,d,pre,sig)
                    all_raw=propose(np.random.default_rng(np.random.SeedSequence([seed,i,0])),64,ik[i],base,previous,res,m.actuator_ctrlrange)
                    ids=np.array(log['shortlist']);chosen=int(np.flatnonzero(ids==log['choice'])[0]);cmd=all_raw[ids]
                    seq=continuations(cmd,ik,i,m.actuator_ctrlrange);horizon=min(3,len(ik)-i)
                    _,_,v0,L0=kinematics(pre[None],chart.qa,chart.da,chart.com,I,m.nq)
                    with torch.inference_mode():pred=net(torch.from_numpy(features(chart,pre,d.qacc_warmstart,seq))).numpy().reshape(-1,3,12)
                    st,_=engine.rollout(m,workers,np.tile(pre,(8,1)),control=np.repeat(seq[:,:horizon],64,axis=1),initial_warmstart=np.tile(d.qacc_warmstart,(8,1)))
                    actual=[];forecast=[]
                    for h in range(1,horizon+1):
                        p,R,v,L=kinematics(st[:,64*h-1],chart.qa,chart.da,chart.com,I,m.nq)
                        netw=np.column_stack([m.body_mass[body]*((v-v0)/(.02*h)-m.opt.gravity),(L-L0)/(.02*h)])
                        for dest,(ep,eR,ew) in [(actual,(p,R,netw)),(forecast,decode(chart,pre,pred[:,h-1],h))]:
                            mm=1000*np.linalg.norm(ep-pos[i+h],axis=1);deg=np.rad2deg((eR*rot[i+h].inv()).magnitude())
                            cost=(mm/10)**2+(deg/10)**2+.01*np.mean(((seq[:,h-1]-ik[i+h-1])/.12)**2,axis=1)
                            if loads['valid_interval'][i:i+h].all():
                                delta=ew-loads['wrench_world'][i:i+h].mean(0)
                                sq=np.sum(delta[:,:3]**2,axis=1)/chart.weight**2+np.sum(delta[:,3:]**2,axis=1)/(.05*chart.weight)**2
                                cost+=.2*sq/(1+sq)
                            dest.append((np.maximum(1.,np.maximum(mm,deg)/10),cost))
                    g,c=np.array(actual).transpose(1,0,2);pg,pc=np.array(forecast).transpose(1,0,2)
                    hybrid_g=pg.copy();hybrid_g[0]=g[0];hybrid_c=pc.copy();hybrid_c[0]=c[0]
                    recovered=order_horizon(g[0],hybrid_g.max(0),hybrid_c.sum(0))[0]
                    if recovered!=chosen and not np.array_equal(cmd[recovered],cmd[chosen]):
                        raise ValueError('Could not reproduce learned choice on archived source')
                    greedy=int(order_horizon(g[0],g[0],c[0])[0]);oracle=int(order_horizon(g[0],g.max(0),c.sum(0))[0])
                    eligible=(g<=1).all(0);predicted=(hybrid_g<=1).all(0)
                    rows.append(dict(clip=item['clip'],seed=seed,phase=i+1,horizon=horizon,any_feasible=bool(eligible.any()),
                        chosen_feasible=bool(eligible[chosen]),greedy_feasible=bool(eligible[greedy]),predicted_safe=bool(predicted[chosen]),
                        oracle_match=bool(np.array_equal(cmd[oracle],cmd[chosen])),eligible_count=int(eligible.sum()),
                        false_safe=int((predicted&~eligible).sum()),false_unsafe=int((~predicted&eligible).sum()),
                        predicted_count=int(predicted.sum()),choices=len(cmd),native_ticks=8*64*horizon))
                    np.testing.assert_array_equal(cmd[chosen],arc['ctrl'][i])
                    d.ctrl[:]=cmd[chosen];mujoco.mj_step(m,d,nstep=64);np.testing.assert_array_equal(d.qpos,arc['qpos'][i])
                    previous=cmd[chosen];res=previous-ik[i]
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();run(args.parent.resolve(),args.out.resolve())
