"""Check oracle-action coverage of learned and random shortlists on identical native states."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from test_mug_learned_planning import physics,file_pin,verify_file_pin,write_json,MugStepChart,MugStepHead,propose,rank,mean_reference_wrench


def run(parent,out):
    source=file_pin(parent/'report.json');suite=physics.read_json(source)
    root=Path(__file__).resolve().parents[1]/'runs'
    fit=physics.read_json(file_pin(root/'mug_shared_step_v1/report.json'));inputs=physics.read_json(fit['intent'])['inputs']
    pin=physics.read_json(fit['frozen'])['pins']['shared']
    ck=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=True)
    net=MugStepHead(ck['n']);net.load_state_dict(ck['state']);net.eval();torch.set_num_threads(1)
    old=physics.load_arrays(inputs['reference']);model=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['model'])))
    chart=MugStepChart(model,physics.read_json(inputs['binding']));b=chart.body
    R=Rotation.from_quat(model.body_iquat[b,[1,2,3,0]]).as_matrix();I=R@np.diag(model.body_inertia[b])@R.T
    out.mkdir(exist_ok=False)
    rows=[]
    for name in ('mug_annotated_windows_original_v1','mug_unilateral_interval_v1'):
        item,=physics.read_json(file_pin(root/name/'report.json'))['rows']
        ref=physics.load_arrays(item['reference']);T=ref['body']
        pos=T[:,:3,3]+T[:,:3,:3]@chart.com;rot=Rotation.from_matrix(T[:,:3,:3])
        target=mean_reference_wrench(pos,rot.as_quat()[:,[3,0,1,2]],model.body_mass[b],I,model.opt.gravity)
        base=old['feedforward_PD_target20'][0];direct=np.clip(ref['q'][1:]+base-old['initial_qpos27'][:20],*model.actuator_ctrlrange.T)
        for row in suite['rows']:
            if row['clip']!=item['clip'] or row['mode']!='native64':continue
            tr=physics.load_arrays(row['trace'])
            d=mujoco.MjData(model);d.qpos[:]=old['initial_qpos27'];d.qvel[:]=old['initial_qvel26'];d.ctrl[:]=base
            mujoco.mj_forward(model,d);last=base;res=np.zeros(20);seed=row['seed']
            sig=mujoco.mjtState.mjSTATE_FULLPHYSICS
            for i,log in enumerate(row['log']):
                state=np.empty(mujoco.mj_stateSize(model,sig));mujoco.mj_getState(model,d,state,sig)
                cmds=propose(np.random.default_rng(np.random.SeedSequence([seed,i,0])),64,direct[i],base,last,res,model.actuator_ctrlrange)
                gold=tr['ctrl'][i];np.testing.assert_array_equal(gold,cmds[log['choice']])
                # Identical commands can occur in multiple slots, especially at reset.
                matching=(cmds==gold).all(1)
                with torch.no_grad():y=net(torch.from_numpy(chart.features(state,d.qacc_warmstart,cmds))).numpy()
                preds=chart.decode(state,y);_,r,_,_=chart.motion(state)
                order=rank(*preds,pos[i+1],rot[i+1],target['wrench_world'][i],target['valid_interval'][i],
                    chart.weight,r,cmds,direct[i])
                rng=np.random.default_rng(np.random.SeedSequence([seed,i,810]))
                learned=np.r_[np.arange(4),order[order>=4][:4]];random=np.r_[np.arange(4),rng.choice(np.arange(4,64),4,replace=False)]
                rows.append(dict(clip=row['clip'],seed=seed,phase=i+1,anchor=bool(matching[:4].any()),
                    learned=bool(matching[learned].any()),random=bool(matching[random].any())))
                d.ctrl[:]=gold;mujoco.mj_step(model,d,nstep=64);np.testing.assert_array_equal(d.qpos,tr['qpos'][i])
                last=gold;res=gold-direct[i]
    write_json(out/'report.json',dict(source=source,weights=pin,code=file_pin(__file__),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();run(a.parent.resolve(),a.out.resolve())
