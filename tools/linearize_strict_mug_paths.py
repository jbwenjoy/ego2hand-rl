"""Freeze feedback gains from the nominal plant and a previously trained shared response head."""
import argparse
from pathlib import Path
from time import perf_counter
import mujoco
from mujoco.rollout import Rollout
import numpy as np
import torch
from collect_mug_handoff_labels import h3
from supermimic_research.mechanics.mug_local_feedback import gains,local_dynamics,learned_object_rows,tangent_errors
from supermimic_research.mechanics.mug_step_response import MugStepChart,MugStepHead


def main(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    fit=h3.file_pin(root/'mug_shared_step_v1/report.json')
    record=h3.physics.read_json(fit)
    inputs=h3.physics.read_json(record['intent'])['inputs']
    model_pin=h3.physics.read_json(record['frozen'])['pins']['shared']
    ck=torch.load(h3.verify_file_pin(model_pin),weights_only=True,map_location='cpu')
    torch.set_num_threads(1);net=MugStepHead(ck['n']);net.load_state_dict(ck['state']);net.eval()
    m=mujoco.MjModel.from_binary_path(str(h3.verify_file_pin(inputs['model'])))
    binding=h3.physics.read_json(inputs['binding']);old=h3.physics.load_arrays(inputs['reference'])
    chart=MugStepChart(m,binding)
    paths={str(c):h3.file_pin(root/f'mug_strict_extend_{c}_v1/report.json') for c in (1884,1888)}
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(paths=paths,fit=fit,shared_head=model_pin,inputs=inputs,
        code=[h3.file_pin(__file__),h3.file_pin(Path(__file__).resolve().parents[1]/'src/supermimic_research/mechanics/mug_local_feedback.py')],
        scope='Generate gains before any perturbation runs. Nominal physical parameters only. Tangent A52 and B20 around each unassisted witness, actual 64 native ticks, measured nominal warmstart held fixed. Hybrid shares native A and hand rows of B, replaces only twelve mug pose/velocity rows of B by frozen shared head response. Nonlinear probe check uses fixed held-out random command directions.'))
    results={}
    workers=[mujoco.MjData(m) for _ in range(4)]
    sig=mujoco.mjtState.mjSTATE_FULLPHYSICS
    def state(d):
        a=np.empty(mujoco.mj_stateSize(m,sig));mujoco.mj_getState(m,d,a,sig);return a
    with Rollout(nthread=4) as engine:
        for c,pin in paths.items():
            before=perf_counter()
            r=h3.physics.read_json(pin);arc=h3.physics.load_arrays(r['trace'])
            assert r['full_strict']
            d=mujoco.MjData(m);d.qpos[:]=old['initial_qpos27'];d.qvel[:]=old['initial_qvel26'];d.ctrl[:]=old['feedforward_PD_target20'][0]
            mujoco.mj_forward(m,d);states=[];aa=[];bb=[];mixed=[];ticks=0;probe=[]
            for i,u in enumerate(arc['ctrl']):
                src=state(d);warm=d.qacc_warmstart.copy()
                d.ctrl[:]=u;mujoco.mj_step(m,d,nstep=64);end=state(d)
                np.testing.assert_array_equal(d.qpos,arc['qpos'][i]);states.append(src)
                a,b,plus,minus,n=local_dynamics(m,workers,engine,src,warm,u,end,chart.qa,chart.da)
                hybrid=learned_object_rows(chart,net,src,warm,plus,minus,end,b)
                aa.append(a);bb.append(b);mixed.append(hybrid);ticks+=n
                axis=np.random.default_rng(np.random.SeedSequence([8520,i])).normal(size=(4,m.nu))
                axis*=.01/np.linalg.norm(axis,axis=1)[:,None]
                cmds=np.clip(u+axis,*m.actuator_ctrlrange.T)
                test,_=engine.rollout(m,workers,np.tile(src,(4,1)),control=np.repeat(cmds[:,None],64,axis=1),
                                      initial_warmstart=np.tile(warm,(4,1)))
                truth=tangent_errors(m,end,test[:,-1]);pert=cmds-u
                probe.append(np.stack([truth,pert@b.T,pert@hybrid.T]));ticks+=256
            states.append(end);A=np.array(aa);B=np.array(bb);Bh=np.array(mixed)
            f=out/f'{c}_feedback.npz'
            np.savez_compressed(f,state=states,A=A,B_native=B,B_shared=Bh,
                K_native=gains(A,B,m,chart.da),K_shared=gains(A,Bh,m,chart.da),cmd=arc['ctrl'],probe=probe)
            pr=np.array(probe)
            diagnostics={}
            for k,ix in [('translation',slice(chart.da,chart.da+3)),('rotation',slice(chart.da+3,chart.da+6)),
                         ('linear_velocity',slice(m.nv+chart.da,m.nv+chart.da+3)),('angular_velocity',slice(m.nv+chart.da+3,m.nv+chart.da+6))]:
                obs=pr[:,0,:,ix]
                diagnostics[k]=[float(np.linalg.norm(pr[:,j,:,ix]-obs)/np.linalg.norm(obs)) for j in (1,2)]
            results[c]=dict(data=h3.file_pin(f),ticks=ticks,wall_s=perf_counter()-before,control_count=len(B),
                            native_shared_probe_error_ratios=diagnostics)
            print(c,results[c]['wall_s'],flush=True)
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),clips=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);main(p.parse_args().out.resolve())
