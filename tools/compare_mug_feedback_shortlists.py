"""Compare dense and four-candidate nonlinear feedback on matched, held-out mug perturbations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import torch
from test_mug_witness_mpc import context,h3,plant,correct,momentum_load,tangent_errors,LIMITS,VARIANTS
from qualify_mug_window import NativeStrictWindow
from supermimic_research.mechanics.mug_residual_actor import observation,MugResidualActor


class Selector:
    def __init__(self,P,checker,fitted,models,executor):
        self.P=P;self.check=checker;self.fit=fitted;self.models=models;self.executor=executor
        self.scratch=[mujoco.MjData(P.m) for _ in range(12)]
        self.targets=[P.chart.motion(s) for s in fitted['state']]
        self.wrenches=[momentum_load(P,s,e) for s,e in zip(fitted['state'][:-1],fitted['state'][1:])]

    def branch(self,slot,s,i,delta):
        P=self.P;m=P.m;d=self.scratch[slot];cmds=self.fit['cmd']
        mujoco.mj_resetData(m,d);mujoco.mj_setState(m,d,s,mujoco.mjtState.mjSTATE_FULLPHYSICS);d.qacc_warmstart[:]=0.
        first=worst=1.;cost=support=0.
        for j in range(min(2,len(cmds)-i)):
            cmd=np.clip(cmds[i+j]+delta*(.8**j),*m.actuator_ctrlrange.T);before=P.state(d)
            metrics=self.check.step(d,i+j,cmd);end=P.state(d)
            p,R,_,_=P.chart.motion(end);pt,Rt,_,_=self.targets[i+j+1]
            pose=(np.linalg.norm(p-pt)/.005)**2+((R*Rt.inv()).magnitude()/np.deg2rad(8))**2
            hand=np.mean(((d.qpos[P.chart.qids]-self.fit['state'][i+j+1,1+P.chart.qids])/.05)**2)
            cost+=pose+.1*hand+.04*np.mean(((cmd-cmds[i+j])/.08)**2)
            w=momentum_load(P,before,end)-self.wrenches[i+j]
            z=w[:3]@w[:3]/P.chart.weight**2+w[3:]@w[3:]/(.05*P.chart.weight)**2
            support+=z/(1+z)
            ratio=max(1.,float(np.max(metrics/LIMITS)));worst=max(worst,ratio)
            if j==0:first=ratio
        return first,worst,cost+.2*support,slot

    def choose(self,mode,s,i,previous):
        P=self.P;cmd0=self.fit['cmd'][i];ref=self.fit['state'][i]
        dx=tangent_errors(P.m,ref,s[None])[0]
        if np.max(abs(dx))<1e-10:return cmd0,0,0
        if mode=='students4':
            ctx,err=observation(P.m,ref,s,cmd0,previous)
            with torch.inference_mode():
                actions=[self.models[a](torch.from_numpy(ctx),torch.from_numpy(err)).numpy() for a in ('pose','wrench')]
            raw=[np.zeros(20),.65*previous,*actions]
        else:
            kin=correct('kinematic',P.chart,s,ref,dx,self.fit,i)
            lin=correct('native_gain',P.chart,s,ref,dx,self.fit,i)
            raw=[np.zeros(20),.65*previous,kin,.25*lin]
            if mode=='dense12':
                raw=[np.zeros(20),.65*previous,.5*kin,kin,.25*lin,.65*lin]
                rng=np.random.default_rng(np.random.SeedSequence([8541,i]))
                noise=rng.normal(size=(3,20));noise/=np.sqrt(np.mean(noise**2,axis=1,keepdims=True))
                amp=min(.09,.018+.003*np.linalg.norm(dx[P.chart.da:P.chart.da+3])/.001)
                for direction in noise:raw.extend([.6*previous+amp*direction,.6*previous-amp*direction])
        deltas=np.clip(np.array(raw),-.12,.12)
        futures=[self.executor.submit(self.branch,k,s,i,v) for k,v in enumerate(deltas)]
        selected=min(f.result() for f in futures)[-1]
        return np.clip(cmd0+deltas[selected],*P.m.actuator_ctrlrange.T),selected,64*min(2,len(self.fit['cmd'])-i)*len(deltas)


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    student_pin=h3.file_pin(root/'mug_feedback_distill_v1/report.json');sr=h3.physics.read_json(student_pin)
    weights=h3.physics.read_json(sr['frozen'])['weights'];nets={}
    torch.set_num_threads(1)
    for a,pin in weights.items():
        ck=torch.load(h3.verify_file_pin(pin),weights_only=True,map_location='cpu')
        net=MugResidualActor(*ck['dims']);net.load_state_dict(ck['state']);nets[a]=net.eval()
    prior=h3.file_pin(root/'mug_witness_mpc_v1/report.json');raw=h3.physics.read_json(prior)
    gains=h3.physics.read_json(h3.physics.read_json(raw['intent'])['fit'])
    inputs=h3.physics.read_json(gains['intent'])['inputs'];old=h3.physics.load_arrays(inputs['reference'])
    bind=h3.physics.read_json(inputs['binding']);asset=h3.verify_file_pin(inputs['model'])
    qualification=h3.file_pin(root/'mug_window_native_check_v1/report.json')
    build=h3.physics.read_json(h3.physics.read_json(qualification)['intent'])['build']
    lib=h3.verify_file_pin(h3.physics.read_json(build)['helper'])
    # Reuse the student-only test's frozen new combinations; prove that the dense path
    # is unchanged on preexisting calibration cases before interpreting faster timings.
    test=h3.file_pin(root/'mug_distilled_feedback_test_v1/report.json');student_test=h3.physics.read_json(test)
    conds=h3.physics.read_json(student_test['intent'])['conditions']
    selected={'nominal':conds['nominal'],'mass105':conds['mass105'],'start_x_minus_mm':conds['start_x_minus_mm'],
              **{k:v for k,v in conds.items() if k.startswith('generated_')}}
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(students=student_pin,prior=prior,qualification=qualification,matched_conditions=test,
        conditions=selected,code=h3.file_pin(__file__),scope='After seeing direct student predictions, compare four candidate analytic and student pools against dense12 and open on the same 12 held-out combinations. All prospective choices predict from live qpos/qvel in the nominal model, exact strict geometry, pose+wrench ranking; 4 native CPU workers and qualified compiled measurement loop. Two prior calibration perturbations must reproduce archived dense controls and states.'))
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for clip,group in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
            fit=h3.physics.load_arrays(gains['clips'][str(clip)]['data'])
            row,=h3.physics.read_json(h3.file_pin(root/group/'report.json'))['rows']
            ref=h3.physics.load_arrays(row['reference']);T=ref['body']
            pos=T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'];rot=Rotation.from_matrix(T[:,:3,:3])
            nominal=plant(asset,bind,1.,1.,1.);Pn,_=context(nominal,bind,pos,rot)
            policy=Selector(Pn,NativeStrictWindow(Pn,lib),fit,nets,executor)
            for condition,(mass,mu,tau,offset) in selected.items():
                m=plant(asset,bind,mass,mu,tau);P,python=context(m,bind,pos,rot);fast=NativeStrictWindow(P,lib)
                def initial():
                    d=mujoco.MjData(m);d.qpos[:]=old['initial_qpos27'];d.qvel[:]=old['initial_qvel26'];d.ctrl[:]=old['feedforward_PD_target20'][0]
                    qa=P.chart.qa;d.qpos[qa:qa+3]+=Rotation.from_quat(d.qpos[qa+np.array([4,5,6,3])]).apply([offset,0,0]);mujoco.mj_forward(m,d);return d
                for mode in ('dense12','analytic4','students4'):
                    d=initial();qs=[];metrics=[];cs=[];choices=[];ticks=0;wall=0.;prev=np.zeros(20)
                    for i,cmd0 in enumerate(fit['cmd']):
                        start=perf_counter();cmd,index,work=policy.choose(mode,P.state(d),i,prev)
                        wall+=perf_counter()-start;ticks+=work;prev=cmd-cmd0
                        v=fast.step(d,i,cmd);qs.append(d.qpos.copy());cs.append(cmd.copy());metrics.append(v);choices.append(index)
                        if np.any(v>LIMITS):break
                    passed=len(metrics)-int(np.any(metrics[-1]>LIMITS))
                    restored=initial()
                    for i,c in enumerate(cs):
                        np.testing.assert_array_equal(python.step(restored,i,c),metrics[i]);np.testing.assert_array_equal(restored.qpos,qs[i])
                    if not condition.startswith('generated') and mode=='dense12':
                        prevcase=next(r for r in raw['rows'] if r['clip']==clip and r['condition']==condition and r['mode']=='wrench')
                        trace=h3.physics.load_arrays(prevcase['trace'])
                        np.testing.assert_array_equal(cs,trace['ctrl']);np.testing.assert_array_equal(qs,trace['qpos'])
                    if condition=='nominal':np.testing.assert_array_equal(qs,fit['state'][1:,1:1+m.nq])
                    dest=out/f'{clip}_{condition}_{mode}.npz';np.savez_compressed(dest,qpos=qs,ctrl=cs,metrics=metrics,choices=choices)
                    rows.append(dict(clip=clip,condition=condition,mode=mode,passing=passed,full=passed==len(fit['cmd']),
                        controls=len(cs),model_ticks=ticks,wall_s=wall,trace=h3.file_pin(dest)))
                h3.write_json(out/'progress.json',rows);print(clip,condition,[r['passing'] for r in rows[-3:]],flush=True)
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
