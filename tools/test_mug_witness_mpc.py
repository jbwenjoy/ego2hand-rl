"""Local two-control prospective corrections anchored to fully passing HOT3D witnesses."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from test_strict_mug_feedback import h3,plant,VARIANTS,correct,StrictCheck,LIMITS,MugStepChart,tangent_errors


def context(m,bind,pos,rot):
    chart=MugStepChart(m,bind);sig=mujoco.mjtState.mjSTATE_FULLPHYSICS
    def state(d):
        v=np.empty(mujoco.mj_stateSize(m,sig));mujoco.mj_getState(m,d,v,sig);return v
    p=SimpleNamespace(m=m,binding=bind,chart=chart,state=state,mug=m.geom_bodyid==chart.body,
        table=m.geom('mug_task_table').id,pos=pos,rot=rot)
    b=chart.body;qi=Rotation.from_quat(m.body_iquat[b,[1,2,3,0]]).as_matrix()
    p.inertia=qi@np.diag(m.body_inertia[b])@qi.T;p.mass=m.body_mass[b];p.g=m.opt.gravity
    return p,StrictCheck(p)


def momentum_load(P,s0,s1):
    _,r,v,w=P.chart.motion(s0);_,re,ve,we=P.chart.motion(s1)
    force=P.mass*((ve-v)/.02-P.g);torque=(re.apply(P.inertia@we)-r.apply(P.inertia@w))/.02
    return np.r_[r.inv().apply(force),r.inv().apply(torque)]


def controller(P,audit,scratch,live,i,fitted,prev,mode,nominal,nominal_load,rng_seed):
    ref=fitted['state'];cmds=fitted['cmd']
    s0=live;dx=tangent_errors(nominal.model,ref[i],s0[None])[0]
    if np.max(abs(dx))<1e-10:
        return cmds[i],dict(choice=0,eval_ticks=0,comparison=None)
    # The same RAW pool uses only the nominal gains, geometry, current phase and observed state.
    kinematic=correct('kinematic',nominal,s0,ref[i],dx,fitted,i)
    linear=correct('native_gain',nominal,s0,ref[i],dx,fitted,i)
    raw=[np.zeros(20),.65*prev,.5*kinematic,kinematic,.25*linear,.65*linear]
    rng=np.random.default_rng(np.random.SeedSequence([rng_seed,i]))
    noise=rng.normal(size=(3,20));noise/=np.sqrt(np.mean(noise**2,axis=1,keepdims=True))
    amplitude=min(.09,.018+.003*np.linalg.norm(dx[nominal.da:nominal.da+3])/.001)
    for n in noise:
        raw.extend([.6*prev+amplitude*n,.6*prev-amplitude*n])
    raw=np.clip(np.array(raw),-.12,.12)
    horizon=min(2,len(cmds)-i);ranks=[[],[]];ledger=[]
    for k,delta in enumerate(raw):
        mujoco.mj_resetData(P.m,scratch);mujoco.mj_setState(P.m,scratch,s0,mujoco.mjtState.mjSTATE_FULLPHYSICS)
        # Deliberately use a fresh solver estimate in every predictor.
        scratch.qacc_warmstart[:]=0.
        worst=1.;first=1.;cost=0.;support=0.
        for j in range(horizon):
            cmd=np.clip(cmds[i+j]+delta*(.8**j),*P.m.actuator_ctrlrange.T);before=P.state(scratch)
            metrics=audit.step(scratch,i+j,cmd);end=P.state(scratch)
            p,R,_,_=P.chart.motion(end);pt,Rt,_,_=nominal.motion(ref[i+j+1])
            pose=(np.linalg.norm(p-pt)/.005)**2+((R*Rt.inv()).magnitude()/np.deg2rad(8))**2
            hand=np.mean(((scratch.qpos[P.chart.qids]-ref[i+j+1,1+P.chart.qids])/.05)**2)
            cost+=pose+.10*hand+.04*np.mean(((cmd-cmds[i+j])/.08)**2)
            w=momentum_load(P,before,end)-nominal_load[i+j]
            z=w[:3]@w[:3]/nominal.weight**2+w[3:]@w[3:]/(.05*nominal.weight)**2
            support+=z/(1+z)
            ratio=max(1.,float(np.max(metrics/LIMITS)));worst=max(worst,ratio)
            if j==0:first=ratio
        for group,weight in enumerate((0.,.2)):
            ranks[group].append((first,worst,cost+weight*support,k))
        ledger.append((first,worst,cost,support))
    pose_pick=min(ranks[0])[-1];wrench_pick=min(ranks[1])[-1]
    chosen=pose_pick if mode=='pose' else wrench_pick
    return np.clip(cmds[i]+raw[chosen],*P.m.actuator_ctrlrange.T),dict(choice=chosen,eval_ticks=64*horizon*len(raw),
        comparison=[pose_pick,wrench_pick],chosen_scores=ledger[chosen],baseline_scores=ledger[0])


def main(out):
    root=Path(__file__).resolve().parents[1]/'runs';fit=h3.file_pin(root/'mug_nominal_tracking_gains_v1/report.json')
    models=h3.physics.read_json(fit);cfg=h3.physics.read_json(models['intent'])['inputs'];bind=h3.physics.read_json(cfg['binding'])
    old=h3.physics.load_arrays(cfg['reference']);asset=h3.verify_file_pin(cfg['model'])
    prior=h3.file_pin(root/'mug_tracking_feedback_v1/report.json');base_cases=h3.physics.read_json(prior)['rows']
    conditions=['nominal','mass105','mass095','mu095','mu105','start_x_plus_mm','start_x_minus_mm','heavy_low','cross_a','cross_b']
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(fit=fit,prior=prior,conditions=conditions,code=h3.file_pin(__file__),
        pool=12,horizon=2,perturbation_seed=8541,scope='All intervention arms have identical proposals at matched inputs and 12x2 native checks on a private model each active decision. Pose and wrench arms assume nominal dynamics. Oracle uses the actual model for evaluation only. Both predict from qpos/qvel, with zero planner warmstart. Goal is maintaining the archived dynamically feasible state, gated also on original six conditions. Nominal equality deadband. Report all conditions and actual command replays.'))
    rows=[]
    for clip,group in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
        fitpack=h3.physics.load_arrays(models['clips'][str(clip)]['data'])
        item,=h3.physics.read_json(h3.file_pin(root/group/'report.json'))['rows'];ref=h3.physics.load_arrays(item['reference'])
        T=ref['body'];pos=T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'];rot=Rotation.from_matrix(T[:,:3,:3])
        nominal=plant(asset,bind,1.,1.,1.);Pn,An=context(nominal,bind,pos,rot)
        nominal_load=[momentum_load(Pn,s,e) for s,e in zip(fitpack['state'][:-1],fitpack['state'][1:])]
        for condition in conditions:
            mass,mu,tau,offset=VARIANTS[condition]
            actual=plant(asset,bind,mass,mu,tau);Pa,Aa=context(actual,bind,pos,rot)
            for mode in ('pose','wrench','oracle'):
                P,A=(Pa,Aa) if mode=='oracle' else (Pn,An)
                scratch=mujoco.MjData(P.m)
                def initial():
                    d=mujoco.MjData(actual);d.qpos[:]=old['initial_qpos27'];d.qvel[:]=old['initial_qvel26'];d.ctrl[:]=old['feedforward_PD_target20'][0]
                    qa=Pa.chart.qa;d.qpos[qa:qa+3]+=Rotation.from_quat(d.qpos[qa+np.array([4,5,6,3])]).apply([offset,0,0]);mujoco.mj_forward(actual,d);return d
                d=initial();qs=[];metrics=[];commands=[];log=[];wall=0.;prev=np.zeros(20)
                for i,cmd0 in enumerate(fitpack['cmd']):
                    st=perf_counter();cmd,record=controller(P,A,scratch,Pa.state(d),i,fitpack,prev,mode,Pn.chart,nominal_load,8541)
                    wall+=perf_counter()-st;prev=cmd-cmd0
                    v=Aa.step(d,i,cmd);metrics.append(v);commands.append(cmd.copy());qs.append(d.qpos.copy());log.append(record)
                    if np.any(v>LIMITS):break
                again=initial()
                for i,cmd in enumerate(commands):
                    np.testing.assert_array_equal(metrics[i],Aa.step(again,i,cmd));np.testing.assert_array_equal(qs[i],again.qpos)
                n=len(metrics)-int(np.any(metrics[-1]>LIMITS))
                if condition=='nominal':
                    np.testing.assert_array_equal(commands,fitpack['cmd']);np.testing.assert_array_equal(qs,fitpack['state'][1:,1:1+actual.nq])
                p=out/f'{clip}_{condition}_{mode}.npz';np.savez_compressed(p,qpos=qs,ctrl=commands,metrics=metrics)
                rows.append(dict(clip=clip,condition=condition,mode=mode,passed=n,full=n==len(fitpack['cmd']),
                    open_passed=next(r['passing'] for r in base_cases if r['arm']=='open' and r['clip']==clip and r['condition']==condition),
                    trace=h3.file_pin(p),wall_s=wall,log=log))
            h3.write_json(out/'progress.json',rows);print(clip,condition,[r['passed'] for r in rows[-3:]],flush=True)
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);main(p.parse_args().out.resolve())
