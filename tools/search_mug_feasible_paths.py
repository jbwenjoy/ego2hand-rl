"""Compare equal-size action budgets allocated to one or four surviving mug paths."""
import argparse
from pathlib import Path
from time import perf_counter
import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.spatial.transform import Rotation
from compare_hot3d_mug_wrench import physics,file_pin,verify_file_pin,write_json,mean_reference_wrench,MugContactImpulse


def propose(rng,k,direct,base,previous,residual,limits):
    # Directions are paired and scaled in joint radians. Keep a small set of
    # exact incumbent/IK/hold options even when a parent has few allocations.
    pool=[direct,base,previous,direct+residual]
    for i in range((k-3)//2):
        axis=rng.standard_normal(20)
        axis/=np.sqrt(np.mean(axis**2))
        offset=min(.24,.04*(1+i%4))*axis
        center=direct+.8*residual
        pool.extend([center+offset,center-offset])
    return np.clip(np.asarray(pool[:k]),*limits.T)


def kinematics(states,qa,da,com,inertia,nq):
    q=states[:,1:1+nq];v=states[:,1+nq:]
    rot=Rotation.from_quat(q[:,qa+np.array([4,5,6,3])])
    arm=rot.apply(np.tile(com,(len(q),1)))
    omega=rot.apply(v[:,da+3:da+6])
    return (q[:,qa:qa+3]+arm,rot,v[:,da:da+3]+np.cross(omega,arm),
            rot.apply(v[:,da+3:da+6]@inertia.T))


def search(model,model_path,binding,z,direct,base,width,seed,out):
    sig=mujoco.mjtState.mjSTATE_FULLPHYSICS
    sensor=MugContactImpulse(model,binding);body=sensor.body
    qa=binding['mug_free_qpos'];da=sensor.dof
    assert not any((model.na,model.nmocap,model.nplugin,model.neq,model.nsensor,model.nuserdata))
    assert model.opt.timestep==.02/64 and not model.body_gravcomp.any()
    inertia_rot=Rotation.from_quat(model.body_iquat[body,[1,2,3,0]]).as_matrix()
    I=inertia_rot@np.diag(model.body_inertia[body])@inertia_rot.T
    m=model.body_mass[body];weight=m*np.linalg.norm(model.opt.gravity)
    target=mean_reference_wrench(z['COM_world_m'],z['quaternion_wxyz'],m,I,model.opt.gravity)
    refrot=Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]])
    actual=mujoco.MjData(model)
    actual.qpos[:]=z['initial_qpos27'];actual.qvel[:]=z['initial_qvel26'];actual.ctrl[:]=base
    mujoco.mj_forward(model,actual)
    def state(d):
        arr=np.empty(mujoco.mj_stateSize(model,sig));mujoco.mj_getState(model,d,arr,sig);return arr
    initial=state(actual)
    parents=[dict(state=initial,warm=actual.qacc_warmstart.copy(),cmd=base,delta=np.zeros(20),
                  path=[],cost=0.,states=[])]
    nthread=4
    workers=[mujoco.MjData(model) for _ in range(nthread)]
    records=[];checks=0;ledger=[];winning=None
    with Rollout(nthread=nthread) as pool:
        for phase,nominal in enumerate(direct):
            counts=np.full(len(parents),64//len(parents));counts[:64%len(parents)]+=1
            proposals=[propose(np.random.default_rng(np.random.SeedSequence([seed,phase,i])),int(k),nominal,base,p['cmd'],p['delta'],model.actuator_ctrlrange)
                       for i,(p,k) in enumerate(zip(parents,counts))]
            cmds=np.vstack(proposals);assert len(cmds)==64
            index=np.repeat(np.arange(len(parents)),counts)
            pre=np.array([parents[i]['state'] for i in index])
            warm=np.array([parents[i]['warm'] for i in index])
            states,_=pool.rollout(model,workers,pre,control=np.repeat(cmds[:,None],64,axis=1),initial_warmstart=warm)
            end=states[:,-1,:]
            assert np.isfinite(end).all() and np.allclose(end[:,0]-pre[:,0],.02,rtol=0,atol=2e-12)
            co,rot,v1,L1=kinematics(end,qa,da,z['COM_BODY_m'],I,model.nq)
            _,r0,v0,L0=kinematics(pre,qa,da,z['COM_BODY_m'],I,model.nq)
            net=np.column_stack([m*((v1-v0)/.02-model.opt.gravity),(L1-L0)/.02])
            mm=1000*np.linalg.norm(co-z['COM_world_m'][phase+1],axis=1)
            deg=np.rad2deg((rot*refrot[phase+1].inv()).magnitude())
            wcost=np.zeros(len(cmds))
            if target['valid_interval'][phase]:
                diff=net-target['wrench_world'][phase]
                e=np.sum((r0.inv().apply(diff[:,:3])/weight)**2+(r0.inv().apply(diff[:,3:])/(weight*.05))**2,axis=1)
                wcost=e/(1+e)
            per_step=(mm/10)**2+(deg/10)**2+.2*wcost+.01*np.mean(((cmds-nominal)/.12)**2,axis=1)
            cost=np.array([parents[i]['cost'] for i in index])+per_step
            in_bounds=(mm<=10)&(deg<=10)
            ledger.append(dict(phase=phase+1,parents=len(parents),object_gate_passing=int(in_bounds.sum())))
            records.append((pre,warm,cmds,end.copy(),net,mm,deg,index))
            survivors=[];used=set();naccepted=np.zeros(len(parents),int);verified={}
            for cap in (max(1,width//2) if len(parents)>1 else width,width):
                for j in np.argsort(cost,kind='stable'):
                    i=index[j];signature=(int(i),cmds[j].tobytes())
                    if not in_bounds[j] or signature in used or naccepted[i]>=cap:continue
                    if signature not in verified:
                        mujoco.mj_resetData(model,actual);mujoco.mj_setState(model,actual,pre[j],sig)
                        actual.qacc_warmstart[:]=warm[j];actual.ctrl[:]=cmds[j]
                        assert not actual.xfrc_applied.any() and not actual.qfrc_applied.any()
                        peak=0.
                        for _ in range(64):
                            mujoco.mj_step(model,actual)
                            a,b=actual.contact.geom.T
                            for ic in np.flatnonzero(((a==sensor.table)&sensor.is_mug[b])|((b==sensor.table)&sensor.is_mug[a])):
                                mujoco.mj_contactForce(model,actual,ic,sensor.local);peak=max(peak,sensor.local[0])
                        checks+=64;np.testing.assert_array_equal(state(actual),end[j])
                        verified[signature]=(peak,actual.qacc_warmstart.copy())
                    peak,newwarm=verified[signature]
                    if peak>.1:continue
                    used.add(signature);naccepted[i]+=1;p=parents[i]
                    survivors.append(dict(state=end[j].copy(),warm=newwarm,cmd=cmds[j],delta=cmds[j]-nominal,
                        path=[*p['path'],cmds[j].copy()],states=[*p['states'],end[j].copy()],cost=cost[j]))
                    if len(survivors)==width:break
                if len(survivors)==width:break
            ledger[-1]['retained']=len(survivors)
            if not survivors:break
            parents=survivors;winning=min(parents,key=lambda p:p['cost'])
    if winning is None:raise ValueError('No valid first action')
    cmds=np.array(winning['path'])
    detail,trace=physics.replay(model_path,binding,z,base,cmds,1.,1.)
    np.testing.assert_array_equal(trace['qpos'],np.array(winning['states'])[:,1:1+model.nq])
    assert detail['right_censored'] and np.isclose(detail['valid_s'],len(cmds)*.02)
    p=out/'selected.npz';np.savez_compressed(p,**trace)
    bank=out/'candidates.npz'
    keys=('source','warmstart','cmd','end','net_wrench','mm','deg','parent')
    np.savez_compressed(bank,**{k:np.stack(v) for k,v in zip(keys,zip(*records))})
    return dict(valid_s=detail['valid_s'],full=len(cmds)==len(direct),controls=len(cmds),requested=len(direct),
        path=file_pin(p),bank=file_pin(bank),ledger=ledger,search_ticks=64*64*len(records),verification_ticks=checks,
        final_replay=detail)


def main(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    prior=file_pin(root/'mug_ego_wrench_comparison_v1/intent.json')
    cfg=physics.read_json(prior);inputs=cfg['inputs'];base_z=physics.load_arrays(inputs['reference'])
    path=verify_file_pin(inputs['model']);binding=physics.read_json(inputs['binding'])
    model=mujoco.MjModel.from_binary_path(str(path))
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(source=prior,seeds=[8311,8312],widths=[1,4],per_phase_budget=64,
        code=file_pin(__file__),scope='Fixed two known HOT3D unilateral windows. Compare spending 64 proposals on one parent or sharing them among up to four distinct surviving paths. Accumulated pose plus normalized impulse cost, table gate checked on survivors, exact state replay from time zero. Stop when no candidate survives. Different absolute actions across widths after path divergence. Candidate net wrench from endpoint momentum includes any table load, not a hand-only label for rejected candidates.'))
    rows=[]
    for pin in cfg['sources']:
        row,=physics.read_json(pin)['rows'];ref=physics.load_arrays(row['reference']);T=ref['body']
        z={**base_z,'elapsed_s':ref['t'],'COM_world_m':T[:,:3,3]+T[:,:3,:3]@base_z['COM_BODY_m'],
           'quaternion_wxyz':Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]}
        base=z['feedforward_PD_target20'][0];direct=np.clip(ref['q'][1:]+base-z['initial_qpos27'][:20],*model.actuator_ctrlrange.T)
        for seed in (8311,8312):
            for width in (1,4):
                folder=out/f"{row['clip']}_{seed}_{width}";folder.mkdir();start=perf_counter()
                result=search(model,path,binding,z,direct,base,width,seed,folder)
                rows.append(dict(clip=row['clip'],seed=seed,width=width,wall_s=perf_counter()-start,**result))
                write_json(out/'progress.json',rows)
                print(row['clip'],seed,width,result['valid_s'],flush=True)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);main(p.parse_args().out.resolve())
