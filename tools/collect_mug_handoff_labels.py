"""Counterfactual actions followed by the actual frozen H3 controller on two HOT3D windows."""
import argparse
from pathlib import Path
from time import perf_counter
import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import test_hot3d_model_horizon as h3


class Planner:
    def __init__(self,item,inputs,weights):
        self.clip=item['clip']
        self.path=h3.verify_file_pin(inputs['model'])
        self.m=mujoco.MjModel.from_binary_path(str(self.path))
        self.binding=h3.physics.read_json(inputs['binding'])
        self.chart=h3.MugStepChart(self.m,self.binding)
        self.old=h3.physics.load_arrays(inputs['reference'])
        ref=h3.physics.load_arrays(item['reference']);self.frame=ref['body']
        self.reference_q=ref['q'];self.base=self.old['feedforward_PD_target20'][0]
        self.ik=np.clip(ref['q'][1:]+self.base-self.old['initial_qpos27'][:20],*self.m.actuator_ctrlrange.T)
        self.pos=self.frame[:,:3,3]+self.frame[:,:3,:3]@self.chart.com
        self.rot=Rotation.from_matrix(self.frame[:,:3,:3])
        body=self.chart.body;self.mass=self.m.body_mass[body]
        Q=Rotation.from_quat(self.m.body_iquat[body,[1,2,3,0]]).as_matrix()
        self.inertia=Q@np.diag(self.m.body_inertia[body])@Q.T
        self.load=h3.mean_reference_wrench(self.pos,self.rot.as_quat()[:,[3,0,1,2]],self.mass,self.inertia,self.m.opt.gravity)
        self.net=weights;self.sig=mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.workers=[mujoco.MjData(self.m) for _ in range(4)]
        self.table=self.m.geom('mug_task_table').id;self.mug=self.m.geom_bodyid==body

    def initial(self):
        d=mujoco.MjData(self.m)
        d.qpos[:]=self.old['initial_qpos27'];d.qvel[:]=self.old['initial_qvel26'];d.ctrl[:]=self.base
        mujoco.mj_forward(self.m,d)
        return d

    def state(self,d):
        s=np.empty(mujoco.mj_stateSize(self.m,self.sig));mujoco.mj_getState(self.m,d,s,self.sig);return s

    def score(self,p,R,w,i,h,cmd):
        mm=1000*np.linalg.norm(p-self.pos[i+h],axis=1);deg=np.rad2deg((R*self.rot[i+h].inv()).magnitude())
        cost=(mm/10)**2+(deg/10)**2+.01*np.mean(((cmd-self.ik[i+h-1])/.12)**2,axis=1)
        if self.load['valid_interval'][i:i+h].all():
            diff=w-self.load['wrench_world'][i:i+h].mean(0)
            sq=(diff[:,:3]**2).sum(1)/self.chart.weight**2+(diff[:,3:]**2).sum(1)/(.05*self.chart.weight)**2
            cost+=.2*sq/(1+sq)
        return np.maximum(1.,np.maximum(mm,deg)/10),cost

    def options(self,d,i,seed,roll,*,include_features=False):
        pre=self.state(d);last=d.ctrl.copy()
        residual=last-self.ik[i-1] if i else np.zeros(20)
        raw=h3.propose(np.random.default_rng(np.random.SeedSequence([seed,i,0])),64,self.ik[i],self.base,last,residual,self.m.actuator_ctrlrange)
        seq=h3.continuations(raw,self.ik,i,self.m.actuator_ctrlrange)
        x=h3.features(self.chart,pre,d.qacc_warmstart,seq)
        with torch.inference_mode():pred=self.net(torch.from_numpy(x)).numpy().reshape(-1,3,12)
        g1,c1=self.score(*h3.decode(self.chart,pre,pred[:,0],1),i,1,raw)
        fg=np.ones_like(g1);fc=np.zeros_like(c1)
        for h in range(2,min(3,len(self.ik)-i)+1):
            g,c=self.score(*h3.decode(self.chart,pre,pred[:,h-1],h),i,h,seq[:,h-1]);fg=np.maximum(fg,g);fc+=c
        order=h3.order_horizon(g1,fg,c1+fc)
        ids=np.r_[np.arange(4),order[order>=4][:4]];cs=raw[ids]
        st,_=roll.rollout(self.m,self.workers,np.tile(pre,(8,1)),control=np.repeat(cs[:,None],64,axis=1),
                          initial_warmstart=np.tile(d.qacc_warmstart,(8,1)))
        end=st[:,-1].copy();_,_,v0,L0=h3.kinematics(pre[None],self.chart.qa,self.chart.da,self.chart.com,self.inertia,self.m.nq)
        p,R,v,L=h3.kinematics(end,self.chart.qa,self.chart.da,self.chart.com,self.inertia,self.m.nq)
        w=np.column_stack([self.mass*((v-v0)/.02-self.m.opt.gravity),(L-L0)/.02])
        eg,ec=self.score(p,R,w,i,1,cs)
        order=h3.order_horizon(eg,fg[ids],ec+fc[ids])
        packed=None
        if include_features:
            # Native verification of the current candidates is already available at inference.
            pos_error=(p-self.pos[i+1])/.01
            angle_error=(self.rot[i+1].inv()*R).as_rotvec()/np.deg2rad(10)
            packed=np.column_stack([x[ids],pred[ids].reshape(8,-1),pos_error,angle_error,
                v,L,w,eg,ec,fg[ids],fc[ids]]).astype(np.float32)
        return dict(ids=ids,cmd=cs,end=end,order=order,x=packed,gate=eg,cost=ec+fc[ids])

    def advance(self,d,i,cmd,expected=None):
        assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
        d.ctrl[:]=cmd;peak=0.;force=np.empty(6)
        for _ in range(64):
            mujoco.mj_step(self.m,d);a,b=d.contact.geom.T
            for k in np.flatnonzero(((a==self.table)&self.mug[b])|((b==self.table)&self.mug[a])):
                mujoco.mj_contactForce(self.m,d,k,force);peak=max(peak,force[0])
        state=self.state(d)
        if expected is not None:np.testing.assert_array_equal(state,expected)
        p,R,_,_=self.chart.motion(state)
        mm=1000*np.linalg.norm(p-self.pos[i+1]);deg=np.rad2deg((R*self.rot[i+1].inv()).magnitude())
        return mm<=10 and deg<=10 and peak<=.1, (mm,deg,peak)


def configuration():
    root=Path(__file__).resolve().parents[1]/'runs'
    bank=h3.file_pin(root/'mug_model_horizon_trials_v1/report.json');report=h3.physics.read_json(bank)
    fit=h3.physics.read_json(h3.physics.read_json(report['intent'])['fit']);inputs=h3.physics.read_json(fit['intent'])['inputs']
    ck=torch.load(h3.verify_file_pin(fit['model']),map_location='cpu',weights_only=True)
    torch.set_num_threads(1);net=h3.MugHorizonHead(ck['n']);net.load_state_dict(ck['state']);net.eval()
    planners={}
    for group in ('mug_annotated_windows_original_v1','mug_unilateral_interval_v1'):
        item,=h3.physics.read_json(h3.file_pin(root/group/'report.json'))['rows']
        planners[item['clip']]=Planner(item,inputs,net)
    return bank,report,planners


def collect(out):
    bank,report,planners=configuration()
    out.mkdir(exist_ok=False,parents=True)
    h3.write_json(out/'intent.json',dict(source=bank,code=h3.file_pin(__file__),budget=6,
        scope='Each H3 archive, phase 3 and then every 3 controls including the failure region. Fork all eight shortlisted actions from the identical live data, then run the frozen H3 feedback policy for up to five more controls. Count consecutive passing windows including action one. Stop at first position/rotation/table breach. Numeric bit parity required along the archived branch. Fixed heldout seeds 8437,8438; first six seeds train. Results not independent human/physical trials.'))
    records=[];summaries=[];started=perf_counter()
    for run in report['rows']:
        if run['mode']!='h3':continue
        clip=run['clip'];seed=run['seed'];P=planners[clip];arc=h3.physics.load_arrays(run['trace'])
        d=P.initial();branch=mujoco.MjData(P.m);states=[];used=0
        with Rollout(nthread=4) as roll:
            for i,rec in enumerate(run['records']):
                if i%3==2:
                    o=P.options(d,i,seed,roll,include_features=True);best=o['order'][0]
                    np.testing.assert_array_equal(o['ids'],rec['shortlist']);assert o['ids'][best]==rec['choice']
                    values=[];observed=[];steps=min(6,len(P.ik)-i)
                    for k in range(8):
                        mujoco.mj_copyData(branch,P.m,d);count=0;endpoints=[]
                        for h in range(steps):
                            if h:
                                child=P.options(branch,i+h,seed,roll);idx=child['order'][0];used+=8*64
                                cmd=child['cmd'][idx];expected=child['end'][idx]
                            else:cmd=o['cmd'][k];expected=o['end'][k]
                            ok,info=P.advance(branch,i+h,cmd,expected);endpoints.append(info)
                            if k==best:
                                np.testing.assert_array_equal(branch.qpos,arc['qpos'][i+h])
                            if not ok:break
                            count+=1
                        values.append(count);observed.append(endpoints)
                    assert values[best]==min(steps,max(0,round(run['valid_s']/.02)-i))
                    states.append(dict(clip=clip,seed=seed,phase=i+1,best=int(best),values=values,first_pass=[v>=1 for v in values],observed=observed))
                    records.append((clip,seed,i+1,o['x'],o['cmd'],o['ids'],o['gate'],o['cost'],int(best),values))
                    used+=8*64
                P.advance(d,i,arc['ctrl'][i]);np.testing.assert_array_equal(d.qpos,arc['qpos'][i])
        summaries.extend(states)
        h3.write_json(out/'progress.json',dict(last=[clip,seed],groups=len(records),seconds=perf_counter()-started))
        print(clip,seed,len(states),used,flush=True)
    data=out/'groups.npz';np.savez_compressed(data,**{key:np.array(items) for key,items in zip(
        ('clip','seed','phase','x','cmd','ids','gate','cost','baseline','passing'),zip(*records))})
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),data=h3.file_pin(data),rows=summaries,wall_s=perf_counter()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);collect(p.parse_args().out.resolve())
