"""Subtract the frozen model's phase-specific bias and verify only the first planning interval."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import torch
from test_mug_event_budget import setup,initial,NativeStrictWindow,context,plant,LIMITS,h3
from test_mug_witness_mpc import correct,momentum_load,tangent_errors
from supermimic_research.mechanics.mug_horizon import features,decode,MugHorizonHead


class CenteredScreen:
    def __init__(self,P,checker,fit,net,executor):
        self.P=P;self.check=checker;self.fit=fit;self.net=net;self.executor=executor
        self.buffers=[mujoco.MjData(P.m) for _ in range(4)]
        states=fit['state'];cmds=fit['cmd'];self.nom=[P.chart.motion(x) for x in states]
        self.load=[momentum_load(P,a,b) for a,b in zip(states[:-1],states[1:])]
        self.reference_pred=[]
        for i in range(len(cmds)):
            seq=self.sequence(i,np.zeros((1,P.m.nu)))
            with torch.inference_mode():
                y=net(torch.from_numpy(features(P.chart,states[i],np.zeros(P.m.nv),seq))).numpy().reshape(1,3,12)
            self.reference_pred.append([decode(P.chart,states[i],y[:,h-1],h) for h in (1,2,3)])

    def sequence(self,i,delta):
        return np.stack([np.clip(self.fit['cmd'][min(i+j,len(self.fit['cmd'])-1)]+(.8**j)*delta,
                        *self.P.m.actuator_ctrlrange.T) for j in range(3)],axis=1)

    def predicted(self,i,state,seq,centered):
        P=self.P
        with torch.inference_mode():
            y=self.net(torch.from_numpy(features(P.chart,state,np.zeros(P.m.nv),seq))).numpy().reshape(-1,3,12)
        scores=[];gates=[];r0=P.chart.motion(state)[1];nominal_r0=self.nom[i][1]
        for h in range(1,min(3,len(self.fit['cmd'])-i)+1):
            p,R,w=decode(P.chart,state,y[:,h-1],h)
            refp,refr,_,_=self.nom[i+h]
            average_world=np.mean([self.nom[k][1].apply(self.load[k].reshape(2,3)) for k in range(i,i+h)],axis=0)
            refw=nominal_r0.inv().apply(average_world).ravel()
            w=r0.inv().apply(w.reshape(-1,3)).reshape(-1,6)
            if centered:
                pbase,rbase,wbase=self.reference_pred[i][h-1]
                wbase=nominal_r0.inv().apply(wbase.reshape(-1,3)).reshape(-1,6)
                p=refp+(p-pbase);R=R*rbase.inv()*refr;w=refw+(w-wbase)
            delta=w-refw
            e=np.sum(delta[:,:3]**2,axis=1)/P.chart.weight**2+np.sum(delta[:,3:]**2,axis=1)/(.05*P.chart.weight)**2
            c=(np.linalg.norm(p-refp,axis=1)/.005)**2+((R*refr.inv()).magnitude()/np.deg2rad(8))**2+.2*e/(1+e)
            c+=.04*np.mean(((seq[:,h-1]-self.fit['cmd'][i+h-1])/.08)**2,axis=1)
            risk=np.maximum(100*np.linalg.norm(p-P.pos[i+h],axis=1),np.rad2deg((R*P.rot[i+h].inv()).magnitude())/10.)
            scores.append(c);gates.append(np.maximum(1.,risk))
        return np.array(scores),np.array(gates)

    def verify(self,slot,state,i,command):
        P=self.P;d=self.buffers[slot];m=P.m
        mujoco.mj_resetData(m,d);mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_FULLPHYSICS);d.qacc_warmstart[:]=0.
        metric=self.check.step(d,i,command);end=P.state(d)
        p,R,_,_=P.chart.motion(end);pt,rt,_,_=self.nom[i+1]
        delta=momentum_load(P,state,end)-self.load[i]
        e=delta[:3]@delta[:3]/P.chart.weight**2+delta[3:]@delta[3:]/(.05*P.chart.weight)**2
        cost=(np.linalg.norm(p-pt)/.005)**2+((R*rt.inv()).magnitude()/np.deg2rad(8))**2+.2*e/(1+e)
        cost+=.04*np.mean(((command-self.fit['cmd'][i])/.08)**2)
        return max(1.,float(np.max(metric/LIMITS))),float(cost)

    def choose(self,mode,state,i,prev):
        P=self.P;ref=self.fit['state'][i];cmd0=self.fit['cmd'][i];dx=tangent_errors(P.m,ref,state[None])[0]
        if np.max(abs(dx))<1e-10 or mode=='open':return cmd0,0,0
        kin=correct('kinematic',P.chart,state,ref,dx,self.fit,i)
        lin=correct('native_gain',P.chart,state,ref,dx,self.fit,i)
        raw=[np.zeros(20),.65*prev,.5*kin,kin,.25*lin,.65*lin]
        rng=np.random.default_rng(np.random.SeedSequence([8541,i]));noise=rng.normal(size=(3,20))
        noise/=np.sqrt(np.mean(noise**2,axis=1,keepdims=True))
        amp=min(.09,.018+.003*np.linalg.norm(dx[P.chart.da:P.chart.da+3])/.001)
        for v in noise:raw.extend([.6*prev+amp*v,.6*prev-amp*v])
        seq=self.sequence(i,np.clip(raw,-.12,.12))
        if mode=='analytic4':
            ids=np.array([0,1,3,4]);fc=np.zeros(12);fg=np.ones(12)
        else:
            cost,gate=self.predicted(i,state,seq,mode!='uncentered4')
            fc=(cost[1:]*np.array([.5,.25])[:len(cost)-1,None]).sum(0)
            fg=gate[1:].max(axis=0,initial=1.)
            order=np.lexsort((np.arange(12),cost[0]+fc,np.maximum(gate[0],fg),gate[0]))
            if mode=='learned':
                k=order[0];return seq[k,0],int(k),0
            count=2 if mode=='centered2' else 4
            ids=np.r_[0,order[order!=0][:count-1]]
        futures=[self.executor.submit(self.verify,j,state,i,seq[k,0]) for j,k in enumerate(ids)]
        measured=np.array([f.result() for f in futures]);g,c=measured.T
        k=ids[np.lexsort((ids,c+fc[ids],np.maximum(g,fg[ids]),g))[0]]
        return seq[k,0],int(k),len(ids)*64


def run(out):
    torch.set_num_threads(1);root,_,models,asset,bind,old,lib,_=setup()
    fitpin=h3.file_pin(root/'mug_shared_horizon_v1/report.json');weights=h3.physics.read_json(fitpin)['model']
    ck=torch.load(h3.verify_file_pin(weights),map_location='cpu',weights_only=True)
    net=MugHorizonHead(ck['n']);net.load_state_dict(ck['state']);net.eval()
    samples=qmc.scale(qmc.LatinHypercube(d=4,seed=8581).random(10),[.95,.95,.90,-.001],[1.05,1.05,1.10,.001])
    conditions={'nominal':[1.,1.,1.,0.],**{f'new_{i:02}':v.tolist() for i,v in enumerate(samples)}}
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(fit=fitpin,code=h3.file_pin(__file__),conditions=conditions,seed=8581,
        scope='New state combinations, same dense12 candidate pool and known nominal trajectory. Subtract phase-specific nominal predictions to transport deviations of predicted COM/orientation/mean wrench to the true witness. Compare averaged loads in the respective source BODY frames. Verify first window only. All verified shortlists include the witness action. Compare centered4 to uncentered4, analytic4, open. Primary: both nominal trajectories complete and positive median paired time over open and uncentered4 on both clips. Report all deadline overruns, excluding plant execution.'))
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for clip,group in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
            fit=h3.physics.load_arrays(models['clips'][str(clip)]['data'])
            item,=h3.physics.read_json(h3.file_pin(root/group/'report.json'))['rows'];ref=h3.physics.load_arrays(item['reference']);T=ref['body']
            pos=T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'];rot=Rotation.from_matrix(T[:,:3,:3])
            nominal=plant(asset,bind,1,1,1);Pn,_=context(nominal,bind,pos,rot);screen=CenteredScreen(Pn,NativeStrictWindow(Pn,lib),fit,net,ex)
            for j,(condition,(mass,mu,tau,offset)) in enumerate(conditions.items()):
                m=plant(asset,bind,mass,mu,tau);P,py=context(m,bind,pos,rot);actual=NativeStrictWindow(P,lib)
                modes=['open','analytic4','uncentered4','centered4','centered2','learned'];modes=modes[j%6:]+modes[:j%6]
                for mode in modes:
                    d=initial(m,P,old,offset);prev=np.zeros(20);cs=[];qs=[];metrics=[];records=[]
                    for i,cmd0 in enumerate(fit['cmd']):
                        t=perf_counter();cmd,idx,ticks=screen.choose(mode,P.state(d),i,prev)
                        records.append(dict(i=i,choice=idx,ticks=ticks,ms=1000*(perf_counter()-t)))
                        prev=cmd-cmd0;v=actual.step(d,i,cmd);qs.append(d.qpos.copy());metrics.append(v);cs.append(cmd)
                        if np.any(v>LIMITS):break
                    passed=len(cs)-int(np.any(metrics[-1]>LIMITS))
                    replay=initial(m,P,old,offset)
                    for i,c in enumerate(cs):
                        np.testing.assert_array_equal(py.step(replay,i,c),metrics[i]);np.testing.assert_array_equal(replay.qpos,qs[i])
                    if condition=='nominal':np.testing.assert_array_equal(qs,fit['state'][1:,1:1+m.nq])
                    f=out/f'{clip}_{condition}_{mode}.npz';np.savez_compressed(f,qpos=qs,ctrl=cs,metrics=metrics)
                    rows.append(dict(clip=clip,condition=condition,mode=mode,passing=passed,full=passed==len(fit['cmd']),records=records,trace=h3.file_pin(f)))
                h3.write_json(out/'progress.json',rows);print(clip,condition,[r['passing'] for r in rows[-6:]],flush=True)
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
