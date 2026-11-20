"""Measure absolute forecast bias and candidate response on exactly matched feedback states."""
import argparse
from pathlib import Path
import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import test_centered_mug_screen as bench


def run(source,out):
    pin=bench.h3.file_pin(source/'report.json');experiment=bench.h3.physics.read_json(pin);intent=bench.h3.physics.read_json(experiment['intent'])
    root,_,gains,asset,bind,old,lib,_=bench.setup()
    checkpoint=torch.load(bench.h3.verify_file_pin(bench.h3.physics.read_json(intent['fit'])['model']),weights_only=True,map_location='cpu')
    torch.set_num_threads(1);net=bench.MugHorizonHead(checkpoint['n']);net.load_state_dict(checkpoint['state']);net.eval()
    out.mkdir(exist_ok=False)
    bench.h3.write_json(out/'intent.json',dict(source=pin,code=bench.h3.file_pin(__file__),scope=
        'One source every four controls on centered4 routes across ten new conditions. Re-evaluate four commands (witness, .65 previous correction, kinematic, quarter native gain), same exact state and warmstart of the executing plant. Two controls with the original 0.8 residual continuation. Compare predicted pose before and after nominal phase correction. Second endpoint is before any replan.'))
    rows=[]
    for clip,group in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
        fit=bench.h3.physics.load_arrays(gains['clips'][str(clip)]['data'])
        item,=bench.h3.physics.read_json(bench.h3.file_pin(root/group/'report.json'))['rows']
        ref=bench.h3.physics.load_arrays(item['reference']);T=ref['body'];pos=T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'];rot=Rotation.from_matrix(T[:,:3,:3])
        m0=bench.plant(asset,bind,1,1,1);Pn,_=bench.context(m0,bind,pos,rot)
        scorer=bench.CenteredScreen(Pn,None,fit,net,None)
        for row in experiment['rows']:
            if row['mode']!='centered4' or row['condition']=='nominal' or row['clip']!=clip:continue
            mass,mu,tau,offset=intent['conditions'][row['condition']];m=bench.plant(asset,bind,mass,mu,tau)
            P,_=bench.context(m,bind,pos,rot);d=bench.initial(m,P,old,offset)
            archive=bench.h3.physics.load_arrays(row['trace']);previous=np.zeros(20)
            scratch=[mujoco.MjData(m) for _ in range(4)]
            with Rollout(nthread=4) as engine:
                for i,cmd in enumerate(archive['ctrl']):
                    if i%4==0 and i+2<=len(fit['cmd']):
                        s=P.state(d);refstate=fit['state'][i];dx=bench.tangent_errors(m0,refstate,s[None])[0]
                        kin=bench.correct('kinematic',Pn.chart,s,refstate,dx,fit,i)
                        lin=bench.correct('native_gain',Pn.chart,s,refstate,dx,fit,i)
                        raw=np.clip(np.array([np.zeros(20),.65*previous,kin,.25*lin]),-.12,.12);seq=scorer.sequence(i,raw)[:,:2]
                        st,_=engine.rollout(m,scratch,np.tile(s,(4,1)),control=np.repeat(seq,64,axis=1),initial_warmstart=np.tile(d.qacc_warmstart,(4,1)))
                        with torch.no_grad():y=net(torch.from_numpy(bench.features(Pn.chart,s,np.zeros(m.nv),scorer.sequence(i,raw)))).numpy().reshape(4,3,12)
                        for h in (1,2):
                            truth,actual_R,_,_=P.chart.motion(st[:,64*h-1])
                            p,R,_=bench.decode(Pn.chart,s,y[:,h-1],h)
                            p0,R0,_=scorer.reference_pred[i][h-1];pr,Rr,_,_=scorer.nom[i+h]
                            for treatment,pe,Re in [('raw',p,R),('centered',pr+(p-p0),R*R0.inv()*Rr)]:
                                err=1000*np.linalg.norm(pe-truth,axis=1);ang=np.rad2deg((Re*actual_R.inv()).magnitude())
                                rows.append(dict(clip=clip,condition=row['condition'],phase=i+1,h=h,mode=treatment,
                                    mm=err.tolist(),deg=ang.tolist()))
                    d.ctrl[:]=cmd;mujoco.mj_step(m,d,nstep=64);np.testing.assert_array_equal(d.qpos,archive['qpos'][i])
                    previous=cmd-fit['cmd'][i]
    bench.h3.write_json(out/'report.json',dict(intent=bench.h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();run(a.source.resolve(),a.out.resolve())
