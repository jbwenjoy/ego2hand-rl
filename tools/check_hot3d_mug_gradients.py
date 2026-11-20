"""Compare learned action gradients with same-source native central differences."""
import argparse
from pathlib import Path
import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from test_mug_learned_planning import physics,file_pin,verify_file_pin,write_json,MugStepHead,MugStepChart,mean_reference_wrench,kinematics


def differentiable_cost(y,position,quaternion,wrench,valid):
    v=y[:,3:6]*np.deg2rad(10)
    angle=torch.linalg.vector_norm(v,dim=1,keepdim=True)
    xyz=v*.5*torch.sinc(angle/(2*np.pi));w=torch.cos(.5*angle)
    rv=quaternion[3]*xyz-w*quaternion[:3]-torch.linalg.cross(quaternion[:3].expand_as(xyz),xyz)
    rw=(w[:,0]*quaternion[3]+(xyz*quaternion[:3]).sum(1)).abs()
    angular=2*torch.atan2(rv.norm(dim=1),rw)/np.deg2rad(10)
    pose=((y[:,:3]-position)**2).sum(1)+angular**2
    e=((y[:,6:]-wrench)**2).sum(1)
    return pose,(e/(1+e))*valid


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    fit_pin=file_pin(root/'mug_shared_step_v1/report.json');fit=physics.read_json(fit_pin)
    intent=physics.read_json(fit['intent']);inputs=intent['inputs'];bank=physics.read_json(intent['source'])
    path=verify_file_pin(inputs['model']);m=mujoco.MjModel.from_binary_path(str(path))
    binding=physics.read_json(inputs['binding']);chart=MugStepChart(m,binding);body=chart.body
    mass=m.body_mass[body];R=Rotation.from_quat(m.body_iquat[body,[1,2,3,0]]).as_matrix();I=R@np.diag(m.body_inertia[body])@R.T
    ck=torch.load(verify_file_pin(physics.read_json(fit['frozen'])['pins']['shared']),weights_only=True,map_location='cpu')
    net=MugStepHead(ck['n']);net.load_state_dict(ck['state']);net.double().requires_grad_(False).eval();torch.set_num_threads(1)
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float64)
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(fit=fit_pin,code=file_pin(__file__),delta_rad=.01,probe_norm_rad=.03,
        scope='Six evenly spaced noninitial control states per held-out seed8312 one-parent path, valid wrench target required. Center the nominal IK proposal inside limits by 0.031 rad. 20 paired +/-0.01 rad native probes. Test descent/ascent commands at 0.03 rad joint vector norm for each objective on the exact same state; reference pose and momentum target frozen. Native comparison includes pose and/or bounded normalized net wrench; no effort penalty.'))
    rows=[]
    for src in bank['rows']:
        if src['seed']!=8312 or src['width']!=1:continue
        clip=src['clip']
        group='mug_annotated_windows_original_v1' if clip==1884 else 'mug_unilateral_interval_v1'
        row,=physics.read_json(file_pin(root/group/'report.json'))['rows']
        ref=physics.load_arrays(row['reference']);T=ref['body']
        p=T[:,:3,3]+T[:,:3,:3]@chart.com;rot=Rotation.from_matrix(T[:,:3,:3])
        targets=mean_reference_wrench(p,rot.as_quat()[:,[3,0,1,2]],mass,I,m.opt.gravity)
        with np.load(verify_file_pin(src['bank'])) as f:archive=dict(f)
        eligible=np.flatnonzero(targets['valid_interval'][:len(archive['source'])]);eligible=eligible[eligible>0]
        phases=eligible[np.round(np.linspace(0,len(eligible)-1,min(6,len(eligible)))).astype(int)]
        workers=[mujoco.MjData(m) for _ in range(4)]
        with Rollout(nthread=4) as roll:
            for t in phases:
                state=archive['source'][t,0];warm=archive['warmstart'][t,0]
                center=np.clip(archive['cmd'][t,0],m.actuator_ctrlrange[:,0]+.031,m.actuator_ctrlrange[:,1]-.031)
                pos,r0,v,wb=chart.motion(state)
                cv=r0*Rotation.from_rotvec(.02*wb)
                wtarget=targets['wrench_world'][t]
                target_local=r0.inv().apply(wtarget.reshape(2,3))/np.array([[chart.weight],[chart.weight*.05]])
                features=chart.features(state,warm,np.vstack([center,center+np.eye(20)]))
                x0=tensor(features[0]);deriv=tensor(features[1:].astype(float)-features[0].astype(float))
                def learned(delta):
                    y=net(x0+delta@deriv)
                    pose,load=differentiable_cost(y,tensor(r0.inv().apply(p[t+1]-pos-v*.02)/.01),
                        tensor((cv.inv()*rot[t+1]).as_quat()),tensor(target_local.ravel()),1.)
                    return torch.stack([pose,load,pose+.2*load],dim=1)
                zero=torch.zeros((1,20),dtype=torch.double,requires_grad=True);value=learned(zero)
                grad=np.stack([torch.autograd.grad(value[0,i],zero,retain_graph=True)[0].numpy()[0] for i in range(3)])
                check_eps=1e-5
                check=(learned(tensor(check_eps*np.eye(20)))-learned(tensor(-check_eps*np.eye(20)))).numpy().T/(2*check_eps)
                fd_error=np.max(abs(check-grad))
                if not np.allclose(check,grad,atol=1e-5,rtol=.005):raise ValueError('Autograd verification failed')
                unit=grad/np.maximum(np.linalg.norm(grad,axis=1,keepdims=True),1e-12)
                offsets=np.vstack([np.zeros(20),.01*np.eye(20),-.01*np.eye(20),-.03*unit,.03*unit])
                cmd=center+offsets;n=len(cmd);assert (cmd>=m.actuator_ctrlrange[:,0]).all() and (cmd<=m.actuator_ctrlrange[:,1]).all()
                states,_=roll.rollout(m,workers,np.tile(state,(n,1)),control=np.repeat(cmd[:,None],64,axis=1),
                                      initial_warmstart=np.tile(warm,(n,1)))
                ends=states[:,-1,:]
                _,_,v0,L0=kinematics(state[None],chart.qa,chart.da,chart.com,I,m.nq)
                ep,eR,v1,L1=kinematics(ends,chart.qa,chart.da,chart.com,I,m.nq)
                wrench=np.column_stack([mass*((v1-v0)/.02-m.opt.gravity),(L1-L0)/.02])
                mm=1000*np.linalg.norm(ep-p[t+1],axis=1);degrees=np.rad2deg((eR*rot[t+1].inv()).magnitude())
                e=(np.linalg.norm(wrench[:,:3]-wtarget[:3],axis=1)/chart.weight)**2+(np.linalg.norm(wrench[:,3:]-wtarget[3:],axis=1)/(.05*chart.weight))**2
                pose=(mm/10)**2+(degrees/10)**2;support=e/(1+e);cost=np.column_stack([pose,support,pose+.2*support])
                encoded=tensor(chart.targets(state,ends,wrench))
                pose2,load2=differentiable_cost(encoded,tensor(r0.inv().apply(p[t+1]-pos-v*.02)/.01),
                    tensor((cv.inv()*rot[t+1]).as_quat()),tensor(target_local.ravel()),1.)
                np.testing.assert_allclose(pose2.numpy(),pose,rtol=1e-5,atol=1e-6)
                np.testing.assert_allclose(load2.numpy(),support,rtol=1e-5,atol=1e-6)
                truth=((cost[1:21]-cost[21:41])/.02).T
                metrics={}
                for i,name in enumerate(('pose','wrench','combined')):
                    norm=np.linalg.norm(truth[i])
                    metrics[name]=dict(cosine=(unit[i]@truth[i])/max(norm,1e-12),relative_error=np.linalg.norm(grad[i]-truth[i])/max(norm,1e-12),
                        base=cost[0,i],descent=cost[41+i,i],ascent=cost[44+i,i],native=truth[i].tolist(),learned=grad[i].tolist())
                rows.append(dict(clip=clip,phase=int(t+1),metrics=metrics,self_gradient_max_error=fd_error))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows,native_ticks=len(rows)*47*64))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
