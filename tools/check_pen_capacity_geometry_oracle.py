"""Compare predicted pose interpolation with exact-state CPU geometry at native tick times."""
from pathlib import Path
import json,sys
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation,Slerp
from check_pen_native_load_suggestions import pen_load_step,LoadForecast
from supermimic_research.mechanics.transition_records import state_rotation
from supermimic_research.mechanics.pen_wrench_capacity import body_wrench_rays,directional_support,balanced_directions
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json
read=lambda p:json.loads(verify_file_pin(p).read_bytes())


def geometry(m,d,q,body,dirs):
    d.qpos[:]=q;mujoco.mj_forward(m,d)
    cs=[];sign=[]
    for c in d.contact:
        g,h=map(int,c.geom);o=m.geom_bodyid[[g,h]]==body
        if o.sum()==1 and c.dist<=0 and (m.geom(h if o[0] else g).name or '').startswith('robot/'):
            cs.append(c);sign.append(1 if o[1] else -1)
    rays=body_wrench_rays(cs,sign,com_world=d.xipos[body],world_from_body=d.xmat[body].reshape(3,3),length_scale_m=.10)
    return directional_support(rays,dirs)


def main(root,out):
    out.mkdir(exist_ok=False);base=root.parent;master=json.loads((root/'report.json').read_bytes())
    pin=read(file_pin(base/'pen_load_forecast_v1/report.json'))['outcomes']['state']['model']
    pars=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=True)
    net=LoadForecast(torch.zeros(2,len(pars['center'])),torch.zeros(2,len(pars['out_center'])));net.load_state_dict(pars);net.double().eval().requires_grad_(False)
    dirs=balanced_directions()
    write_json(out/'intent.json',dict(source=file_pin(root/'report.json'),load_net=pin,role='Reanalysis of previously inspected held-out TEST policy, no model fitting. Same CPU collision scene across conditions. True eight pre-step poses vs true start/end interpolated vs predicted end interpolated. Time fractions 0..7/8.'))
    role='TEST';folder=root/'native'/role
    m=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'));d=mujoco.MjData(m);body=m.body('object/cube').id;qa=m.jnt_qposadr[m.body_jntadr[body]];qs=m.jnt_qposadr[m.actuator_trnid[:,0]]
    assert not m.npair and m.opt.cone==mujoco.mjtCone.mjCONE_PYRAMIDAL
    with np.load(folder/'physical_context.npz') as f:
        mass=float(f['body_mass'][0,body]);g=f['gravity'][0];inert=f['body_inertia'][0,body];Q=Rotation.from_quat(f['body_iquat'][0,body,[1,2,3,0]]).as_matrix();I=Q@np.diag(inert)@Q.T
    results=[]
    for row in master['roles'][role]['rows']:
        with np.load(verify_file_pin(row['packet'])) as f:x=f['x'];pd=f['pd'];target=f['capacity']
        with np.load(verify_file_pin(row['trace'])) as f:
            pre=f['pre_qpos'];end=f['post_qpos'][-1];d.mocap_pos[:]=f['post_mocap_pos'][0];d.mocap_quat[:]=f['post_mocap_quat'][0]
        x0=torch.from_numpy(x[:55]).double();nxt=pen_load_step(x0,torch.from_numpy(pd).double(),None,net,mass,I,g)[0].detach().numpy()
        hat=end.copy();hat[qs]=nxt[:20];hat[qa:qa+3]=nxt[40:43];hat[qa+3:qa+7]=Rotation.from_matrix(state_rotation(nxt)).as_quat()[[3,0,1,2]]
        def interpolate(b):
            v=np.arange(8)/8;start=pre[0];q=start+v[:,None]*(b-start)
            ids=qa+np.array([4,5,6,3]);rot=Slerp([0.,1.],Rotation.from_quat(np.stack([start[ids],b[ids]])))(v)
            q[:,qa+3:qa+7]=rot.as_quat()[:,[3,0,1,2]]
            return q
        hand_only=end.copy();hand_only[qs]=hat[qs]
        object_only=end.copy();object_only[qa:qa+7]=hat[qa:qa+7]
        pred={}
        for k,poses in [('measured_substeps',pre),('measured_endpoint',interpolate(end)),('learned_hand_true_object',interpolate(hand_only)),('true_hand_learned_object',interpolate(object_only)),('learned_endpoint',interpolate(hat))]:
            pred[k]=np.mean([geometry(m,d,q,body,dirs) for q in poses],axis=0).tolist()
        results.append(dict(phase=row['phase'],name=row['name'],ok=row['valid'] and not row['terminal'],native=target.tolist(),**pred))
    summary={}
    for k in pred:
        groups={}
        for r in results:groups.setdefault(r['phase'],{})[r['name']]=r
        summary[k]={}
        for kind,names in [('basis',[f'j{j}' for j in master['roles'][role]['selected']]),('mixed',['mixed'])]:
            D=[];T=[]
            for group in groups.values():
                for name in names:
                    a,b=[group[name+s] for s in ('_minus','_plus')]
                    if not(a['ok'] and b['ok']):continue
                    D.append(np.subtract(b[k],a[k]));T.append(np.subtract(b['native'],a['native']))
            D,T=np.array(D),np.array(T)
            summary[k][kind]=dict(pairs=len(D),relative_error=float(np.linalg.norm(D-T)/np.linalg.norm(T)),positive_dot=int(np.sum(np.sum(D*T,axis=1)>0)),amp_ratio=float(np.linalg.norm(D)/np.linalg.norm(T)))
        summary[k]['MAE']=float(np.mean([np.mean(np.abs(np.subtract(r[k],r['native']))) for r in results if r['ok']]))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),summary=summary,rows=results))
    print(summary)
if __name__=='__main__':main(*map(Path,sys.argv[1:]))

