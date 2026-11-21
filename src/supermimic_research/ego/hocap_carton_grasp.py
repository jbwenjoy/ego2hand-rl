"""Contact-aware offline correction of robot finger targets around source-object geometry."""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from .hocap_carton_control import TIPS, quat_from_rotation


def pad_corrected_ik(model, hand, source_object, wrist, baseline_q, mesh, preload):
    """Use intended contact only when source fingertips approach opposing faces.

    Each pad follows its source point tangentially. Signed mesh distances add
    a commanded preload. The actual carton remains free during subsequent replay.
    """
    d=mujoco.MjData(model)
    joints=model.actuator_trnid[:,0]
    qidx=model.jnt_qposadr[joints]
    dof=model.jnt_dofadr[joints]
    lo=np.maximum(model.jnt_range[joints,0],model.actuator_ctrlrange[:,0])+1e-7
    hi=np.minimum(model.jnt_range[joints,1],model.actuator_ctrlrange[:,1])-1e-7
    finger_ids=[[model.geom(f'right_finger{f}_{part}_col').id
                 for part in ('link2','link3','link4','tip_sensor')] for f in range(1,6)]
    sites=[model.site(f'right_finger{i}_tip').id for i in range(1,6)]
    gid=model.geom('carton_collision').id
    eq=ConvexHull(mesh.vertices).equations
    local=np.einsum('fij,fkj->fki',source_object[:,:3,:3].transpose(0,2,1),
                    hand[:,TIPS]-source_object[:,None,:3,3])
    source_gap=(local@eq[:,:3].T+eq[:,3]).max(axis=2)
    active=np.clip((.028-np.maximum(source_gap[:,0],np.median(source_gap[:,1:],axis=1)))/.014,0,1)
    # Smoothly release each finger according to its own distance to the source scan.
    gates=active[:,None]*np.clip((.025-source_gap)/.012,0,1)
    qs=[];errors=[]
    jp=np.empty((3,model.nv));jr=np.empty_like(jp)
    closest=np.empty(6)
    for t,pose in enumerate(source_object):
        d.qpos[:3]=wrist[t,:3,3];d.qpos[3:7]=quat_from_rotation(wrist[t,:3,:3])
        d.qpos[-7:-4]=pose[:3,3];d.qpos[-4:]=quat_from_rotation(pose[:3,:3])
        initial=baseline_q[t] if t==0 else np.clip(qs[-1]+baseline_q[t]-baseline_q[t-1],lo,hi)
        if np.min(source_gap[t])>.045:
            qs.append(baseline_q[t]);errors.append(0.);continue
        last=None
        cache={}
        def evaluate(q):
            nonlocal last
            if last is not None and np.array_equal(last,q): return
            last=q.copy()
            d.qpos[qidx]=q
            mujoco.mj_kinematics(model,d);mujoco.mj_comPos(model,d);mujoco.mj_collision(model,d)
            contacts={}
            for c in d.contact:
                p=(int(c.geom1),int(c.geom2))
                if gid not in p: continue
                other=p[0] if p[1]==gid else p[1]
                normal=c.frame[:3] * (1. if p[0]==gid else -1.)
                if other not in contacts or c.dist<contacts[other][0]:
                    contacts[other]=(float(c.dist),normal.copy(),c.pos.copy())
            vals=[]; rows=[]
            for f,group in enumerate(finger_ids):
                delta=d.site_xpos[sites[f]]-hand[t,TIPS[f]]
                mujoco.mj_jacSite(model,d,jp,jr,sites[f])
                vals.extend(delta*55.)
                rows.extend(jp[:,dof]*55.)
                for j,g in enumerate(group):
                    if g in contacts:
                        gap,normal,point=contacts[g]
                    else:
                        gap=mujoco.mj_geomDistance(model,d,gid,g,.25,closest)
                        normal=closest[3:]-closest[:3]
                        if np.linalg.norm(normal)>1e-9:
                            normal=normal/np.linalg.norm(normal)*np.sign(gap)
                        else:
                            # A GJK zero-distance tie carries no reliable derivative.
                            normal=np.zeros(3)
                        point=closest[3:].copy()
                    # A virtual pad may enter the fixed object to generate preload
                    # through compliant joint tracking in the independent free replay.
                    if j==3:
                        target=-preload
                        residual=(gap-target)*gates[t,f]+min(gap-.001,0)*(1-gates[t,f])
                        slope=gates[t,f]+(1-gates[t,f])*(gap<.001)
                        weight=350.
                    else:
                        allowed=preload*gates[t,f] if j==2 else 0.
                        residual=min(gap+allowed,0.)
                        slope=float(gap < -allowed)
                        weight=280.
                    if slope:
                        mujoco.mj_jac(model,d,jp,jr,point,model.geom_bodyid[g])
                        row=normal@jp[:,dof]*weight*slope
                    else: row=np.zeros(20)
                    vals.append(residual*weight);rows.append(row)
            vals.extend(.22*(q-baseline_q[t]))
            vals.extend(.18*(q-initial))
            cache['y']=np.array(vals)
            cache['J']=np.vstack([*rows,.22*np.eye(20),.18*np.eye(20)])
        def residual(q):
            evaluate(q)
            return cache['y']
        def jac(q):
            evaluate(q)
            return cache['J']
        result=least_squares(residual,initial,jac=jac,bounds=(lo,hi),max_nfev=45,ftol=2e-5,xtol=2e-5,gtol=2e-5)
        qs.append(result.x);errors.append(result.cost)
    return np.stack(qs),gates,errors
