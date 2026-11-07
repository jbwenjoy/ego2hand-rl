"""Explicit, separately saved contact-mode variants and full native wrench bases.

No material parameter here is calibrated. Frozen models must be loaded afresh;
only declared pad/object pair dimensions and spin coefficients may differ.
"""
import numpy as np


def differences(first,second):
    changed=[]
    for scope,a,b in [('model',first,second),('option',first.opt,second.opt),('stat',first.stat,second.stat)]:
        for name in sorted(n for n in dir(a) if not n.startswith('_')):
            x,y=getattr(a,name),getattr(b,name)
            if isinstance(x,np.ndarray):
                if x.dtype!=y.dtype or x.shape!=y.shape or x.tobytes()!=y.tobytes():changed.append(scope+'.'+name)
            elif isinstance(x,(int,float,str,bytes,bool)) and x!=y:changed.append(scope+'.'+name)
    return changed


def enable_pad_spin(model,parts,pads,spin_m):
    """Edit only known explicit pad/object pairs of a fresh native model."""
    if not np.isfinite(spin_m) or not 0<spin_m<=.003:raise ValueError('Declared exploratory spin length in(0,.003]metres')
    parts,pads=set(map(int,parts)),set(map(int,pads));changed=[]
    for i,(a,b) in enumerate(zip(model.pair_geom1,model.pair_geom2)):
        if (int(a) in parts and int(b) in pads) or (int(b) in parts and int(a) in pads):
            if model.pair_dim[i]!=3:raise ValueError('Fresh condim3 baseline required')
            model.pair_dim[i]=4;model.pair_friction[i,2]=spin_m;changed.append(i)
    if len(changed)!=len(parts)*len(pads):raise ValueError('Missing or duplicated explicit pad/object pair')
    return changed


def local_pyramid_wrenches(dim,friction):
    """Local[force3,couple3] unit-normal rays; tangents then spin/rolling."""
    if dim not in (3,4,6):raise ValueError('Supported frictional contact dimensions3/4/6')
    mu=np.asarray(friction,float)
    if mu.shape!=(5,) or not np.isfinite(mu).all() or np.any(mu[:dim-1]<=0):raise ValueError('Positive active friction dimensions')
    rays=np.zeros((2*(dim-1),6));rays[:,0]=1.
    for j in range(dim-1):rays[2*j,1+j]=mu[j];rays[2*j+1,1+j]=-mu[j]
    return rays


def contact_wrench_basis(model,data,parts,body,allowed,*,free_dof=20,hand_dofs=tuple(range(20))):
    """Manual point/angular Jacobian basis checked against native EFC columns.

    Read-only unit EFC probes are never written into constraint force arrays.
    All moment columns are about moving COM; free rotation entries use BODY axes.
    """
    import mujoco
    if model.opt.cone!=mujoco.mjtCone.mjCONE_PYRAMIDAL:raise ValueError('Pyramidal model required')
    parts,allowed=set(map(int,parts)),set(map(int,allowed));hand_dofs=np.asarray(hand_dofs,dtype=int)
    if model.body_parentid[body]!=0 or model.jnt_type[model.body_jntadr[body]]!=mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Direct world free object required for native root wrench readback')
    R=data.xmat[body].reshape(3,3);COM=data.xipos[body];root=data.xpos[body]
    W=[];L=[];nativeW=[];nativeL=[];records=[];actual=np.zeros(len(hand_dofs))
    size=mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION);before=np.zeros(size);after=np.zeros(size)
    mujoco.mj_getState(model,data,before,mujoco.mjtState.mjSTATE_INTEGRATION)
    for ci in range(data.ncon):
        con=data.contact[ci];a,b=map(int,con.geom);oa,ob=a in parts,b in parts
        if oa==ob or con.efc_address<0:continue
        other=a if ob else b
        if other not in allowed:raise ValueError('Nonintent/table contact cannot silently support object')
        sign=1. if ob else -1.;frame=con.frame.reshape(3,3)
        jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(model,data,jp,jr,con.pos.copy(),int(model.geom_bodyid[other]))
        local=np.zeros(6);mujoco.mj_contactForce(model,data,ci,local)
        actual-=jp[:,hand_dofs].T@(sign*frame.T@local[:3])+jr[:,hand_dofs].T@(sign*frame.T@local[3:])
        rays=local_pyramid_wrenches(int(con.dim),con.friction)
        for j,ray in enumerate(rays):
            F=sign*(frame.T@ray[:3]);T=sign*(frame.T@ray[3:])
            W.append(np.r_[F,T+np.cross(con.pos-COM,F)])
            L.append(jp[:,hand_dofs].T@F+jr[:,hand_dofs].T@T)
            coeff=np.zeros(data.nefc);coeff[con.efc_address+j]=1.;generalized=np.zeros(model.nv)
            mujoco.mj_mulJacTVec(model,data,generalized,coeff)
            nf=generalized[free_dof:free_dof+3];nt=R@generalized[free_dof+3:free_dof+6]-np.cross(COM-root,nf)
            nativeW.append(np.r_[nf,nt]);nativeL.append(-generalized[hand_dofs])
        records.append(dict(contact_index=ci,geom_ids=[a,b],other_geom=other,dim=int(con.dim),efc_address=int(con.efc_address),
            position_world_m=con.pos.copy(),frame_world_rows=frame.copy(),friction5=con.friction.copy(),local_force6=local))
    W=np.array(W).reshape(-1,6).T;L=np.array(L).reshape(len(W.T),len(hand_dofs)).T
    nw=np.array(nativeW).reshape(-1,6).T;nl=np.array(nativeL).reshape(len(W.T),len(hand_dofs)).T
    werror=float(np.abs(W-nw).max()) if W.size else 0.;lerror=float(np.abs(L-nl).max()) if L.size else 0.
    mujoco.mj_getState(model,data,after,mujoco.mjtState.mjSTATE_INTEGRATION)
    if before.tobytes()!=after.tobytes() or werror>1e-10 or lerror>1e-10:raise ValueError('Native EFC/point-angular basis or readonly state check failed')
    return dict(wrench=W,loads=L,native_wrench=nw,native_loads=nl,contacts=records,
        actual_hand_contact_torque=actual,other_constraint_torque=data.qfrc_constraint[hand_dofs]-actual,
        native_wrench_error=werror,native_load_error=lerror,native_EFC_unit_queries=W.shape[1],readonly=True)
