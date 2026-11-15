"""Independent native EFC packet decoding, shared by file and streaming audits."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def decode_saved_case(c,z,m,p):
    raw=z['raw_contact_tick_plus23'];addresses=z['GPU_contact_efc_addresses'];ranges=z['GPU_efc_ranges'];flat=z['GPU_efc_force_flat'];N=len(z['post_qpos']);ticks=raw[:,0].astype(int)
    if m.npair:raise ValueError('Explicit pair friction requires separate captured pair metadata')
    if len(ranges)!=N+1 or ranges[0]!=0 or ranges[-1]!=len(flat) or len(addresses)!=len(raw):raise ValueError('EFC/contact receipt ranges')
    local=np.zeros((len(raw),6));valid=True
    for i,row in enumerate(raw):
        tick=int(row[0]);dim=int(row[3]);count=1 if dim==1 else (2*(dim-1) if int(m.opt.cone)==0 else dim);ids=addresses[i,:count];nefc=ranges[tick+1]-ranges[tick]
        if row[4]!=addresses[i,0]:raise ValueError('Source GPU address relabeled')
        if ids[0]<0:continue
        if np.any((ids<0)|(ids>=nefc)):valid=False;continue
        edges=flat[ranges[tick]+ids].astype(np.float64)
        if int(m.opt.cone)==0 and dim>1:
            # Independently sum each edge pair and its signed tangent.
            for j in range(dim-1):
                plus,minus=edges[2*j:2*j+2];local[i,0]+=plus+minus
            # These tasks have no explicit pairs or interval material DR.
            # Reconstruct contact friction from saved actual GPU geom tables
            # and native priority/max rules, independently of reader output.
            g,h=map(int,row[1:3]);fg,fh=p['geom_friction'][0,[g,h]]
            friction=fg if m.geom_priority[g]>m.geom_priority[h] else fh if m.geom_priority[h]>m.geom_priority[g] else np.maximum(fg,fh)
            mu=friction[[0,0,1,2,2]]
            for j in range(dim-1):local[i,j+1]=(edges[2*j]-edges[2*j+1])*mu[j]
        else:local[i,:dim]=edges
    label_error=float(abs(local-raw[:,18:24]).max()) if len(raw) else 0.
    body=c['model_layout']['object_body'];qa=c['model_layout']['object_qposadr'];da=c['model_layout']['object_dofadr'];dt=c['dt'];mass=float(p['body_mass'][0,body])
    g1,g2=raw[:,1].astype(int),raw[:,2].astype(int);o1,o2=m.geom_bodyid[g1]==body,m.geom_bodyid[g2]==body;mask=o1^o2
    row=raw[mask];ti=ticks[mask];sign=np.where(o2[mask],1.,-1.);frame=row[:,9:18].reshape(-1,3,3)
    F=sign[:,None]*np.einsum('nji,nj->ni',frame,local[mask,:3]);T=sign[:,None]*np.einsum('nji,nj->ni',frame,local[mask,3:])+np.cross(row[:,6:9]-z['pre_object_COM'][ti],F)
    world=np.zeros((N,6));np.add.at(world,ti,np.c_[F,T])
    R=Rotation.from_quat(z['pre_qpos'][:,qa+3:qa+7][:,[1,2,3,0]]).as_matrix()
    BODY=np.c_[np.einsum('nji,nj->ni',R,world[:,:3]),np.einsum('nji,nj->ni',R,world[:,3:])]
    # Both controlled objects have COM at their free-body origin.
    if np.any(np.array(c['model_layout']['COM_BODY'])!=0):raise ValueError('Explicit zero-COM free-body force identity')
    force_error=world[:,:3]-z['post_qfrc_constraint'][:,da:da+3]
    torque_error=BODY[:,3:]-z['post_qfrc_constraint'][:,da+3:da+6]
    dv=z['post_qvel'][:,da:da+3]-z['pre_qvel'][:,da:da+3]
    momentum=mass*dv-dt*(world[:,:3]+mass*m.opt.gravity)
    legacy=z['legacy_host_local_force6'];adhesion=z['legacy_host_adhesion'];finite=np.isfinite(legacy).all(axis=1)&np.isfinite(adhesion)
    legacy_error=np.full(len(raw),np.nan);legacy_error[finite]=legacy[finite,0]-local[finite,0]+adhesion[finite]
    old_difference=np.max(abs(legacy[finite]-local[finite]),axis=1) if finite.any() else np.zeros(0)
    max_force=float(np.linalg.norm(force_error,axis=1).max());max_torque=float(np.linalg.norm(torque_error,axis=1).max());max_impulse=float(np.linalg.norm(momentum,axis=1).max())
    checks=dict(valid_source_indices=valid,local_labels_exact=label_error<1e-10,
        generalized_force=max_force<1e-4,generalized_torque=max_torque<1e-5,linear_impulse=max_impulse<5e-7,
        copied_host_difference_explained=bool(np.nanmax(abs(legacy_error),initial=0.)<1e-9),
        all_joint_actuators=bool(np.all(m.actuator_trntype==int(mujoco.mjtTrn.mjTRN_JOINT))))

    return dict(checks=checks,raw_contacts=len(raw),local_label_max_error=label_error,max_force_residual_N=max_force,max_torque_residual_Nm=max_torque,
        max_linear_impulse_residual_N_s=max_impulse,changed_host_contact_rows=int(np.count_nonzero(old_difference>1e-7)),
        nonfinite_host_rows=int(np.count_nonzero(~finite)),max_legacy_normal_plus_adhesion_error=float(np.nanmax(abs(legacy_error),initial=0.))),dict(
        world_COM_wrench=world,object_BODY_COM_wrench=BODY,force_residual_N=force_error,torque_residual_Nm=torque_error,linear_impulse_residual_N_s=momentum)
