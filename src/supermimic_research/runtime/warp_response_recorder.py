"""Held absolute-PD native graph responses with saved independent EFC receipts."""
import numpy as np

from ..mechanics.warp_contact_labels import read_warp_contact_labels


def record_response(model,device_model,data,graph,aids,target20,nsteps,*,instrument=False):
    import mujoco
    import mujoco_warp as mw
    import warp as wp
    host=mujoco.MjData(model);body=model.body('object/cube').id
    qa=int(model.jnt_qposadr[model.body_jntadr[body]])
    ctrl=data.ctrl.numpy().copy();ctrl[0,aids]=target20;data.ctrl.assign(ctrl)
    mw.get_data_into(host,model,data)
    signature=mujoco.mjtState.mjSTATE_INTEGRATION;initial=np.empty(mujoco.mj_stateSize(model,signature));mujoco.mj_getState(model,host,initial,signature)
    trace={key:[] for key in ['pre_qpos','pre_qvel','post_qpos','post_qvel','post_ctrl','post_qacc','post_qacc_warmstart',
        'pre_object_COM','post_qfrc_constraint','post_qfrc_applied','post_xfrc_applied','post_GPU_time','overflow','post_nefc',
        'post_mocap_pos','post_mocap_quat']}
    if instrument:
        trace.update({key:[] for key in ['pre_qacc_warmstart','pre_solver_niter','post_solver_niter','pre_nacon','pre_nefc',
            'post_M','post_qacc_smooth','post_efc_Ma','post_qfrc_smooth','post_qfrc_bias','post_qfrc_passive','post_qfrc_actuator']})
    raw=[];addresses=[];efc=[];ranges=[0];legacy=[];adhesion=[]
    for tick in range(nsteps):
        preq,prev=data.qpos.numpy()[0].copy(),data.qvel.numpy()[0].copy()
        if instrument:
            pre_instrument=dict(pre_qacc_warmstart=data.qacc_warmstart.numpy()[0].copy(),
                pre_solver_niter=int(data.solver_niter.numpy()[0]),pre_nacon=int(data.nacon.numpy()[0]),pre_nefc=int(data.nefc.numpy()[0]))
        wp.capture_launch(graph);mw.get_data_into(host,model,data);labels=read_warp_contact_labels(device_model,data)
        if not np.array_equal(labels['geom'],host.contact.geom):raise ValueError('Fork host/GPU contact order mismatch')
        for i,(g,h) in enumerate(labels['geom']):
            c=host.contact[i];old=np.zeros(6);mujoco.mj_contactForce(model,host,i,old)
            raw.append(np.r_[tick,g,h,labels['dim'][i],labels['efc_address'][i,0],c.dist,c.pos,c.frame,labels['local_force6'][i]])
            addresses.append(labels['efc_address'][i]);legacy.append(old);adhesion.append(float(c.adhesion))
        efc.append(labels['efc_force']);ranges.append(ranges[-1]+len(labels['efc_force']))
        values=dict(pre_qpos=preq,pre_qvel=prev,post_qpos=host.qpos.copy(),post_qvel=host.qvel.copy(),post_ctrl=host.ctrl.copy(),
            post_qacc=host.qacc.copy(),post_qacc_warmstart=data.qacc_warmstart.numpy()[0].copy(),pre_object_COM=host.xipos[body].copy(),
            post_qfrc_constraint=host.qfrc_constraint.copy(),post_qfrc_applied=host.qfrc_applied.copy(),post_xfrc_applied=host.xfrc_applied.copy(),
            post_GPU_time=float(host.time),overflow=bool(labels['overflow'] or data.overflow.numpy()[0]),post_nefc=labels['nefc'],
            post_mocap_pos=host.mocap_pos.copy(),post_mocap_quat=host.mocap_quat.copy())
        if instrument:
            values.update(pre_instrument,post_solver_niter=int(data.solver_niter.numpy()[0]),post_M=data.M.numpy()[0].copy(),
                post_qacc_smooth=data.qacc_smooth.numpy()[0].copy(),post_efc_Ma=data.efc.Ma.numpy()[0].copy(),
                **{'post_'+key:getattr(data,key).numpy()[0].copy() for key in ['qfrc_smooth','qfrc_bias','qfrc_passive','qfrc_actuator']})
        for key,value in values.items():trace[key].append(value)
    return {**{key:np.array(values) for key,values in trace.items()},
        'initial_native_state':initial,'raw_contact_tick_plus23':np.array(raw).reshape(-1,24),
        'GPU_contact_efc_addresses':np.array(addresses).reshape(-1,data.contact.efc_address.shape[-1]),'GPU_efc_force_flat':np.concatenate(efc),
        'GPU_efc_ranges':np.array(ranges),'legacy_host_local_force6':np.array(legacy).reshape(-1,6),'legacy_host_adhesion':np.array(adhesion)}
