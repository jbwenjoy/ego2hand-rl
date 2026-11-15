"""One-world resident finite-action native oracle; live Data stays read-only."""
import gc
import time

import numpy as np
import torch
import warp as wp

from .gpu_data_fork import fork_dynamic_data
from .gpu_data_snapshot import ResidentDataSnapshot
from .gpu_model_snapshot import model_leaves
from .pen_native_planning_math import score_candidates,score_sequence
from ..mechanics.pen_contact_groups import normal_group


@wp.kernel
def _clear(slot:wp.array(dtype=int),pen:wp.array2d(dtype=float)):
    j=wp.tid();pen[slot[0],j]=0.0


@wp.kernel
def _set_ctrl(slot:wp.array(dtype=int),targets:wp.array2d(dtype=float),ctrl:wp.array2d(dtype=float)):
    j=wp.tid();ctrl[0,j]=targets[slot[0],j]


@wp.kernel
def _set_scheduled_ctrl(slot:wp.array(dtype=int),phase:int,targets:wp.array3d(dtype=float),ctrl:wp.array2d(dtype=float)):
    j=wp.tid();ctrl[0,j]=targets[slot[0],phase,j]


@wp.kernel
def _record(slot:wp.array(dtype=int),tick:int,q:wp.array2d(dtype=float),v:wp.array2d(dtype=float),ctrl:wp.array2d(dtype=float),
            force:wp.array2d(dtype=float),R:wp.array2d(dtype=wp.mat33),time:wp.array(dtype=float),
            nacon:wp.array(dtype=int),nefc:wp.array(dtype=int),overflow:wp.array(dtype=int),nconmax:int,njmax:int,body:int,da:int,
            qids:wp.array(dtype=int),lower:wp.array(dtype=float),upper:wp.array(dtype=float),
            Q:wp.array3d(dtype=float),V:wp.array3d(dtype=float),U:wp.array3d(dtype=float),W:wp.array3d(dtype=float),
            T:wp.array2d(dtype=float),pen:wp.array2d(dtype=float),other:wp.array2d(dtype=float),bad:wp.array2d(dtype=int)):
    j=wp.tid();s=slot[0]
    if j<q.shape[1]:Q[s,tick,j]=q[0,j]
    if j<v.shape[1]:V[s,tick,j]=v[0,j]
    if j<ctrl.shape[1]:U[s,tick,j]=ctrl[0,j]
    if j<3:
        torque=R[0,body]*wp.vec3(force[0,da+3],force[0,da+4],force[0,da+5])
        W[s,tick,j]=force[0,da+j];W[s,tick,j+3]=torque[j]
    if j<20:
        value=q[0,qids[j]]
        wp.atomic_max(pen,s,2,wp.max(0.0,wp.max(lower[j]-value,value-upper[j])))
    if j==0:
        T[s,tick]=time[0];other[s,tick]=0.0
        flag=int(0)
        if nacon[0]>nconmax or nefc[0]>njmax or overflow[0]!=0:flag=1
        bad[s,tick]=flag


@wp.kernel
def _contacts(slot:wp.array(dtype=int),tick:int,nacon:wp.array(dtype=int),geom:wp.array(dtype=wp.vec2i),
              dist:wp.array(dtype=float),dim:wp.array(dtype=int),address:wp.array2d(dtype=int),
              force:wp.array2d(dtype=float),nefc:wp.array(dtype=int),tags:wp.array(dtype=int),
              pen:wp.array2d(dtype=float),other:wp.array2d(dtype=float),bad:wp.array2d(dtype=int)):
    i=wp.tid();s=slot[0]
    if i<nacon[0]:
        g=geom[i][0];h=geom[i][1]
        if g<0 or h<0 or g>=tags.shape[0] or h>=tags.shape[0]:
            wp.atomic_max(bad,s,tick,1)
        else:
            depth=wp.max(0.0,-dist[i]);wp.atomic_max(pen,s,0,depth)
            if tags[g]==1 and tags[h]==1:wp.atomic_max(pen,s,1,depth)
            if (tags[g]==2 and tags[h]==0) or (tags[h]==2 and tags[g]==0):
                normal=float(0.0)
                if address[i,0]>=0:
                    count=int(1)
                    if dim[i]>1:count=2*(dim[i]-1)
                    for edge in range(count):
                        index=address[i,edge]
                        if index>=0 and index<force.shape[1] and index<nefc[0]:normal+=force[0,index]
                        else:wp.atomic_max(bad,s,tick,1)
                wp.atomic_add(other,s,tick,wp.max(0.0,normal))


@wp.kernel
def _clear_response(slot:wp.array(dtype=int),tick:int,Fn:wp.array3d(dtype=float)):
    j=wp.tid();Fn[slot[0],tick,j]=0.0


@wp.kernel
def _response_heads(slot:wp.array(dtype=int),tick:int,warm:wp.array2d(dtype=float),nacon:wp.array(dtype=int),
                    geom:wp.array(dtype=wp.vec2i),dim:wp.array(dtype=int),address:wp.array2d(dtype=int),force:wp.array2d(dtype=float),
                    nefc:wp.array(dtype=int),tags:wp.array(dtype=int),groups:wp.array(dtype=int),
                    Warm:wp.array3d(dtype=float),Fn:wp.array3d(dtype=float),bad:wp.array2d(dtype=int)):
    i=wp.tid();s=slot[0]
    if i<warm.shape[1]:Warm[s,tick,i]=warm[0,i]
    if i<nacon[0]:
        g=geom[i][0];h=geom[i][1]
        if g<0 or h<0 or g>=tags.shape[0] or h>=tags.shape[0]:wp.atomic_max(bad,s,tick,1)
        else:
            other=int(-1)
            if tags[g]==2 and tags[h]==1:other=h
            elif tags[h]==2 and tags[g]==1:other=g
            if other>=0:
                group=groups[other];normal=float(0.0)
                if group<0 or group>5:wp.atomic_max(bad,s,tick,1)
                elif address[i,0]>=0:
                    count=int(1)
                    if dim[i]>1:count=2*(dim[i]-1)
                    for edge in range(count):
                        index=address[i,edge]
                        if index>=0 and index<nefc[0] and index<force.shape[1]:normal+=force[0,index]
                        else:wp.atomic_max(bad,s,tick,1)
                    wp.atomic_add(Fn,s,tick,group,wp.max(0.0,normal))


class NativePenPlanner:
    def __init__(self,model,device_model,live,qids,aids,nsteps,*,response_heads=False,horizon_controls=1):
        import mujoco
        import mujoco_warp as mw
        if (live.nworld!=1 or model.na or model.nhistory or model.nplugin or model.neq
                or int(model.opt.cone)!=0 or model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)):
            raise ValueError('One-world no-hidden-state/no-equality pyramidal controlled plant')
        if (isinstance(horizon_controls,bool) or horizon_controls not in range(1,6) or nsteps<1 or nsteps%horizon_controls or
                abs(float(model.opt.timestep)*(nsteps//horizon_controls)-.02)>1e-9 or response_heads and horizon_controls!=1):
            raise ValueError('Whole20ms controls, horizon1..5, response heads only H1')
        self.horizon_controls=horizon_controls;self.control_substeps=nsteps//horizon_controls
        self.model,self.live,self.nsteps=device_model,live,nsteps;self.host_model=model;self.aids=aids;self.response_heads=response_heads
        self.body=model.body('object/cube').id;joint=model.body_jntadr[self.body]
        self.qa=int(model.jnt_qposadr[joint]);self.da=int(model.jnt_dofadr[joint])
        if np.any(device_model.body_ipos.numpy()[0,self.body]):raise ValueError('COM at free-body origin')
        self.torch_stream=torch.cuda.Stream();self.stream=wp.stream_from_torch(self.torch_stream)
        live_arrays,self.live_scalars=model_leaves(live)
        self.live_layout={k:(a.ptr,a.shape,a.dtype,a.strides) for k,a in live_arrays.items()}
        with wp.ScopedStream(self.stream):
            self.data,self.layout=fork_dynamic_data(live);self.source=ResidentDataSnapshot(self.data)
            self.slot=wp.zeros(1,dtype=int,device='cuda');self.target_ctrl=wp.zeros((57,model.nu),device='cuda')
            if horizon_controls>1:self.schedule_ctrl=wp.zeros((57,horizon_controls,model.nu),device='cuda')
            self.Q=wp.zeros((57,nsteps,model.nq),device='cuda');self.V=wp.zeros((57,nsteps,model.nv),device='cuda')
            self.U=wp.zeros((57,nsteps,model.nu),device='cuda');self.W=wp.zeros((57,nsteps,6),device='cuda')
            self.T=wp.zeros((57,nsteps),device='cuda');self.pen=wp.zeros((57,3),device='cuda')
            self.other=wp.zeros((57,nsteps),device='cuda');self.bad=wp.zeros((57,nsteps),dtype=int,device='cuda')
            qids=np.asarray(qids,dtype=np.int32);jids=model.actuator_trnid[aids,0]
            self.qids=wp.array(qids,device='cuda');ranges=device_model.jnt_range.numpy()[0,jids]
            self.lower=wp.array(ranges[:,0].astype(np.float32),device='cuda');self.upper=wp.array(ranges[:,1].astype(np.float32),device='cuda')
            tags=np.array([2 if model.geom_bodyid[g]==self.body else 1 if (model.geom(g).name or '').startswith('robot/') else 0 for g in range(model.ngeom)],np.int32)
            self.tags=wp.array(tags,device='cuda')
            if response_heads:
                self.Warm=wp.zeros((57,nsteps,model.nv),device='cuda');self.Fn=wp.zeros((57,nsteps,6),device='cuda')
                groups=np.array([normal_group(model.geom(g).name or '') for g in range(model.ngeom)],np.int32)
                self.normal_groups=wp.array(groups,device='cuda')
            wp.load_module(module=__name__,device='cuda')
            was_gc=gc.isenabled();gc.disable()
            try:
                with wp.ScopedCapture(stream=self.stream) as capture:
                    self.source.restore(self.data)
                    wp.launch(_clear,3,inputs=[self.slot,self.pen],stream=self.stream)
                    if horizon_controls==1:
                        wp.launch(_set_ctrl,model.nu,inputs=[self.slot,self.target_ctrl,self.data.ctrl],stream=self.stream)
                    for tick in range(nsteps):
                        if horizon_controls>1 and tick%self.control_substeps==0:
                            wp.launch(_set_scheduled_ctrl,model.nu,inputs=[self.slot,tick//self.control_substeps,self.schedule_ctrl,self.data.ctrl],stream=self.stream)
                        mw.step(device_model,self.data)
                        wp.launch(_record,max(model.nq,model.nv,model.nu,20),inputs=[self.slot,tick,self.data.qpos,self.data.qvel,self.data.ctrl,
                            self.data.qfrc_constraint,self.data.xmat,self.data.time,self.data.nacon,self.data.nefc,self.data.overflow,
                            self.data.naconmax,self.data.njmax,self.body,self.da,self.qids,self.lower,self.upper,
                            self.Q,self.V,self.U,self.W,self.T,self.pen,self.other,self.bad],stream=self.stream)
                        wp.launch(_contacts,self.data.naconmax,inputs=[self.slot,tick,self.data.nacon,self.data.contact.geom,self.data.contact.dist,
                            self.data.contact.dim,self.data.contact.efc_address,self.data.efc.force,self.data.nefc,self.tags,self.pen,self.other,self.bad],stream=self.stream)
                        if response_heads:
                            wp.launch(_clear_response,6,inputs=[self.slot,tick,self.Fn],stream=self.stream)
                            wp.launch(_response_heads,max(self.data.naconmax,model.nv),inputs=[self.slot,tick,self.data.qacc_warmstart,self.data.nacon,self.data.contact.geom,
                                self.data.contact.dim,self.data.contact.efc_address,self.data.efc.force,self.data.nefc,self.tags,self.normal_groups,self.Warm,self.Fn,self.bad],stream=self.stream)
                self.graph=capture.graph
            finally:
                if was_gc:gc.enable()
        self.views={k:wp.to_torch(getattr(self,k)) for k in ['slot','target_ctrl','Q','V','U','W','T','pen','other','bad']}
        if horizon_controls>1:self.views['schedule_ctrl']=wp.to_torch(self.schedule_ctrl)
        if response_heads:self.views.update({k:wp.to_torch(getattr(self,k)) for k in ['Warm','Fn']})
        self.qid_tensor=torch.as_tensor(qids.astype(np.int64),device='cuda');self.ctrl_tensor=wp.to_torch(live.ctrl)
        self.aid_tensor=torch.as_tensor(np.asarray(aids,dtype=np.int64),device='cuda')
        wp.synchronize_stream(self.stream)

    def load_source(self):
        self.torch_stream.wait_stream(torch.cuda.current_stream())
        arrays,scalars=model_leaves(self.live)
        if scalars!=self.live_scalars or set(arrays)!=set(self.live_layout):raise ValueError('Live Data layout drift')
        for key,a in arrays.items():
            if (a.ptr,a.shape,a.dtype,a.strides)!=self.live_layout[key]:raise ValueError('Live Data allocation drift')
        with wp.ScopedStream(self.stream):
            for key,a in arrays.items():
                if a.size:wp.copy(self.source.arrays[key],a)

    def search(self,targets,ema,reference_COM,reference_shaft,reference_hand,target_wrench,wrench_valid,support_weight,cfg,
               *,reference_velocity=None,velocity_valid=False,velocity_weight=0.,reference_quaternion=None,reference_sequence=None):
        wall_started=time.perf_counter()
        caller=torch.cuda.current_stream();self.torch_stream.wait_stream(caller)
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.torch_stream),wp.ScopedStream(self.stream):
            if self.horizon_controls==1:
                self.views['target_ctrl'].copy_(self.ctrl_tensor.expand(57,-1))
                self.views['target_ctrl'][:,self.aid_tensor]=targets
            else:
                if reference_sequence is None or velocity_weight:raise ValueError('Explicit scheduled state/support reference')
                self.views['schedule_ctrl'].copy_(self.ctrl_tensor.expand(57,self.horizon_controls,-1))
                self.views['schedule_ctrl'][:,:,self.aid_tensor]=targets
            begin.record()
            for i in range(57):
                self.views['slot'].fill_(i);wp.capture_launch(self.graph,stream=self.stream)
            if self.horizon_controls==1:
                result=score_candidates(self.views['Q'],self.views['V'],self.views['U'],self.views['W'],self.views['pen'],self.views['other'],self.views['bad'],ema,
                    self.qid_tensor,self.qa,reference_COM,reference_shaft,reference_hand,target_wrench,wrench_valid,support_weight,cfg,
                    object_da=self.da,reference_velocity=reference_velocity,velocity_valid=velocity_valid,velocity_weight=velocity_weight,
                    reference_quaternion=reference_quaternion)
            else:
                result=score_sequence(self.views['Q'],self.views['V'],self.views['U'],self.views['W'],self.views['pen'],self.views['other'],self.views['bad'],
                    ema,self.qid_tensor,self.qa,reference_sequence,support_weight,cfg)
            end.record()
        end.synchronize();caller.wait_stream(self.torch_stream)
        readback_started=time.perf_counter()
        index=int(result['index']);arrays={k:v.detach().cpu().numpy().copy() for k,v in self.views.items() if k not in ['slot','target_ctrl','schedule_ctrl']}
        scores={k:v.detach().cpu().numpy().copy() for k,v in result.items()}
        cuda_ms=float(begin.elapsed_time(end));ended=time.perf_counter()
        self.last_search_timing=dict(candidate_cuda_ms=cuda_ms,submit_and_wait_wall_s=readback_started-wall_started,
            readback_and_host_copy_wall_s=ended-readback_started,total_search_wall_s=ended-wall_started,
            scope='Boundary timing only; no new synchronization or per-candidate host checks. Setup/compile and artifact I/O are outside search.')
        return index,arrays,scores,cuda_ms
