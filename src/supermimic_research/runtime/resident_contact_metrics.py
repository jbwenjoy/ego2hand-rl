"""Read-only resident contact/geometry metrics with explicit origins and frames.

Native contacts/xipos/xmat belong to the latest forward (pre-integration when
sampled immediately after step); joint excursion uses post-integration qpos.
No host contact-count read, geometry rebuild, physics write or force application.
"""
import numpy as np
import warp as wp
from supermimic_research.mechanics.pen_contact_groups import normal_group

vec5=wp.types.vector(length=5,dtype=wp.float32)
vec6d=wp.types.vector(length=6,dtype=wp.float64)
SCHEMA='resident_native_contacts_per_world:1'
BAD_WORLD=wp.constant(1);BAD_GEOM=wp.constant(2);BAD_DIM=wp.constant(4);BAD_EFC=wp.constant(8)
NONFINITE=wp.constant(16);CAPACITY=wp.constant(32);NEGATIVE_NORMAL=wp.constant(64);OVERFLOW=wp.constant(128)
BAD_LIMIT=wp.constant(256);BAD_ROTATION=wp.constant(512)
BAD_RESET_ID=wp.constant(1024)


@wp.kernel
def _clear_instant(geom:wp.array2d(dtype=wp.float64),fn:wp.array2d(dtype=wp.float64),other:wp.array(dtype=wp.float64),
    wrench:wp.array3d(dtype=wp.float64),errors:wp.array(dtype=int),global_error:wp.array(dtype=int),counts:wp.array(dtype=int)):
    w=wp.tid()
    for k in range(3):geom[w,k]=wp.float64(0.0)
    for k in range(6):
        fn[w,k]=wp.float64(0.0)
        for j in range(4):wrench[w,j,k]=wp.float64(0.0)
    other[w]=wp.float64(0.0);errors[w]=0;counts[w]=0
    if w==0:global_error[0]=0


@wp.kernel
def _contact_pass(nacon:wp.array(dtype=int),world:wp.array(dtype=int),geom:wp.array(dtype=wp.vec2i),
    distance:wp.array(dtype=float),dimension:wp.array(dtype=int),position:wp.array(dtype=wp.vec3),frame:wp.array(dtype=wp.mat33),
    friction:wp.array(dtype=vec5),address:wp.array2d(dtype=int),force:wp.array2d(dtype=float),nefc:wp.array(dtype=int),
    object_tag:wp.array(dtype=int),digit:wp.array(dtype=int),com:wp.array2d(dtype=wp.vec3),body:int,cone:int,
    output_geom:wp.array2d(dtype=wp.float64),fn:wp.array2d(dtype=wp.float64),other:wp.array(dtype=wp.float64),
    wrench:wp.array3d(dtype=wp.float64),errors:wp.array(dtype=int),global_error:wp.array(dtype=int),counts:wp.array(dtype=int)):
    i=wp.tid()
    if i<nacon[0]:
        w=world[i]
        if w<0 or w>=output_geom.shape[0]:
            wp.atomic_or(global_error,0,BAD_WORLD)
        else:
            g=geom[i][0];h=geom[i][1]
            if g<0 or h<0 or g>=object_tag.shape[0] or h>=object_tag.shape[0]:
                wp.atomic_or(errors,w,BAD_GEOM)
            else:
                wp.atomic_add(counts,w,1)
                finite=wp.isfinite(distance[i])
                for j in range(3):
                    finite=finite and wp.isfinite(position[i][j])
                    for k in range(3):finite=finite and wp.isfinite(frame[i][j,k])
                for j in range(5):finite=finite and wp.isfinite(friction[i][j])
                if not finite:
                    wp.atomic_or(errors,w,NONFINITE)
                else:
                    og=object_tag[g]!=0;oh=object_tag[h]!=0
                    depth=wp.max(wp.float64(0.0),-wp.float64(distance[i]))
                    if og!=oh:wp.atomic_max(output_geom,w,0,depth)
                    if digit[g]>=0 and digit[h]>=0:wp.atomic_max(output_geom,w,1,depth)
                    dim=dimension[i];supported=dim==1 or dim==3 or dim==4 or dim==6
                    local=vec6d();valid=bool(True)
                    if not supported:
                        wp.atomic_or(errors,w,BAD_DIM);valid=False
                    elif address[i,0]>=0:
                        width=dim
                        if cone==0 and dim>1:width=2*(dim-1)
                        if width>address.shape[1]:
                            wp.atomic_or(errors,w,BAD_EFC);valid=False
                        else:
                            for j in range(width):
                                index=address[i,j]
                                if index<0 or index>=nefc[w] or index>=force.shape[1]:
                                    wp.atomic_or(errors,w,BAD_EFC);valid=False
                                elif not wp.isfinite(force[w,index]):
                                    wp.atomic_or(errors,w,NONFINITE);valid=False
                            if valid:
                                if cone==0 and dim>1:
                                    for j in range(width):local[0]=local[0]+wp.float64(force[w,address[i,j]])
                                    for j in range(dim-1):
                                        local[j+1]=(wp.float64(force[w,address[i,2*j]])-wp.float64(force[w,address[i,2*j+1]]))*wp.float64(friction[i][j])
                                else:
                                    for j in range(dim):local[j]=wp.float64(force[w,address[i,j]])
                    if valid:
                        if local[0]<wp.float64(0.0):wp.atomic_or(errors,w,NEGATIVE_NORMAL)
                        if og!=oh:
                            sign=wp.float64(1.0);tag=digit[g]
                            if og:sign=wp.float64(-1.0);tag=digit[h]
                            normal=wp.max(local[0],wp.float64(0.0))
                            if tag>=0:wp.atomic_add(fn,w,tag,normal)
                            else:wp.atomic_add(other,w,normal)
                            F=wp.vec3d();M=wp.vec3d();arm=wp.vec3d()
                            for j in range(3):
                                arm[j]=wp.float64(position[i][j])-wp.float64(com[w,body][j])
                                for k in range(3):
                                    F[j]=F[j]+sign*wp.float64(frame[i][k,j])*local[k]
                                    M[j]=M[j]+sign*wp.float64(frame[i][k,j])*local[k+3]
                            M=M+wp.cross(arm,F)
                            for j in range(3):
                                wp.atomic_add(wrench,w,0,j,F[j]);wp.atomic_add(wrench,w,0,j+3,M[j])
                                if tag>=0:
                                    wp.atomic_add(wrench,w,1,j,F[j]);wp.atomic_add(wrench,w,1,j+3,M[j])


@wp.kernel
def _world_pass(qpos:wp.array2d(dtype=float),joint_ids:wp.array(dtype=int),qids:wp.array(dtype=int),
    limits:wp.array2d(dtype=wp.vec2),R:wp.array2d(dtype=wp.mat33),com:wp.array2d(dtype=wp.vec3),body:int,
    nacon:wp.array(dtype=int),capacity:int,nefc:wp.array(dtype=int),njmax:int,overflow:wp.array(dtype=int),
    output_geom:wp.array2d(dtype=wp.float64),wrench:wp.array3d(dtype=wp.float64),errors:wp.array(dtype=int),global_error:wp.array(dtype=int)):
    w=wp.tid()
    if w==0 and (nacon[0]<0 or nacon[0]>capacity):wp.atomic_or(global_error,0,CAPACITY)
    if nefc[w]<0 or nefc[w]>njmax:wp.atomic_or(errors,w,CAPACITY)
    if overflow[w%overflow.shape[0]]!=0:wp.atomic_or(errors,w,OVERFLOW)
    for j in range(qpos.shape[1]):
        if not wp.isfinite(qpos[w,j]):wp.atomic_or(errors,w,NONFINITE)
    for j in range(qids.shape[0]):
        limit=limits[w%limits.shape[0],joint_ids[j]];value=qpos[w,qids[j]]
        if not wp.isfinite(limit[0]) or not wp.isfinite(limit[1]) or limit[0]>limit[1]:wp.atomic_or(errors,w,BAD_LIMIT)
        else:wp.atomic_max(output_geom,w,2,wp.max(wp.float64(0.0),wp.max(wp.float64(limit[0])-wp.float64(value),wp.float64(value)-wp.float64(limit[1]))))
    mat=R[w,body];proper=bool(True)
    for j in range(3):
        if not wp.isfinite(com[w,body][j]):wp.atomic_or(errors,w,NONFINITE)
        for k in range(3):
            if not wp.isfinite(mat[j,k]):wp.atomic_or(errors,w,NONFINITE)
            dot=float(0.0)
            for c in range(3):dot=dot+mat[c,j]*mat[c,k]
            target=float(0.0)
            if j==k:target=1.0
            if wp.abs(dot-target)>1e-4:proper=False
    if wp.determinant(mat)<0.999:proper=False
    if not proper:wp.atomic_or(errors,w,BAD_ROTATION)
    for j in range(3):
        for part in range(2):
            for pair in range(2):
                body_value=wp.float64(0.0)
                for k in range(3):body_value=body_value+wp.float64(mat[k,j])*wrench[w,pair,k+3*part]
                wrench[w,pair+2,j+3*part]=body_value


@wp.kernel
def _clear_reset(mask:wp.array(dtype=int),error:wp.array(dtype=int),full:int):
    w=wp.tid();mask[w]=full
    if w==0:error[0]=0


@wp.kernel
def _mark_reset(ids:wp.array(dtype=wp.int64),mask:wp.array(dtype=int),error:wp.array(dtype=int)):
    i=wp.tid();value=ids[i]
    if value<wp.int64(0) or value>=wp.int64(mask.shape[0]):wp.atomic_or(error,0,BAD_RESET_ID)
    else:
        if wp.atomic_add(mask,int(value),1)!=0:wp.atomic_or(error,0,BAD_RESET_ID)


@wp.kernel
def _reset_window(mask:wp.array(dtype=int),reset_error:wp.array(dtype=int),full:int,R:wp.array2d(dtype=wp.mat33),com:wp.array2d(dtype=wp.vec3),time:wp.array(dtype=float),body:int,
    source_R:wp.array(dtype=wp.mat33),source_C:wp.array(dtype=wp.vec3),clock:wp.array2d(dtype=wp.float64),
    geometry:wp.array2d(dtype=wp.float64),fn:wp.array2d(dtype=wp.float64),nonhand:wp.array2d(dtype=wp.float64),
    wrench:wp.array3d(dtype=wp.float64),counts:wp.array2d(dtype=int),errors:wp.array(dtype=int),global_error:wp.array(dtype=int)):
    w=wp.tid()
    if mask[w]!=0 and reset_error[0]==0:
        source_R[w]=R[w,body];source_C[w]=com[w,body]
        clock[w,0]=wp.float64(time[w]);clock[w,1]=wp.float64(time[w])
        for j in range(3):geometry[w,j]=wp.float64(0.0)
        for j in range(6):
            fn[w,j]=wp.float64(0.0)
            for k in range(8):wrench[w,k,j]=wp.float64(0.0)
        nonhand[w,0]=wp.float64(0.0);nonhand[w,1]=wp.float64(0.0)
        counts[w,0]=0;counts[w,1]=0;errors[w]=0
    if w==0:
        if full!=0 and reset_error[0]==0:global_error[0]=0
        global_error[0]=global_error[0]|reset_error[0]


@wp.kernel
def _fold(geom:wp.array2d(dtype=wp.float64),fn:wp.array2d(dtype=wp.float64),other:wp.array(dtype=wp.float64),
    wrench:wp.array3d(dtype=wp.float64),errors:wp.array(dtype=int),global_error:wp.array(dtype=int),counts:wp.array(dtype=int),
    current_C:wp.array2d(dtype=wp.vec3),body:int,time:wp.array(dtype=float),source_R:wp.array(dtype=wp.mat33),source_C:wp.array(dtype=wp.vec3),
    window_geom:wp.array2d(dtype=wp.float64),window_fn:wp.array2d(dtype=wp.float64),window_nonhand:wp.array2d(dtype=wp.float64),
    window_wrench:wp.array3d(dtype=wp.float64),window_counts:wp.array2d(dtype=int),window_errors:wp.array(dtype=int),window_global:wp.array(dtype=int),clock:wp.array2d(dtype=wp.float64)):
    w=wp.tid();window_counts[w,0]=window_counts[w,0]+1;window_counts[w,1]=window_counts[w,1]+counts[w]
    window_errors[w]=window_errors[w]|errors[w]
    if w==0:window_global[0]=window_global[0]|global_error[0]
    clock[w,1]=wp.float64(time[w])
    for j in range(3):window_geom[w,j]=wp.max(window_geom[w,j],geom[w,j])
    for j in range(6):window_fn[w,j]=window_fn[w,j]+fn[w,j]
    window_nonhand[w,0]=wp.max(window_nonhand[w,0],other[w]);window_nonhand[w,1]=window_nonhand[w,1]+other[w]
    mat=source_R[w];arm=wp.vec3d()
    for j in range(3):arm[j]=wp.float64(current_C[w,body][j])-wp.float64(source_C[w][j])
    for j in range(6):
        for k in range(4):window_wrench[w,k,j]=window_wrench[w,k,j]+wrench[w,k,j]
    for pair in range(2):
        F=wp.vec3d();M=wp.vec3d()
        for j in range(3):F[j]=wrench[w,pair,j];M[j]=wrench[w,pair,j+3]
        fixed=M+wp.cross(arm,F)
        for j in range(3):
            force_source=wp.float64(0.0);moment_moving=wp.float64(0.0);moment_fixed=wp.float64(0.0)
            for k in range(3):
                force_source=force_source+wp.float64(mat[k,j])*F[k]
                moment_moving=moment_moving+wp.float64(mat[k,j])*M[k]
                moment_fixed=moment_fixed+wp.float64(mat[k,j])*fixed[k]
            window_wrench[w,pair+4,j]=window_wrench[w,pair+4,j]+force_source
            window_wrench[w,pair+4,j+3]=window_wrench[w,pair+4,j+3]+moment_moving
            window_wrench[w,pair+6,j]=window_wrench[w,pair+6,j]+force_source
            window_wrench[w,pair+6,j+3]=window_wrench[w,pair+6,j+3]+moment_fixed


class ResidentContactMetrics:
    """Bound native arrays, static labels and captured read-only metric graph."""
    def __init__(self,host_model,device_model,data,body,joint_ids,*,capture_graph=True):
        wp.init();self.model=host_model;self.dm=device_model;self.data=data;self.body=int(body)
        self.worlds=data.qpos.shape[0];self.device=data.qpos.device;self.stream=wp.get_stream(self.device) if self.device.is_cuda else None
        self.cone=int(device_model.opt.cone)
        if self.cone not in [0,1] or not 0<=body<host_model.nbody:raise ValueError('Declared rigid object/body and cone')
        joint_ids=np.asarray(joint_ids,dtype=np.int32)
        if len(set(joint_ids.tolist()))!=len(joint_ids) or np.any(joint_ids<0) or np.any(joint_ids>=host_model.njnt):raise ValueError('Canonical unique joint IDs')
        names=[host_model.geom(i).name or '' for i in range(host_model.ngeom)]
        self.object_tag=wp.array((host_model.geom_bodyid==body).astype(np.int32),dtype=int,device=self.device)
        self.digit=wp.array(np.array([normal_group(n) for n in names],dtype=np.int32),dtype=int,device=self.device)
        self.joint_ids=wp.array(joint_ids,dtype=int,device=self.device)
        self.qids=wp.array(np.asarray(host_model.jnt_qposadr[joint_ids],dtype=np.int32),dtype=int,device=self.device)
        if data.nefc.shape[0]!=self.worlds or data.efc.force.shape[0]!=self.worlds:raise ValueError('Per-world EFC ownership required')
        if device_model.jnt_range.shape[0] not in [1,self.worlds]:raise ValueError('Shared or per-world joint bounds')
        self.instant=dict(geometry=wp.zeros((self.worlds,3),dtype=wp.float64,device=self.device),
            Fn6=wp.zeros((self.worlds,6),dtype=wp.float64,device=self.device),nonhand_N=wp.zeros(self.worlds,dtype=wp.float64,device=self.device),
            wrench=wp.zeros((self.worlds,4,6),dtype=wp.float64,device=self.device),errors=wp.zeros(self.worlds,dtype=int,device=self.device),
            global_error=wp.zeros(1,dtype=int,device=self.device),contacts=wp.zeros(self.worlds,dtype=int,device=self.device))
        self.window=dict(geometry_max=wp.zeros((self.worlds,3),dtype=wp.float64,device=self.device),Fn6_sum=wp.zeros((self.worlds,6),dtype=wp.float64,device=self.device),
            nonhand_max_sum=wp.zeros((self.worlds,2),dtype=wp.float64,device=self.device),wrench_sum=wp.zeros((self.worlds,8,6),dtype=wp.float64,device=self.device),
            counts=wp.zeros((self.worlds,2),dtype=int,device=self.device),errors=wp.zeros(self.worlds,dtype=int,device=self.device),global_error=wp.zeros(1,dtype=int,device=self.device),
            source_R=wp.zeros(self.worlds,dtype=wp.mat33,device=self.device),source_C=wp.zeros(self.worlds,dtype=wp.vec3,device=self.device),clock=wp.zeros((self.worlds,2),dtype=wp.float64,device=self.device))
        self.mask=wp.zeros(self.worlds,dtype=int,device=self.device)
        self.reset_error=wp.zeros(1,dtype=int,device=self.device);self._reset_ids_owner=None;self.graph=None
        self.torch_stream=None
        if self.device.is_cuda:
            import torch
            self.torch_stream=torch.cuda.ExternalStream(self.stream.cuda_stream,device=self.device.ordinal)
        self.reset_window();self._launch()
        if self.device.is_cuda:wp.synchronize_stream(self.stream)
        self.setup_metric_samples=1;self.reset_window()
        if capture_graph and self.device.is_cuda:
            with wp.ScopedCapture(stream=self.stream) as capture:self._launch()
            self.graph=capture.graph

    def _launch(self):
        z=self.instant;v=self.window;d=self.data
        wp.launch(_clear_instant,self.worlds,inputs=[z[k] for k in ['geometry','Fn6','nonhand_N','wrench','errors','global_error','contacts']],device=self.device,stream=self.stream)
        c=d.contact
        wp.launch(_contact_pass,d.naconmax,inputs=[d.nacon,c.worldid,c.geom,c.dist,c.dim,c.pos,c.frame,c.friction,c.efc_address,d.efc.force,d.nefc,
            self.object_tag,self.digit,d.xipos,self.body,self.cone,z['geometry'],z['Fn6'],z['nonhand_N'],z['wrench'],z['errors'],z['global_error'],z['contacts']],device=self.device,stream=self.stream)
        wp.launch(_world_pass,self.worlds,inputs=[d.qpos,self.joint_ids,self.qids,self.dm.jnt_range,d.xmat,d.xipos,self.body,d.nacon,d.naconmax,d.nefc,d.njmax,d.overflow,
            z['geometry'],z['wrench'],z['errors'],z['global_error']],device=self.device,stream=self.stream)
        wp.launch(_fold,self.worlds,inputs=[z[k] for k in ['geometry','Fn6','nonhand_N','wrench','errors','global_error','contacts']]+
            [d.xipos,self.body,d.time,v['source_R'],v['source_C'],v['geometry_max'],v['Fn6_sum'],v['nonhand_max_sum'],v['wrench_sum'],v['counts'],v['errors'],v['global_error'],v['clock']],device=self.device,stream=self.stream)

    def _enter(self):
        if not self.device.is_cuda:return None
        import torch
        caller=torch.cuda.current_stream(self.device.ordinal);target=self.torch_stream;target.wait_stream(caller)
        return caller,target

    def _exit(self,streams):
        if streams is not None:streams[0].wait_stream(streams[1])

    def sample(self):
        streams=self._enter()
        if self.graph is None:self._launch()
        else:wp.capture_launch(self.graph,stream=self.stream)
        self._exit(streams)

    def reset_window(self,ids=None):
        streams=self._enter();full=ids is None
        # GPU IDs are checked on device. A bad list resets no worlds and leaves
        # an explicit sticky window error, rather than clamping/aliasing IDs.
        selected=None
        if ids is not None:
            import torch
            if isinstance(ids,torch.Tensor) and ids.is_cuda:
                if not self.device.is_cuda or ids.device.index!=self.device.ordinal or ids.ndim!=1 or ids.dtype not in (torch.int32,torch.int64):
                    raise ValueError('One-dimensional integer IDs on the bound CUDA device')
                with torch.cuda.stream(self.torch_stream):
                    self._reset_ids_owner=ids.to(dtype=torch.int64).contiguous()
                    selected=wp.from_torch(self._reset_ids_owner,dtype=wp.int64)
            elif isinstance(ids,wp.array):
                if ids.device!=self.device or ids.dtype!=wp.int64 or ids.ndim!=1:raise ValueError('Bound device int64 reset IDs')
                selected=ids;self._reset_ids_owner=ids
            else:
                values=np.asarray(ids)
                if values.ndim!=1 or (values.size and values.dtype.kind not in 'iu') or np.any(values<0) or np.any(values>=self.worlds) or len(set(values.tolist()))!=len(values):
                    raise ValueError('Unique in-range integer reset worlds')
                selected=wp.array(values.astype(np.int64),dtype=wp.int64,device=self.device);self._reset_ids_owner=selected
            full=selected.shape[0]==self.worlds
        wp.launch(_clear_reset,self.worlds,inputs=[self.mask,self.reset_error,int(ids is None)],device=self.device,stream=self.stream)
        if selected is not None and selected.shape[0]:
            wp.launch(_mark_reset,selected.shape[0],inputs=[selected,self.mask,self.reset_error],device=self.device,stream=self.stream)
        v=self.window;d=self.data
        wp.launch(_reset_window,self.worlds,inputs=[self.mask,self.reset_error,int(full),d.xmat,d.xipos,d.time,self.body,v['source_R'],v['source_C'],v['clock'],
            v['geometry_max'],v['Fn6_sum'],v['nonhand_max_sum'],v['wrench_sum'],v['counts'],v['errors'],v['global_error']],device=self.device,stream=self.stream)
        self._exit(streams)

    def numpy(self):
        if self.device.is_cuda:wp.synchronize_stream(self.stream)
        return dict(instant={k:v.numpy().copy() for k,v in self.instant.items()},window={k:v.numpy().copy() for k,v in self.window.items()})

    def metadata(self):
        return dict(schema=SCHEMA,geometry=['object_penetration_m','robot_self_penetration_m','joint_excursion_rad'],
            instantaneous_wrench=['all_WORLD_COM(t)','hand_WORLD_COM(t)','all_BODY(t)_COM(t)','hand_BODY(t)_COM(t)'],
            window_wrench_sum=['all_WORLD_COM(t)','hand_WORLD_COM(t)','all_BODY(t)_COM(t)','hand_BODY(t)_COM(t)',
                'all_SOURCE_axes_COM(t)','hand_SOURCE_axes_COM(t)','all_SOURCE_axes_SOURCE_COM','hand_SOURCE_axes_SOURCE_COM'],
            window_normal=['max_nonhand_N','sum_nonhand_N'],counts=['native_samples','contact_rows'],
            normal_group='fingers1..5,other robot',error_bits=dict(BAD_WORLD=1,BAD_GEOM=2,BAD_DIM=4,BAD_EFC=8,NONFINITE=16,
                CAPACITY=32,NEGATIVE_NORMAL=64,OVERFLOW=128,BAD_LIMIT=256,BAD_ROTATION=512,BAD_RESET_ID=1024),
            origin_shift='M_SOURCE_COM = M_COM(t) + (COM(t)-SOURCE_COM) cross F',
            scope='Read-only metrics/FP64 sums, not dynamics/gradient/physical repeat qualification. Divide sums by native_samples; no mean across changing axes/origins silently.')
