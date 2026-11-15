"""Validate layouts and overlapping saved views BEFORE any runtime write."""
from dataclasses import dataclass
import numpy as np
import torch
import warp as wp


def host(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().numpy().copy()
    if isinstance(value,wp.array):return value.numpy().copy()
    return np.asarray(value).copy()


def layout(target,template):
    if isinstance(target,torch.Tensor):return target.data_ptr(),tuple(s*target.element_size() for s in target.stride()),str(target.device)
    if isinstance(target,wp.array):
        tail=template.shape[len(target.shape):];strides=[];size=template.dtype.itemsize
        for n in reversed(tail):strides.insert(0,size);size*=n
        return target.ptr,tuple(target.strides)+tuple(strides),str(target.device)
    return target.__array_interface__['data'][0],tuple(target.strides),'cpu'


@dataclass
class Write:
    name:str
    target:object
    value:np.ndarray
    pointer:int
    strides:tuple
    device:str
    reduced:np.ndarray
    zero_axes:tuple
    copy_target:object

    def apply(self):
        target=self.target
        if isinstance(target,torch.Tensor):
            for axis in reversed(self.zero_axes):target=target.select(axis,0)
            target.copy_(torch.as_tensor(self.reduced.copy(),device=target.device))
        elif isinstance(target,wp.array):
            if target.size:
                value=self.reduced
                # A collapsed scalar array uses one explicit outer element.
                if len(self.zero_axes)==target.ndim:value=value.reshape((1,)+self.value.shape[target.ndim:])
                self.copy_target.assign(value)
        else:target[...]=self.value

    def equal(self):return host(self.target).tobytes()==self.value.tobytes()


def prepare_writes(entries):
    """Return a checked copy plan; never change targets, including on failure.

    Aliases are validated by physical byte addresses, including native Warp /
    Torch views and vector dtypes. NaN scratch bytes are compared exactly.
    """
    result=[]
    for name,target,saved in entries:
        template=host(target);value=host(saved)
        if isinstance(target,np.ndarray) and not target.flags.writeable:raise ValueError('Readonly array destination '+name)
        if template.shape!=value.shape or template.dtype!=value.dtype:raise ValueError('Saved layout mismatch '+name)
        pointer,strides,device=layout(target,template)
        if any(s<0 for s in strides):raise ValueError('Negative-stride destination unsupported '+name)
        axes=tuple(i for i,(n,s) in enumerate(zip(value.shape,strides)) if n>1 and s==0);reduced=value
        for axis in reversed(axes):
            first=np.take(reduced,0,axis=axis)
            expanded=np.broadcast_to(np.expand_dims(first,axis),reduced.shape)
            if expanded.tobytes()!=reduced.tobytes():raise ValueError('Conflicting expanded view '+name)
            reduced=first
        copy_target=target
        if isinstance(target,wp.array) and axes:
            if any(axis>=target.ndim for axis in axes):raise ValueError('Expanded Warp vector component unsupported '+name)
            shape=tuple(n for i,n in enumerate(target.shape) if i not in axes)
            sub_strides=tuple(s for i,s in enumerate(target.strides) if i not in axes)
            if not shape:shape=(1,);sub_strides=(wp.types.type_size_in_bytes(target.dtype),)
            copy_target=wp.array(ptr=target.ptr,dtype=target.dtype,shape=shape,strides=sub_strides,device=target.device,copy=False)
        result.append(Write(name,target,value,pointer,strides,device,reduced,axes,copy_target))
    # Interval grouping is only a fast candidate search. Exact byte offsets
    # below determine whether padded/strided aliases really overlap.
    by_device={}
    for write in result:
        if write.value.size:
            end=write.pointer+sum((n-1)*s for n,s in zip(write.value.shape,write.strides))+write.value.dtype.itemsize
            by_device.setdefault(write.device,[]).append((write.pointer,end,write))
    for intervals in by_device.values():
        groups=[]
        for start,end,write in sorted(intervals,key=lambda x:x[0]):
            if not groups or start>=groups[-1][1]:groups.append([start,end,[write]])
            else:groups[-1][1]=max(groups[-1][1],end);groups[-1][2].append(write)
        for start,end,writes in groups:
            # A lone contiguous destination cannot conflict with another view.
            if len(writes)==1 and not writes[0].zero_axes:
                w=writes[0]
                if w.strides==w.value.strides:continue
            shadow=np.zeros(end-start,dtype=np.uint8);used=np.zeros(end-start,dtype=bool)
            for w in writes:
                shape=list(w.value.shape);strides=list(w.strides)
                for axis in reversed(w.zero_axes):shape.pop(axis);strides.pop(axis)
                offsets=np.zeros(tuple(shape),dtype=np.int64)
                for axis,(n,s) in enumerate(zip(shape,strides)):
                    form=[1]*len(shape);form[axis]=n;offsets+=np.arange(n).reshape(form)*s
                indices=(offsets.reshape(-1,1)+np.arange(w.value.dtype.itemsize)+w.pointer-start).reshape(-1)
                values=np.ascontiguousarray(w.reduced).reshape(-1).view(np.uint8)
                if len(np.unique(indices))!=len(indices):raise ValueError('Internally overlapping destination '+w.name)
                if np.any(used[indices]&(shadow[indices]!=values)):raise ValueError('Conflicting aliased saved bytes '+w.name)
                shadow[indices]=values;used[indices]=True
    return result
