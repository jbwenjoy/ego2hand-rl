"""Independent dynamic Data with exact-alias preservation, no live-state writes."""
import dataclasses

from .gpu_model_snapshot import model_leaves


def fork_dynamic_data(data):
    import warp as wp
    arrays,scalars=model_leaves(data)
    signatures={};ranges=[]
    for name,a in arrays.items():
        if a.size and not a.is_contiguous:raise ValueError(f'Noncontiguous dynamic field: {name}')
        signature=(a.ptr,a.shape,a.dtype,a.strides,str(a.device))
        signatures[name]=signature
        if a.size:ranges.append((name,a.ptr,a.ptr+a.size*wp.types.type_size_in_bytes(a.dtype),a.device))
    for i,(name,lo,hi,device) in enumerate(ranges):
        for other,olo,ohi,odevice in ranges[:i]:
            if device==odevice and max(lo,olo)<min(hi,ohi) and signatures[name]!=signatures[other]:
                raise ValueError(f'Unsupported partial/reinterpreted alias: {name}, {other}')
    memo={}
    def clone(value):
        if isinstance(value,wp.array):
            key=(value.ptr,value.shape,value.dtype,value.strides,str(value.device))
            if key not in memo:memo[key]=wp.clone(value)
            return memo[key]
        if dataclasses.is_dataclass(value):return dataclasses.replace(value,**{f.name:clone(getattr(value,f.name)) for f in dataclasses.fields(value)})
        if isinstance(value,list):return [clone(v) for v in value]
        if isinstance(value,tuple):return tuple(clone(v) for v in value)
        return value
    fork=clone(data);targets,current=model_leaves(fork)
    if scalars!=current or set(arrays)!=set(targets):raise ValueError('Fork layout/scalars changed')
    for name,a in arrays.items():
        b=targets[name]
        if (a.shape,a.dtype,a.strides,a.device)!=(b.shape,b.dtype,b.strides,b.device):raise ValueError(f'Fork metadata mismatch: {name}')
        if a.size and any(a.device==dev and max(b.ptr,lo)<min(b.ptr+b.size*wp.types.type_size_in_bytes(b.dtype),hi) for _,lo,hi,dev in ranges):
            raise ValueError('Fork unexpectedly aliases live dynamic storage')
    return fork,dict(arrays=len(arrays),scalar_leaves=len(scalars),
        distinct_nonempty_allocations=len({signatures[n] for n,a in arrays.items() if a.size}),
        exact_alias_groups=[[n for n in arrays if signatures[n]==signature] for signature in set(signatures.values()) if sum(signatures[n]==signature and arrays[n].size>0 for n in arrays)>1])
