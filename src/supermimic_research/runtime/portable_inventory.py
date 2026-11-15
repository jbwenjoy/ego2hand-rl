"""Stable-path inventory of owned mutable runtime leaves, with opaque objects explicit.

This inventory alone is not a portable checkpoint or a completeness proof.
Native engine model/Data arrays are enumerated separately by model_leaves.
"""
import enum
from pathlib import Path
import types

import numpy as np
import torch


def runtime_inventory(root,*,native_roots=(),excluded_paths=()):
    import warp as wp
    seen={id(v):('NATIVE_ROOT',i) for i,v in enumerate(native_roots)}
    tensors={};arrays={};scalars={};opaque=[];aliases=[];objects={};containers={};excluded=[]
    owned=('mjlab.','dex_supermimic.','supermimic_research.')
    def walk(value,path):
        name=repr(path)
        if path in excluded_paths:
            excluded.append(dict(path=path,type=type(value).__module__+'.'+type(value).__qualname__))
            return
        if isinstance(value,(str,int,float,bool,type(None),np.generic,enum.Enum,Path,torch.device,torch.dtype,slice)):
            scalars[path]=value
            return
        if isinstance(value,(types.FunctionType,types.MethodType,types.BuiltinFunctionType,type)):
            opaque.append(dict(path=path,kind='callable',type=type(value).__module__+'.'+type(value).__qualname__,
                qualified_name=getattr(value,'__module__','')+'.'+getattr(value,'__qualname__','')))
            return
        if id(value) in seen:
            aliases.append(dict(path=path,target=seen[id(value)]));return
        seen[id(value)]=path
        if isinstance(value,torch.Tensor):tensors[path]=value;return
        if isinstance(value,(np.ndarray,wp.array)):arrays[path]=value;return
        if isinstance(value,(dict,list,tuple,set,frozenset)):
            kind=type(value).__name__;containers[path]=kind
            if isinstance(value,dict):
                for key,child in value.items():
                    if not isinstance(key,(str,int,float,tuple)):
                        opaque.append(dict(path=path,kind='unsupported_dict_key',type=str(type(key))))
                    else:walk(child,path+(('item',key),))
            elif isinstance(value,(list,tuple)):
                for i,child in enumerate(value):walk(child,path+(('item',i),))
            else:
                # Configuration sets have no ordering; preserve as a single leaf.
                scalars[path]=value.copy() if isinstance(value,set) else value
            return
        module=type(value).__module__;cls=module+'.'+type(value).__qualname__
        if module.startswith(owned) and hasattr(value,'__dict__'):
            objects[path]=cls
            for key,child in vars(value).items():walk(child,path+(('attr',key),))
            return
        opaque.append(dict(path=path,kind='opaque',type=cls))
    walk(root,())
    return dict(tensors=tensors,arrays=arrays,scalars=scalars,opaque=opaque,aliases=aliases,
        objects=objects,containers=containers,excluded=excluded)


def path_value(root,path):
    for kind,key in path:root=getattr(root,key) if kind=='attr' else root[key]
    return root


def assign_path(root,path,value):
    if not path:raise ValueError('Cannot replace runtime root')
    parent=path_value(root,path[:-1]);kind,key=path[-1]
    if readonly_scalar_path(root,path):
        if path_value(root,path)!=value:raise ValueError('Changed immutable scalar field')
    elif kind=='attr':setattr(parent,key,value)
    else:parent[key]=value


def readonly_scalar_path(root,path):
    parent=path_value(root,path[:-1]);kind,_=path[-1]
    params=getattr(type(parent),'__dataclass_params__',None)
    return (kind=='item' and isinstance(parent,tuple)) or (kind=='attr' and params is not None and params.frozen)


def describe_inventory(inv):
    def shape(value):
        return dict(shape=list(value.shape),dtype=str(value.dtype),device=str(getattr(value,'device','cpu')))
    result=dict(tensors=[dict(path=p,**shape(v)) for p,v in inv['tensors'].items()],
        arrays=[dict(path=p,**shape(v)) for p,v in inv['arrays'].items()],
        scalars=[dict(path=p,type=type(v).__module__+'.'+type(v).__qualname__) for p,v in inv['scalars'].items()],
        objects=[dict(path=p,type=v) for p,v in inv['objects'].items()],
        containers=[dict(path=p,type=v) for p,v in inv['containers'].items()],opaque=inv['opaque'],aliases=inv['aliases'],excluded=inv['excluded'])
    # Attribute insertion order differs before/after the env's first step. Stable
    # paths and alias targets, not incidental traversal order, define the schema.
    return {key:sorted(rows,key=lambda r:repr(r['path'])) for key,rows in result.items()}
