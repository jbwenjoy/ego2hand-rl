"""Active multiworld native contacts with explicit owning-world EFC decoding."""
import numpy as np
from .warp_contact_labels import decode_local_forces


def contacts(model,data,host_model,body):
    total=int(data.nacon.numpy()[0]);count=min(total,data.naconmax)
    fields={key:getattr(data.contact,key).numpy()[:count].copy() for key in
        ['worldid','geom','dim','dist','pos','frame','friction','efc_address']}
    nefc=data.nefc.numpy().copy();efc=data.efc.force.numpy().copy();worlds=len(nefc)
    valid=(fields['worldid']>=0)&(fields['worldid']<worlds)
    local=np.zeros((count,6));invalid=np.zeros(count,dtype=bool)
    for w in range(worlds):
        ids=np.flatnonzero(fields['worldid']==w)
        local[ids],invalid[ids]=decode_local_forces(int(model.opt.cone),fields['dim'][ids],fields['friction'][ids],
            fields['efc_address'][ids],efc[w],int(nefc[w]))
    com=data.xipos.numpy()[:,body].copy();R=data.xmat.numpy()[:,body].copy()
    wrench=np.zeros((worlds,6));Fn=np.zeros((worlds,6));nonhand=np.zeros(worlds)
    for i in np.flatnonzero(valid):
        w=int(fields['worldid'][i]);g,h=map(int,fields['geom'][i])
        if min(g,h)<0 or max(g,h)>=host_model.ngeom:invalid[i]=True;continue
        og,oh=host_model.geom_bodyid[[g,h]]==body
        if og==oh:continue
        sign=1. if oh else -1.;other=g if oh else h
        frame=fields['frame'][i];F=sign*frame.T@local[i,:3];T=sign*frame.T@local[i,3:]
        wrench[w,:3]+=F;wrench[w,3:]+=T+np.cross(fields['pos'][i]-com[w],F)
        name=host_model.geom(other).name or ''
        if name.startswith('robot/'):
            digit=next((j-1 for j in range(1,6) if name.startswith(f'robot/right_finger{j}_')),5)
            Fn[w,digit]+=local[i,0]
        else:nonhand[w]+=local[i,0]
    return dict(**fields,local_force6=local,nefc=nefc,efc_force=efc,
        COM_efc_phase_world=com,R_efc_phase_world=R,wrench_WORLD_COM=wrench,Fn6=Fn,nonhand_N=nonhand,
        flags=dict(valid_world_ids=bool(valid.all()),valid_addresses=not invalid.any(),
            capacity=total<=data.naconmax and bool(np.all(nefc<=data.njmax)),
            finite_active=all(np.isfinite(v).all() for v in [local,fields['dist'],fields['pos'],fields['frame'],wrench,Fn]),
            engine_overflow=not data.overflow.numpy().any()))
