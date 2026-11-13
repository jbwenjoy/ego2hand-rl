"""Read actual GPU constraint indices; inactive contacts retain zero wrench."""
import numpy as np


def decode_local_forces(cone, dimensions, friction, addresses, efc_force, nefc):
    dimensions=np.asarray(dimensions);addresses=np.asarray(addresses);force=np.asarray(efc_force)
    result=np.zeros((len(dimensions),6));invalid=np.zeros(len(dimensions),dtype=bool)
    for i,dim in enumerate(dimensions):
        dim=int(dim)
        if dim not in [1,3,4,6]:raise ValueError('Supported rigid contact dimension')
        if addresses[i,0]<0:continue
        width=1 if dim==1 else (2*(dim-1) if int(cone)==0 else dim)
        ids=addresses[i,:width]
        if len(ids)!=width or np.any(ids<0) or np.any(ids>=nefc) or np.any(ids>=len(force)):
            invalid[i]=True
            continue
        values=force[ids].astype(np.float64)
        if int(cone)==0 and dim>1:
            result[i,0]=values.sum()
            result[i,1:dim]=(values[0::2]-values[1::2])*friction[i,:dim-1]
        else:
            result[i,:dim]=values
    return result,invalid


def read_warp_contact_labels(model, data, world_id=0):
    count=int(data.nacon.numpy()[0]);limit=min(count,data.naconmax)
    ids=np.flatnonzero(data.contact.worldid.numpy()[:limit]==world_id)
    fields={key:getattr(data.contact,key).numpy()[ids] for key in
            ['dim','geom','dist','pos','frame','friction','efc_address']}
    force=data.efc.force.numpy()[world_id];nefc=int(data.nefc.numpy()[world_id])
    local,invalid=decode_local_forces(int(model.opt.cone),fields['dim'],fields['friction'],fields['efc_address'],force,nefc)
    fields.update(local_force6=local,invalid_constraint_index=invalid,contact_ids=ids,
                  nefc=nefc,nacon=count,overflow=count>data.naconmax or nefc>data.njmax or bool(invalid.any()))
    return fields
