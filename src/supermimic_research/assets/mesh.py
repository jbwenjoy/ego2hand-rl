"""Baked source triangles and declared surface-lamina inertial prior.

No model/data is loaded on import. Exact position welding preserves every
triangle's geometric coordinates, including open/nonmanifold source regions.
"""
import numpy as np


def bake_scene(scene,expected_bounds):
    import trimesh
    vertices=[];faces=[];nodes=[];offset=0
    for name in sorted(scene.graph.nodes_geometry):
        transform,key=scene.graph[name];geometry=scene.geometry[key];T=np.asarray(transform,dtype=float)
        if not isinstance(geometry,trimesh.Trimesh) or T.shape!=(4,4) or not np.isfinite(T).all() or not np.array_equal(T[3],[0,0,0,1]) or np.linalg.det(T[:3,:3])<=0:raise ValueError('Reviewed finite positive-orientation triangular scene required')
        v=np.asarray(geometry.vertices,dtype=float);f=np.asarray(geometry.faces,dtype=np.int64)
        if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v):raise ValueError('Raw finite indexed triangles')
        baked=v@T[:3,:3].T+T[:3,3];vertices.append(baked);faces.append(f+offset)
        nodes.append(dict(node=name,geometry=key,transform_to_model=T.tolist(),determinant=float(np.linalg.det(T[:3,:3])),vertices=len(v),faces=len(f),vertex_offset=offset));offset+=len(v)
    v=np.concatenate(vertices);f=np.concatenate(faces);bounds=np.array([v.min(0),v.max(0)])
    error=float(abs(bounds-np.asarray(expected_bounds)).max())
    if error>1e-10:raise ValueError('Scene/model-info bounds mismatch: no implicit unit guess or recentering')
    return v,f,dict(nodes=nodes,bounds_model_m=bounds.tolist(),bounds_error_m=error,policy='Apply each node transform once; original model origin retained, no extra unit scale.')


def exact_weld(vertices,faces):
    v,inverse=np.unique(np.asarray(vertices),axis=0,return_inverse=True);f=inverse[np.asarray(faces)]
    if not np.array_equal(v[f],np.asarray(vertices)[np.asarray(faces)]):raise ValueError('Exact welding changed source triangles')
    return v,f


def topology(vertices,faces):
    import trimesh
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    _,incidence=np.unique(mesh.edges_sorted,axis=0,return_counts=True)
    return dict(vertices=len(vertices),faces=len(faces),boundary_edges=int(np.sum(incidence==1)),nonmanifold_edges=int(np.sum(incidence>2)),
        degenerate_triangles=int(np.sum(mesh.area_faces<=1e-18)),watertight=bool(mesh.is_watertight),winding_consistent=bool(mesh.is_winding_consistent),euler=int(mesh.euler_number),
        signed_volume_descriptor_m3=float(mesh.volume),volume_not_used_for_inertia=True)


def surface_mass_prior(vertices,faces,mass):
    t=np.asarray(vertices)[np.asarray(faces)];area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2
    if not np.isfinite(t).all() or mass<=0 or area.sum()<=0:raise ValueError('Finite surface and positive declared mass')
    weight=area/area.sum();S=t.sum(1);com=np.sum(weight[:,None]*S/3,axis=0)
    second=(np.einsum('nki,nkj->nij',t,t)+np.einsum('ni,nj->nij',S,S))/12
    covariance=np.sum(weight[:,None,None]*second,axis=0)-np.outer(com,com)
    inertia=mass*(np.trace(covariance)*np.eye(3)-covariance)
    eigenvalues=np.linalg.eigvalsh(inertia)
    if eigenvalues.min()<=0 or eigenvalues[-1]>eigenvalues[:2].sum()+1e-14:raise ValueError('Physical inertia triangle inequalities')
    return dict(mass_kg=float(mass),COM_model_m=com.tolist(),inertia_model_kg_m2=inertia.tolist(),principal_values_kg_m2=eigenvalues.tolist(),
        surface_area_m2=float(area.sum()),model='Uniform triangle-area lamina prior, exact triangle second moments; not measured material density/mass/COM/inertia. Source volume unused.')


def union_membership(points,halfspaces):
    p=np.asarray(points,dtype=float);inside=np.zeros(len(p),dtype=bool)
    for equations in halfspaces:inside|=np.all(p@equations[:,:3].T+equations[:,3]<=1e-10,axis=1)
    return inside


def winding_numbers(points,vertices,faces):
    """Generalized winding descriptor; open meshes give fractional/unknown values."""
    triangles=np.asarray(vertices)[np.asarray(faces)];values=[]
    for p in np.asarray(points):
        a,b,c=(triangles-p).transpose(1,0,2);la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
        numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
        denominator=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
        values.append(float(np.sum(2*np.arctan2(numerator,denominator))/(4*np.pi)))
    return np.array(values)
