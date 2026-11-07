"""Independent mesh-vertex/cylinder support-plane separation at failed state.

No distance/collision query is called here. The trial axis is taken from the
previously logged original-option query, but any axis giving positive support
gap suffices to demonstrate nonintersection of the two modeled convex sets.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib,json,sys
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path('D:/workspace/supermimic-observation-retarget-study');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from contact_penetration_geometry import ContactGeometry
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}

def main():
    source=OUT/'failure_snapshot_review.json';review=json.loads(source.read_text())
    option=next(row for row in review['queries_and_native_collision'] if row['variant']=='native_original_MJB_options' and row.get('distmax')==1. and row['geom_first']=='robot/right_finger1_link4_col')
    n=np.array(option['normal_if_divided_by_signed_distance']);n/=np.linalg.norm(n)
    snapshot=ROOT/'outputs/contact_refined_geometry_debug_v1/query_failure_snapshot.npz';z=np.load(snapshot,allow_pickle=False)
    model=Path(review['model']['path']);context=Path(review['context']['path']);before={str(p):sha(p) for p in [model,context,ROOT/'scripts/contact_penetration_geometry.py']}
    g=ContactGeometry(model,context);g.set_object_pose(z['object_pose']);g.robot.forward(z['q']);m,d=g.model,g.data;h,o=map(int,g.pairs[12])
    assert m.geom_type[h]==mujoco.mjtGeom.mjGEOM_MESH and m.geom_type[o]==mujoco.mjtGeom.mjGEOM_CYLINDER
    mesh=int(m.geom_dataid[h]);assert np.array_equal(m.mesh_scale[mesh],np.ones(3))
    vertices=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]].astype(np.float64)
    handR=d.geom_xmat[h].reshape(3,3);objectR=d.geom_xmat[o].reshape(3,3)
    handcenter=d.geom_xpos[h].copy();objectcenter=d.geom_xpos[o].copy();worldvertices=vertices@handR.T+handcenter
    mesh_max=float(np.max(worldvertices@n));localn=objectR.T@n;r,half_length=map(float,m.geom_size[o,:2])
    cyl_support=r*np.linalg.norm(localn[:2])+half_length*abs(localn[2])
    cylinder_min=float(n@objectcenter-cyl_support);gap=cylinder_min-mesh_max
    # Cross-check the algebra in relative coordinates to avoid global offsets.
    relative_gap=float(n@(objectcenter-handcenter)-np.max(vertices@(handR.T@n))-cyl_support)
    data={'status':'POSITIVE_SEPARATING_AXIS_LOWER_BOUND_PAIR12_ONLY','pair_index':12,'hand_geom':g.names[h],'object_geom':g.names[o],'hand_geom_type':'mesh convex hull','object_geom_type':'finite cylinder','unit_axis_hand_to_object_w':n.tolist(),'mesh_vertex_count':len(vertices),'compiled_mesh_vertex_sha256':hashlib.sha256(vertices.tobytes()).hexdigest(),'mesh_scale':[1.,1.,1.],
        'hand_max_projection_m':mesh_max,'cylinder_min_projection_m':cylinder_min,'positive_separation_lower_bound_m':gap,'relative_coordinates_same_gap_m':relative_gap,'relative_vs_global_difference_m':abs(gap-relative_gap),'original_options_distance_m':option['distance_m'],'lower_bound_vs_original_query_difference_m':gap-option['distance_m'],
        'strict_options_reported_penetration_m':next(row['distance_m'] for row in review['queries_and_native_collision'] if row['variant']=='native_frozen_query_options' and row.get('distmax')==1. and row['geom_first']=='robot/right_finger1_link4_col'),
        'support_formula':'max_hand(n)=max_i n.dot(R_hand*compiled_mesh_vertex_i+p_hand); min_cylinder(n)=n.dot(p_obj)-r*sqrt(local_nx^2+local_ny^2)-half_length*abs(local_nz); gap=min_cylinder-max_hand.',
        'interpretation':'All compiled mesh vertices and therefore their convex hull are below one support plane; every finite-cylinder point lies above the other. Positive gap demonstrates this pair is separated at the snapshot, contradicting the strict-option negative distance. This is a FP64 geometric separating-axis bound, not an interval-arithmetic distance certificate, not a claim that every other hand/object pair is clear, and not physical-contact evidence.',
        'trial_axis_origin':'Frozen logged original-MJB-option positive witness; correctness of the support-plane inequality does not assume that query found the optimal axis.',
        'source_query_review':pin(source),'snapshot':pin(snapshot),'model':pin(model),'context':pin(context),'code':pin(__file__),'model_and_source_unchanged':before=={p:sha(p) for p in before},'distance_or_collision_API_calls':0,'dynamics_steps':0,'physical_reference_qualified':False}
    assert gap>0 and data['relative_vs_global_difference_m']<1e-12
    (OUT/'separation_certificate.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(data,indent=2))

if __name__=='__main__':main()
