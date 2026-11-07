"""Independent finite triangle arithmetic, no producer projection helper/model."""
from pathlib import Path
import hashlib,json,math
import numpy as np
H=Path(__file__).resolve().parent
ROOT=H.parent
P=ROOT.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(v):
    p=Path(v['path']);assert pin(p)==v,'Changed pin '+str(p);return p
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)
def close(a,b,tol=2e-12):return bool(np.max(abs(np.asarray(a)-np.asarray(b)),initial=0)<=tol)
def bits(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);return a.shape==b.shape and a.tobytes()==b.tobytes()
def norm(v):return v/np.linalg.norm(v)
def angle(a,b):
    a,b=norm(a),norm(b);return math.atan2(np.linalg.norm(np.cross(a,b)),float(np.clip(a@b,-1,1)))
def bary(q,tri):
    a,b,c=tri;e0=b-a;e1=c-a;n=np.cross(e0,e1);nn=float(n@n)
    if nn<=0:raise ValueError('Degenerate frozen triangle')
    v=float(np.cross(q-a,e1)@n)/nn;w=float(np.cross(e0,q-a)@n)/nn
    return np.array([1-v-w,v,w])
def closest(q,tri):
    weights=bary(q,tri);candidates=[]
    if min(weights)>=0:candidates.append(weights@tri)
    for a,b in zip(tri,np.roll(tri,-1,axis=0)):
        d=b-a;t=float((q-a)@d)/float(d@d);candidates.append(a+min(1.,max(0.,t))*d)
    return min(candidates,key=lambda p:math.fsum(float(x*x) for x in p-q))
def boundary(q,tri):
    distances=[]
    for a,b in zip(tri,np.roll(tri,-1,axis=0)):
        d=b-a;t=min(1.,max(0.,float((q-a)@d)/float(d@d)));distances.append(float(np.linalg.norm(q-a-t*d)))
    return min(distances),float(np.linalg.norm(tri-q,axis=1).min())

def main():
    rp=ROOT/'report_v1.json';report=read(rp);cfg=read(checked(report['preregistration']))
    checked(report['source']);assert cfg['source']==report['source']
    for rec in cfg['inputs'].values():checked(rec)
    pairs=read(checked(cfg['inputs']['original_anchors']));catalog=read(checked(report['catalog']));comparisons=read(checked(cfg['inputs']['source_face_comparison']))['comparisons']
    with np.load(checked(cfg['inputs']['geometry_arrays']),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    Rp=np.asarray(pairs['source_pose']['phone_body_R_world']);pp=np.asarray(pairs['source_pose']['phone_root_world_m']);Rg=a['geom71_current_world_R'];pg=a['geom71_current_world_p'];Vg=a['geom71_vertices_geom_local_FP32'].astype(float);T=a['geom71_hull_triangles']
    V=np.linalg.solve(Rp,(Vg@Rg.T+pg-pp).T).T;triangles=V[T];center=V.mean(0);normals=[]
    for tri in triangles:
        n=norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]));n=n if n@(tri.mean(0)-center)>0 else -n
        assert np.max((V-tri[0])@n)<2e-12;normals.append(n)
    normals=np.asarray(normals);source=np.asarray([c['original_ego_material_target_phone_body_m'] for c in comparisons]);ranked_by_finger={};catalog_checks=[]
    for i,entry in enumerate(catalog['results']):
        q=source[i];desired=norm(source[i]-source[1-i]);Q=np.asarray([closest(q,t) for t in triangles]);dist=np.linalg.norm(Q-q,axis=1);angles=np.asarray([angle(n,desired) for n in normals]);eligible=np.flatnonzero(dist<=cfg['sensitivity_radius_m']+1e-12)
        ranked=sorted((int(x) for x in eligible),key=lambda x:(float(angles[x]),float(dist[x]),x));saved=entry['candidates_ranked']
        assert ranked==[c['triangle_id'] for c in saved]
        for c in saved:
            tid=c['triangle_id'];p=Q[tid];weights=bary(p,triangles[tid]);ed,vd=boundary(p,triangles[tid])
            assert c['source_vertex_ids']==T[tid].tolist() and close(p,c['projected_point_phone_body_m']) and close(weights,c['barycentric'],1e-9)
            assert weights.min()>=-1e-10 and close(weights@triangles[tid],p)
            assert close(normals[tid],c['outward_normal_phone_body']) and close(dist[tid],c['source_displacement_m']) and close(math.degrees(angles[tid]),c['angle_to_source_pair_outward_line_deg'],1e-8)
            assert close(ed,c['to_edge_m']) and close(vd,c['to_vertex_m'])
        ranked_by_finger[entry['finger']]=dict(ranked=ranked,Q=Q,distance=dist,angles=angles)
        catalog_checks.append(dict(finger=entry['finger'],eligible_triangle_ids=ranked,count=len(ranked),rank0=ranked[0],rank0_to_rank1_normal_angle_difference_deg=math.degrees(angles[ranked[1]]-angles[ranked[0]])))
    branches=[];outputpins={}
    for outpin in report['outputs']:
        output=read(checked(outpin));outputpins[output['branch']]=outpin;chosen_points=[];chosen_normals=[];rows=[]
        for i,(pair,old) in enumerate(zip(output['anchors'],pairs['anchors'])):
            finger=pair['finger'];phone=pair['surfaces']['phone'];tid=phone['source_triangle_id'];q=np.asarray(phone['projected_point_phone_body_m']);weights=bary(q,triangles[tid]);ed,vd=boundary(q,triangles[tid]);n=normals[tid]
            assert pair['surfaces']['hand']==old['surfaces']['hand'] and bits(pair['original_ego_material_target_phone_body_m'],source[i]) and output['source_pose']==pairs['source_pose']
            assert phone['source_triangle_vertex_ids']==T[tid].tolist() and close(triangles[tid],phone['source_triangle_vertices_phone_body_m']) and close(weights@triangles[tid],q) and weights.min()>=-1e-10
            assert close(weights,phone['barycentric_in_source_vertex_order'],1e-9) and close(n,phone['outward_normal_phone_body'])
            assert close(ed,phone['to_edge_m']) and close(vd,phone['to_vertex_m'])
            assert close(np.linalg.norm(q-source[i]),pair['projected_phone_anchor_to_original_ego_material_target_m'])
            source_tid=comparisons[i]['original_ego_triangle']['triangle_id'];normalchange=angle(n,normals[source_tid])
            assert close(math.degrees(normalchange),pair['source_outward_normal_difference_deg'],1e-8)
            if output['branch']=='exact_source_original_hand':assert tid==source_tid and bits(q,source[i])
            else:assert tid==ranked_by_finger[finger]['ranked'][0] and close(q,ranked_by_finger[finger]['Q'][tid]) and np.linalg.norm(q-source[i])<=.001+1e-12
            edge=triangles[tid][1]-triangles[tid][0];t1=norm(edge-n*(edge@n));t2=norm(np.cross(n,t1))
            rows.append(dict(finger=finger,triangle_id=tid,vertex_ids=T[tid].tolist(),point_phone_body_m=q.tolist(),barycentric=weights.tolist(),outward_normal_phone_body=n.tolist(),tangent_basis_body_rows=[t1.tolist(),t2.tolist()],source_displacement_m=float(np.linalg.norm(q-source[i])),source_normal_difference_deg=math.degrees(normalchange),to_triangle_edge_m=ed,to_triangle_vertex_m=vd,boundary_status='ON_OR_WITHIN_1UM_OF_TRIANGLE_EDGE' if ed<=1e-6 else 'TRIANGLE_INTERIOR',original_hand_triangle_unchanged=pair['surfaces']['hand']['source_triangle_id']))
            chosen_points.append(q);chosen_normals.append(n)
        line=norm(chosen_points[1]-chosen_points[0]);inward=[-n for n in chosen_normals];antipodal=[math.degrees(angle(inward[0],line)),math.degrees(angle(inward[1],-line))];half=math.degrees(math.atan(.7))
        branches.append(dict(name=output['branch'],rows=rows,phone_pair_span_m=float(np.linalg.norm(chosen_points[1]-chosen_points[0])),inward_normal_to_compressive_pair_line_deg=antipodal,assumed_circular_friction_mu=.7,circular_friction_half_angle_deg=half,pure_collinear_compressive_pair_angular_descriptor=all(x<=half for x in antipodal),scope='Circular-Coulomb two-point pure compression descriptor only; not gravity-wrench, native pyramid basis, measured contact, actual manifold or holding qualification.'))
    checks={k:True for k in ['frozen_builder_prereg_inputs_and_outputs_pins','independent_cross_normals_outward_and_filled_triangle_projections','all_source_ball_members_and_predeclared_full_ranking','all_candidate_and_branch_barycentric_normal_boundary_arithmetic','exact_source_point_bytes_and_original_hand307_unchanged','sensitivity_within_registered1mm_original_targets_not_overwritten']}
    result=dict(schema='source_surface_mapping_saved_geometry_review:11',status='PASS_SAVED_FINITE_GEOMETRY_MAPPING_NOT_CONTACT_OR_GRASP_QUALIFICATION',source=pin(__file__),producer_report=pin(rp),preregistration=report['preregistration'],inputs=cfg['inputs'],branch_pins=outputpins,catalog=report['catalog'],checks=checks,passed=sum(checks.values()),total=len(checks),catalog_membership=catalog_checks,branches=branches,source_domain_review=dict(exact_branch='Keeps original projected ego material points bytes, not a measured contact state.',sensitivity_branch='Algorithmic1mm source-surface ball, not empirical uncertainty. Frozen independently named result, not interchangeable with exact-source claims.',index_normal_sensitivity='About.495mm crosses a38.5degree facet-normal difference. Distance alone cannot proxy normal fidelity.',thumb_boundary='Sensitivity thumb selected point is on triangle edge; one face normal is conditional and need not be the native contact normal.',lexicographic_rank_sensitivity='Index triangles41/40 have extremely close normal scores; chosen41 is exactly the predeclared angle-first rule, not evidence for a mechanically preferred normal or displacement.'),new_model_FK_LP_physics_training_calls=0,independent_filled_triangle_arithmetic_projections=2*len(T),producer_projection_helper_used_as_truth=False,source_or_assembly_or_holding_qualified=False,scope='Independent saved-geometry cross-product/barycentric/segment arithmetic. No producer projection function import, native model/FK, LP, optimization or physics. Geometry-domain/proximity candidates and geometric normals are not measured contact/force labels; hand pad function remains UNKNOWN. Two-point angular diagnostics do not prove or disprove particular gravity support, actual multipoint contact, or2s holding.')
    save(H/'report.json',result);print(json.dumps({'report':pin(H/'report.json'),'passed':result['passed'],'total':result['total'],'branches':[{'name':b['name'],'pair_angles_deg':b['inward_normal_to_compressive_pair_line_deg']} for b in branches]}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(H/'failure.json',dict(error=repr(e),source=pin(__file__),no_automatic_retry=True));raise
