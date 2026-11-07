"""Saved-only, deterministic pose-seed design; no FK/query/LP/native calls."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
P=HERE.parent


def pin(f):
    f=Path(f).resolve();b=f.read_bytes()
    return {'path':str(f),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def read(f):return json.loads(Path(f).read_text(encoding='utf-8'))


def unit(v):return np.asarray(v)/np.linalg.norm(v)


def main():
    rules=read(HERE/'rules.json')
    if (HERE/'report.json').exists():raise FileExistsError('fresh analysis only')
    for rec in rules['input_pins'].values():
        assert pin(rec['path'])==rec
    assert pin(__file__)==rules['source']
    parent=read(rules['input_pins']['native_report']['path'])
    c=next(s for s in parent['seed_results'] if s['seed']==0)['shortlist'][0]
    assert c['rank']==0
    n=c['native'];r=n['raw'];R=np.array(r['phone_body_R_world']);root=np.array(r['phone_root_world_m'])
    COM=np.array(r['phone_COM_world_m']);raw=np.array(r['contact_rows24']);points=raw[:,6:9]
    details=n['native_contact_details'];assert raw.shape==(3,24)
    placement=read(rules['input_pins']['placement']['path']);T=np.array(placement['T_world_from_palm'])
    Rp,pp=T[:3,:3],T[:3,3]
    assert np.max(abs(Rp.T@(root-pp)-n['phone_t_palm_m']))<1e-12
    gravity=np.array([0.,0.,-9.81]);mass=.2;up=-mass*gravity;body_up=R.T@up
    pre=read(rules['input_pins']['fixed_anchors']['path'])
    V=np.array(pre['phone_body_vertices']);material=np.array([a['target_projected_phone_convex_object_m'] for a in pre['preparation']])
    contact_body=(points-root)@R
    normals_body=raw[:,9:12]@R
    line=unit(points[1:].mean(axis=0)-points[0]);grad=np.cross(mass*gravity,line)
    axial_tau=float(grad@(COM-points[0]));delta=-axial_tau*grad/(grad@grad)
    body_z_shift=-axial_tau/float(grad@R[:,2])
    # Finger-balanced height; two nearly duplicated index manifold rows must not
    # give that finger twice the weight when choosing an engineering reference.
    long_slide=float(.5*(contact_body[0,2]+contact_body[1:,2].mean())-(R.T@(COM-root))[2])
    oldlp=read(rules['input_pins']['two_LP_report']['path'])
    rays_world=np.array(oldlp['force_rays3x12']);rays_body=R.T@rays_world
    # Enumerate all candidate dual cone faces, no force allocation or LP solve.
    faces=[]
    for i in range(rays_body.shape[1]):
        for j in range(i+1,rays_body.shape[1]):
            cross=np.cross(rays_body[:,i],rays_body[:,j]);norm=np.linalg.norm(cross)
            if norm<1e-12:continue
            nn=cross/norm;dots=nn@rays_body
            if np.min(dots)>=-1e-10:pass
            elif np.max(dots)<=1e-10:nn=-nn;dots=-dots
            else:continue
            if any(np.dot(nn,f['normal_body'])>1-1e-10 for f in faces):continue
            faces.append({'normal_body':nn.tolist(),'normal_world':(R@nn).tolist(),
                'minimum_dot_all_force_rays':float(dots.min()),'maximum_dot_all_force_rays':float(dots.max()),
                'dot_required_upward_force_N':float(nn@body_up),'ray_pair':[i,j]})
    violated=[f for f in faces if f['dot_required_upward_force_N']<-1e-10]
    # Interior reference is the sum of every positive force ray, not a selected solution.
    f0=unit(body_up);f1=unit(rays_body.sum(axis=1));beta=float(np.arccos(np.clip(f0@f1,-1,1)))
    e=unit(f1-(f0@f1)*f0);axis=unit(np.cross(f0,f1))
    enter=max([math.atan2(-np.dot(f['normal_body'],f0),np.dot(f['normal_body'],e)) for f in violated],default=0.)
    turn=min(beta,enter+math.radians(rules['force_cone_interior_margin_deg']))
    # New R.T*up follows the great circle toward the all-ray interior direction.
    Rtilt=R@Rotation.from_rotvec(-turn*axis).as_matrix()
    target_after=Rtilt.T@up
    ipos=R.T@(COM-root)
    cfg=read(rules['input_pins']['native_protocol']['path'])
    oldpin=cfg['inputs']['preloaded_seed0'];assert pin(oldpin['path'])==oldpin
    with np.load(oldpin['path'],allow_pickle=False) as z:q0=z['qpos27'].copy()
    Rstart=Rotation.from_quat(q0[23:27][[1,2,3,0]]).as_matrix()
    tstart=Rp.T@(q0[20:23]-pp);rvstart=Rotation.from_matrix(Rp.T@Rstart).as_rotvec()
    lo_t=tstart-.03;hi_t=tstart+.03;lo_r=np.maximum(-np.pi,rvstart-.35);hi_r=np.minimum(np.pi,rvstart+.35)
    seeds=[]
    # Four proposed engineering starts, not evaluated geometry or reused GT contacts.
    for name,s,Rnew in [('slide_half',long_slide/2,R),('tilt_to_force_cone',0.,Rtilt),
                        ('tilt_and_slide_half',long_slide/2,Rtilt),('tilt_and_slide_full',long_slide,Rtilt)]:
        COMnew=COM+s*Rnew[:,2];rootnew=COMnew-Rnew@ipos
        tnew=Rp.T@(rootnew-pp);rvnew=Rotation.from_matrix(Rp.T@Rnew).as_rotvec()
        eng_material=material.copy();eng_material[:,2]-=s
        seeds.append({'name':name,'longitudinal_material_reassignment_m':s,'phone_COM_world_m':COMnew.tolist(),
            'phone_root_world_m':rootnew.tolist(),'phone_R_body_world':Rnew.tolist(),
            'phone_root_translation_palm_m':tnew.tolist(),'phone_rotation_vector_palm_rad':rvnew.tolist(),
            'root_delta_palm_m':(tnew-np.array(n['phone_t_palm_m'])).tolist(),
            'rotation_delta_geodesic_deg':math.degrees(Rotation.from_matrix(R.T@Rnew).magnitude()),
            'inside_old_translation_trust':bool(np.all(tnew>=lo_t)&np.all(tnew<=hi_t)),
            'inside_old_rotvec_trust':bool(np.all(rvnew>=lo_r)&np.all(rvnew<=hi_r)),
            'hypothetical_engineering_material_targets_body_m':eng_material.tolist(),
            'ego_material_points_unchanged_in_evidence':True,
            'new_targets_surface_membership_or_robot_reachability':'UNKNOWN_NOT_QUERIED',
            'q20_seed':c['x46'][:20],'geometry_LP_holding_qualified':False})
    contact_records=[]
    for i,(point,detail) in enumerate(zip(points,details)):
        finger=0 if detail['category']=='thumb_pad' else 1
        contact_records.append({'contact_index':i,'finger':detail['category'],'position_body_m':contact_body[i].tolist(),
            'position_palm_m':(Rp.T@(point-pp)).tolist(),'normal_on_object_body':normals_body[i].tolist(),
            'normal_on_object_palm':(Rp.T@raw[i,9:12]).tolist(),
            'distance_to_corresponding_ego_projected_material_point_mm':1000*float(np.linalg.norm(contact_body[i]-material[finger]))})
    dot=float(np.clip(normals_body[0]@normals_body[1],-1,1))
    result={'schema':'saved_reference_seed_geometry_design:7','role':'engineering_pose_seed_proposal_not_reference_qualification',
        'rules':pin(HERE/'rules.json'),'source':pin(__file__),'inputs':rules['input_pins'],
        'current':{'phone_root_palm_m':n['phone_t_palm_m'],'phone_rotvec_palm_rad':n['phone_rotvec_rad'],
            'body_COM_offset_m':ipos.tolist(),'contact_records':contact_records,
            'opposing_inward_normal_angle_deg':math.degrees(math.acos(dot)),
            'gravity_body_m_s2':(R.T@gravity).tolist(),'gravity_palm_m_s2':(Rp.T@gravity).tolist(),
            'phone_body_vertex_min_m':V.min(axis=0).tolist(),'phone_body_vertex_max_m':V.max(axis=0).tolist(),
            'original_ego_projected_material_body_m':material.tolist()},
        'lever_only_diagnostic':{'line_definition':'Thumb point to arithmetic centroid of both index points; descriptor only, all three remain in LP.',
            'line_axis_body':(R.T@line).tolist(),'axial_gravity_moment_Nm':axial_tau,
            'minimum_fixed_world_points_COM_shift_world_m':delta.tolist(),
            'minimum_fixed_world_points_COM_shift_palm_m':(Rp.T@delta).tolist(),
            'minimum_fixed_world_points_COM_shift_body_m':(R.T@delta).tolist(),
            'minimum_shift_norm_m':float(np.linalg.norm(delta)),
            'pure_body_Z_shift_to_zero_this_axial_moment_m':body_z_shift,
            'pure_body_Z_shift_palm_m':(Rp.T@(body_z_shift*R[:,2])).tolist(),
            'preserves_actual_contacts':'UNKNOWN; fixed-point algebra is not a valid robot pose certificate'},
        'force_cone_only_necessary_certificate':{'method':'All pairwise ray cross-product dual planes; no LP, optimization or force allocation.',
            'face_planes':faces,'violated_planes':violated,'required_contact_force_body_N':body_up.tolist(),
            'has_strict_separating_plane':bool(violated),'all_ray_interior_direction_body':f1.tolist(),
            'angle_to_all_ray_interior_deg':math.degrees(beta),'first_halfspace_entry_deg':math.degrees(enter),
            'proposed_body_rotation_deg':math.degrees(turn),'phone_active_rotation_axis_palm':(-Rp.T@R@axis).tolist(),
            'post_tilt_required_force_body_N':target_after.tolist(),
            'post_tilt_minimum_original_cone_plane_dot_N':min(float(np.dot(f['normal_body'],target_after)) for f in faces),
            'scope':'Necessary ideal local force condition for the original material-frame rays only; new hand/contact alignment remains unknown.'},
        'old_search':{'translation_lower_palm_m':lo_t.tolist(),'translation_upper_palm_m':hi_t.tolist(),
            'rotvec_lower_rad':lo_r.tolist(),'rotvec_upper_rad':hi_r.tolist(),
            'current_translation_margin_m':np.minimum(np.array(n['phone_t_palm_m'])-lo_t,hi_t-np.array(n['phone_t_palm_m'])).tolist(),
            'current_rotvec_margin_rad':np.minimum(np.array(n['phone_rotvec_rad'])-lo_r,hi_r-np.array(n['phone_rotvec_rad'])).tolist(),
            'interpretation':'Coordinate boxes around each preload seed, not Euler-angle or geodesic search balls; local finite optimizer did not exhaust them.'},
        'four_proposed_seeds':seeds,'budget':{'saved_NPZ_reads':1,'new_robot_model_FK_forward_step_NN_GPU_LP_calls':0},
        'source_GT_world_gravity':'UNKNOWN; only original robot gravity is used',
        'wrist_scene_geometry_scale_mass_friction_changed':False,'human_contact_truth_or_holding_qualified':False}
    assert not any(m in sys.modules for m in ('mujoco','torch','warp','mujoco_warp'))
    (HERE/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'report':pin(HERE/'report.json'),'normal_angle_deg':result['current']['opposing_inward_normal_angle_deg'],
        'strict_force_separators':len(violated),'force_certificate':violated,
        'rotation_deg':math.degrees(turn),'long_slide_m':long_slide,
        'seed_old_boxes':[(s['name'],s['inside_old_translation_trust'],s['inside_old_rotvec_trust']) for s in seeds]}))


if __name__=='__main__':main()
