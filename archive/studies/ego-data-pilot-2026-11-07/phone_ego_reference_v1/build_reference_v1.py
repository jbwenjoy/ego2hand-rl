"""Pure raw-pose61 reference export and frame/time fixtures; no native/NN calls."""
from pathlib import Path
import hashlib,json
import numpy as np
import reference as ref

H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(r):assert pin(r['path'])==r;return Path(r['path'])
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    out=H/'completed_v1';protocol=H/'protocol_v1.json';assert not out.exists() and not protocol.exists()
    pr=dict(schema='raw_hot3d61_fixed_palm_object_intent_protocol:1',sources={'builder':pin(__file__),'reference':pin(ref.__file__),'codec':pin(ref.CODEC)},
        inputs={'raw_poses':pin(P/'ego_hand_geometry_v1/raw_profile_and_poses.json'),'frame_identities':pin(P/'ego_hand_geometry_v1/frame_identities.json'),
            'anchor_report':pin(P/'functional_tripod_loaded_entry_v1/completed_run1/report.json'),'physical_parameters':pin(P/'phone_object_wrench_labels_v1/completed_v1/physical_parameters.json')},
        source_frames=[42,102],source_hand='left',robot_hand='right',pose_convention='F64 normalized raw quaternions, proper SE3; no matrix projection or reflection',
        relative_rule='T_wo=inv(T_world_wrist) T_world_phone; delta=inv(T_wo[0]) T_wo[k]; robot_T=robot_initial_T delta',
        time_rule='Original integer ref_timestamp_ns differences; preserve61 source samples and duration.20ms grid stays inside span, no stretch to2s/extrapolation.',
        robot_anchor_scope='All two compression candidates retained with original geometry flags. Reference intent only; physical admission stays separate.',
        native_NN_training_authorized=False)
    write(protocol,pr);out.mkdir();raw=read(checked(pr['inputs']['raw_poses']))['frames'];ident=read(checked(pr['inputs']['frame_identities']))
    assert len(raw)==len(ident)==61 and [v['frame_id'] for v in raw]==list(range(42,103))
    ns=np.array([v['info']['ref_timestamp_ns'] for v in raw],dtype=np.int64)
    assert np.array_equal(ns,np.array([v['ref_timestamp_ns'] for v in ident])) and np.all(np.diff(ns)>0)
    time_s=(ns-ns[0]).astype(np.float64)*1e-9
    wrist=[];phone=[];norms=[]
    for row in raw:
        w,nw=ref.pose_from_raw(row['left_umetrack_pose']['T_world_from_wrist']);o,no=ref.pose_from_raw(row['phone24']['T_world_from_object'])
        wrist.append(w);phone.append(o);norms.append([nw,no])
    W=np.array(wrist);O=np.array(phone);rel,delta=ref.relative_object_increments(W,O)
    checks={};checks['source_relative_roundtrip']=bool(np.max(abs(W@rel-O))<1e-14)
    checks['delta_first_near_identity_no_overwrite']=bool(np.max(abs(delta[0]-np.eye(4)))<1e-14)
    G=np.eye(4);G[:3,:3]=ref.codec.exp_world(np.array([.4,-.21,.27]));G[:3,3]=[1.1,-.3,.2]
    _,transformed=ref.relative_object_increments(G@W,G@O)
    checks['global_world_change_cancels_in_relative_intent']=bool(np.max(abs(transformed-delta))<1e-14)
    constant_relative=np.broadcast_to(rel[0],rel.shape)
    _,transport=ref.relative_object_increments(W,W@constant_relative)
    checks['pure_wrist_transport_creates_no_object_relative_task']=bool(np.max(abs(transport-np.eye(4)))<1e-14)
    query=np.arange(int(np.floor(time_s[-1]/.02))+1,dtype=np.float64)*.02
    sample=ref.interpolate_increments(time_s,delta,query)
    same=ref.interpolate_increments(time_s,delta,time_s)
    checks['interpolation_hits_all61_source_poses']=bool(np.max(abs(same-delta))<1e-14)
    try:ref.interpolate_increments(time_s,delta,np.array([time_s[-1]+.001]));rejected=False
    except ValueError:rejected=True
    checks['outside_clock_rejected_without_extrapolation']=rejected
    improper=np.eye(4);improper[0,0]=-1
    try:ref.validate_transform(improper);rejected=False
    except ValueError:rejected=True
    checks['improper_handedness_matrix_rejected']=rejected
    pars=read(checked(pr['inputs']['physical_parameters']));ipos=np.array(pars['body_ipos_body_m']);anchors=read(checked(pr['inputs']['anchor_report']))
    outputs={};summary=[]
    for entry in anchors['candidates']:
        c=read(checked(entry['candidate']));q=np.array(c['qpos27_F64']);T0,_=ref.pose_from_raw({'translation_xyz':q[20:23],'quaternion_wxyz':q[23:27]})
        body,COM=ref.reanchor_object_reference(T0,delta,ipos);control_body,control_COM=ref.reanchor_object_reference(T0,sample,ipos)
        motion=ref.interval_motion_descriptors(time_s,body,COM)
        first=float(np.max(abs(body[0]-T0)));checks['anchor_first_'+c['candidate_id']]=first<1e-14
        reconstructed=ref.inverse_rigid(T0)@body;checks['anchor_increment_roundtrip_'+c['candidate_id']]=bool(np.max(abs(reconstructed-delta))<1e-14)
        name=c['candidate_id']+'.npz';path=out/name
        np.savez_compressed(path,source_frame_ids=np.arange(42,103,dtype=np.int64),source_ref_timestamp_ns=ns,source_time_s=time_s,
            source_T_world_from_wrist_F64=W,source_T_world_from_phone_F64=O,source_T_wrist_from_phone_F64=rel,delta_in_initial_phone_F64=delta,
            target_T_world_from_phone_F64=body,target_COM_world_m=COM,nominal_body_ipos_m=ipos,
            interval_mean_COM_velocity_world_m_s=motion['interval_mean_COM_velocity_world_m_s'],interval_geodesic_angular_velocity_world_rad_s=motion['interval_geodesic_angular_velocity_world_rad_s'],
            control_time_s=query,control_T_world_from_phone_F64=control_body,control_COM_world_m=control_COM)
        outputs[name]=pin(path)
        summary.append(dict(candidate_id=c['candidate_id'],anchor_candidate=entry['candidate'],initial_geometry_qualified=bool(c['geometry']['passed']),
            compression_target_pass=bool(c['compression_target_pass']),source_samples=61,control_samples=len(query),
            maximum_COM_excursion_from_first_m=float(np.linalg.norm(COM-COM[0],axis=1).max()),
            maximum_orientation_excursion_rad=float(max(np.linalg.norm(ref.codec.log_world(v[:3,:3]@body[0,:3,:3].T)) for v in body)),
            actual_control_clock_terminal_s=float(query[-1]),original_source_terminal_s=float(time_s[-1]),
            reference_task_is_executed_or_physics_feasible=False,hand_joint_reference_defined=False,gravity_registered=False))
    rel_dp=float(np.linalg.norm(rel[:,:3,3]-rel[0,:3,3],axis=1).max());rel_angle=float(max(np.linalg.norm(ref.codec.log_world(v[:3,:3]@rel[0,:3,:3].T)) for v in rel))
    report=dict(schema='raw_hot3d61_fixed_palm_object_intent_result:1',status='PASS' if all(checks.values()) else 'FAILED_REFERENCE_CONTRACT',protocol=pin(protocol),
        checks=checks,passed=sum(bool(v) for v in checks.values()),total=len(checks),source_frames=61,source_duration_s=float(time_s[-1]),source_timestep_minmax_s=[float(np.diff(time_s).min()),float(np.diff(time_s).max())],
        source_raw_quaternion_norm_max_deviation=float(np.max(abs(np.array(norms)-1))),source_relative_origin_excursion_m=rel_dp,source_relative_SO3_excursion_rad=rel_angle,
        anchor_summaries=summary,outputs=outputs,counts=dict(raw_pose_JSON_reads=1,identity_JSON_reads=1,anchor_report_reads=1,physical_parameter_JSON_reads=1,robot_candidate_JSON_reads=len(summary),
            original_geometry_NPZ_reads=0,native_model_FK_forward_steps_NN_AD_training_calls=0),
        scope='Real HOT3D pose intent only, reanchored to named functional robot starts. No RGB reconstruction, source gravity/material calibration, humanhand/rightrobot morphology solution, contact correspondence/forces, robotjointreference, physical tracking or holding claim. Original61timestamps preserved, not rescaled to2s.')
    write(out/'report.json',report);print(json.dumps(dict(report=pin(out/'report.json'),passed=report['passed'],total=report['total'],source_duration_s=report['source_duration_s'])))
    assert all(checks.values())
if __name__=='__main__':main()
