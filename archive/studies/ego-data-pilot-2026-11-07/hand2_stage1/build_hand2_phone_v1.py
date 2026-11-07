"""Fixed existing Hand2 + unscaled phone staging, CPU compile/forward only.

No Env, training, integration step, hand optimization or human22->robot20 map.
New files only under hand2_stage1; installed factories/resources remain intact.
"""
from pathlib import Path
import hashlib,importlib.metadata,json,os,re,sys,tarfile,time,xml.etree.ElementTree as ET
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent
DEX=Path('D:/workspace/dex-supermimic-research')
PHONE=PILOT/'asset_stage2_convex/completed_v2'
CLIP=PILOT/'hot3d/train_aria/clip-001856.tar'
sys.path.insert(0,str(DEX/'src'))

def require(v,s):
    if not v:raise ValueError(s)

def pin(path):
    path=Path(path).resolve(strict=True);return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def vec(values):return ' '.join(format(float(x),'.17g') for x in values)

def T(R,p,np):
    a=np.eye(4);a[:3,:3]=R;a[:3,3]=p;return a

def hand_bounds(model,data,palm_id,mujoco,np):
    """Conservative finite collision-geometry enclosure in the unchanged palm."""
    R=data.xmat[palm_id].reshape(3,3);p=data.xpos[palm_id];all_points=[];rows=[]
    from itertools import product
    for gid in range(model.ngeom):
        if not(model.geom_contype[gid] or model.geom_conaffinity[gid]):continue
        kind=int(model.geom_type[gid]);size=model.geom_size[gid];G=data.geom_xmat[gid].reshape(3,3);gp=data.geom_xpos[gid]
        if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
            mid=int(model.geom_dataid[gid]);start=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);local=model.mesh_vert[start:start+n].astype(np.float64)
        else:
            if kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE):half=np.array([size[0],size[0],size[1]+size[0]])
            elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):half=np.array([size[0],size[0],size[1]])
            elif kind==int(mujoco.mjtGeom.mjGEOM_SPHERE):half=np.full(3,size[0])
            elif kind==int(mujoco.mjtGeom.mjGEOM_BOX):half=size.copy()
            else:raise ValueError('Unsupported source collision primitive; no guessed placement enclosure')
            local=np.array(list(product(*[(-v,v) for v in half])))
        world=local@G.T+gp;palm=(world-p)@R;all_points.append(palm)
        rows.append({'geom_id':gid,'geom':model.geom(gid).name,'type':kind,'bounds_palm_m':[palm.min(0).tolist(),palm.max(0).tolist()]})
    require(all_points,'Nonempty source hand collision bounds')
    points=np.concatenate(all_points);return np.array([points.min(0),points.max(0)]),rows

def q_initial(model,init,np):
    q=model.qpos0.copy()
    for jid in range(model.njnt):
        name=model.joint(jid).name;matched=[value for pattern,value in init.joint_pos.items() if re.fullmatch(pattern,name)]
        require(len(matched)==1 and int(model.jnt_type[jid])==3,'One source initial value per existing hinge')
        q[int(model.jnt_qposadr[jid])]=matched[0]
    return q

def state_tables(model,data,palm_id,mujoco,np):
    Rp=data.xmat[palm_id].reshape(3,3);pp=data.xpos[palm_id];joints=[];actuators=[];tips=[]
    for aid in range(model.nu):
        row={'id':aid,'name':model.actuator(aid).name,'joint_id':int(model.actuator_trnid[aid,0]),'transmission_type':int(model.actuator_trntype[aid]),'ctrl_limited':bool(model.actuator_ctrllimited[aid]),'ctrl_range':model.actuator_ctrlrange[aid].tolist(),'force_limited':bool(model.actuator_forcelimited[aid]),'force_range':model.actuator_forcerange[aid].tolist(),'gainprm':model.actuator_gainprm[aid].tolist(),'biasprm':model.actuator_biasprm[aid].tolist(),'gear':model.actuator_gear[aid].tolist()};actuators.append(row)
    for jid in range(model.njnt):
        name=model.joint(jid).name
        if not name.startswith('right_finger'):continue
        lo,hi=model.jnt_range[jid];center=(lo+hi)/2;soft_half=.9*(hi-lo)/2
        joints.append({'id':jid,'name':name,'qpos_adr':int(model.jnt_qposadr[jid]),'dof_adr':int(model.jnt_dofadr[jid]),'limited':bool(model.jnt_limited[jid]),'hard_range_rad':[float(lo),float(hi)],'source_soft_range_factor':.9,'derived_soft_range_rad':[float(center-soft_half),float(center+soft_half)],'initial_q_rad':float(data.qpos[model.jnt_qposadr[jid]]),'axis_local':model.jnt_axis[jid].tolist(),'actuator_ids':[r['id'] for r in actuators if r['joint_id']==jid],'joint_actuator_force_limited':bool(model.jnt_actfrclimited[jid]),'joint_actuator_force_range_Nm':model.jnt_actfrcrange[jid].tolist()})
    for finger in range(1,6):
        name=f'right_finger{finger}_tip';sid=model.site(name).id;world=data.site_xpos[sid].copy()
        tips.append({'finger':finger,'name':name,'site_id':sid,'world_m':world.tolist(),'palm_m':((world-pp)@Rp).tolist(),'meaning':'Existing robot tip site, not an optimized contact point or human landmark'})
    return joints,actuators,tips

def extract_reference():
    rows=[]
    with tarfile.open(CLIP,'r') as tar:
        def load(name):return json.load(tar.extractfile(name))
        shape=load('__hand_shapes.json__')
        for frame in range(150):
            key=f'{frame:06d}';info=load(key+'.info.json');hands=load(key+'.hands.json');objects=load(key+'.objects.json');left=hands.get('left')
            require(left is not None and len(left['umetrack_pose']['joint_angles'])==22,'All actual left22 values retained, never slice to robot20')
            phones=[{'object_bop_id':obj['object_bop_id'],'object_uid':obj['object_uid'],'T_world_from_object':obj['T_world_from_object']} for obj in objects.get('24',[])]
            rows.append({'frame_id':frame,'info':info,'left_umetrack_pose':left['umetrack_pose'],'left_mano_pose':left['mano_pose'],'phone24':phones})
    write(ROOT/'ego1856_left_reference.json',{'schema':'hot3d1856_raw_left_reference_not_robot_retargeting:1','clip_pin':pin(CLIP),'frames':rows,'joint_angles_per_frame':22,'human_hand':'left','robot_asset_hand':'right','left_to_right_mapping':'PENDING_SEPARATE_EXPLICIT_GEOMETRIC_CONVENTION; no implicit reflection or joint truncation','angles_are_not_robot_joint_targets':True,'time_unit':'ns','wrist_translation_unit':'m','source_shape_profile_rest_positions_unit':'mm; separate from world translations','reference_does_not_initialize_robot':True})
    write(ROOT/'ego1856_hand_shapes_raw.json',shape)

def main():
    started=time.perf_counter();require(not (ROOT/'report.json').exists(),'Fresh hand2 stage only')
    ROOT.mkdir(exist_ok=True);(ROOT/'meshes/hand2').mkdir(parents=True,exist_ok=True)
    import numpy as np
    import mujoco
    from dex_supermimic.assets import wuji_hand2 as base
    from dex_supermimic.tasks.pen_spin.config.wuji_hand2 import hand2_constants as constants
    source_paths={'builder':Path(__file__),'hand_factory':Path(base.__file__),'identified_factory':Path(constants.__file__),'upstream_MJCF':base.wuji_hand2_xml_path(),'phone_MJCF':PHONE/'phone_asset.xml','phone_geometry_report':PHONE/'report.json','phone_visual_OBJ':PHONE/'meshes/phone24_visual_m.obj','phone_collision_OBJ':PHONE/'meshes/phone24_collision_convex_m.obj','ego_clip':CLIP}
    sources={name:pin(path) for name,path in source_paths.items()}
    cfg=constants.get_pen_spin_hand2_cfg();spec=cfg.spec_fn()
    for patch in cfg.geoms:patch.edit_spec(spec)
    palm=spec.body('right_palm_link');palm.pos=np.asarray(cfg.init_state.pos);palm.quat=np.asarray(cfg.init_state.rot)
    require(not list(palm.joints) and not palm.mocap,'Existing palm is fixed, no new free/mocap wrist')
    original_world_geoms=[{'name':g.name,'type':int(g.type),'contype':g.contype,'conaffinity':g.conaffinity} for g in spec.worldbody.geoms]
    require(not any(g['contype'] or g['conaffinity'] for g in original_world_geoms),'No hidden existing collidable table/ground; refuse duplicate support')
    require(not any('table' in g.name.lower() or 'jig' in g.name.lower() for g in spec.geoms),'No calibration table/jig in selected pen-spin raw factory')
    hand_model=spec.compile();hand_data=mujoco.MjData(hand_model);hand_data.qpos[:]=q_initial(hand_model,cfg.init_state,np);hand_data.qvel[:]=0;hand_data.ctrl[:]=0;mujoco.mj_forward(hand_model,hand_data)
    palm_id=hand_model.body('right_palm_link').id;bounds,bound_rows=hand_bounds(hand_model,hand_data,palm_id,mujoco,np)
    joints_before,actuators_before,tips_before=state_tables(hand_model,hand_data,palm_id,mujoco,np)
    require(len(joints_before)==hand_model.nu==20 and all(not j['limited'] or j['hard_range_rad'][0]<=j['initial_q_rad']<=j['hard_range_rad'][1] for j in joints_before),'Source20 initial hard limits')
    phone_report=read_json= json.loads((PHONE/'report.json').read_text(encoding='utf-8'));prior=phone_report['preserved_stage1']['assumptions'];center=np.asarray(prior['COM_model_m'],np.float64)
    half_y=(phone_report['hull']['bounds_model_m'][1][1]-phone_report['hull']['bounds_model_m'][0][1])/2
    desired_COM_palm=(bounds[0]+bounds[1])/2;desired_COM_palm[1]=bounds[1,1]+.02+half_y
    local_phone_root=desired_COM_palm-center
    Rp=hand_data.xmat[palm_id].reshape(3,3).copy();pp=hand_data.xpos[palm_id].copy();phone_pos=pp+Rp@local_phone_root;phone_quat=hand_data.xquat[palm_id].copy()
    # Materialize a portable MJCF without changing factory resource files.
    xml=ET.fromstring(spec.to_xml());compiler=xml.find('compiler');compiler.set('meshdir','');assets=spec.assets;copied=[]
    for mesh in xml.findall('./asset/mesh'):
        filename=Path(mesh.get('file').replace('\\','/')).name;matches=[value for key,value in assets.items() if Path(key.replace('\\','/')).name==filename]
        require(matches and all(value==matches[0] for value in matches),'Unambiguous VFS source bytes '+filename)
        destination=ROOT/'meshes/hand2'/filename
        if not destination.exists():destination.write_bytes(matches[0])
        mesh.set('file','meshes/hand2/'+filename);copied.append(pin(destination))
    phone_xml=ET.parse(PHONE/'phone_asset.xml').getroot()
    for mesh in phone_xml.findall('./asset/mesh'):
        filename=Path(mesh.get('file')).name;destination=ROOT/'meshes'/filename;destination.write_bytes((PHONE/'meshes'/filename).read_bytes());mesh.set('file','meshes/'+filename);xml.find('asset').append(mesh)
    phone_body=phone_xml.find('./worldbody/body');phone_body.set('pos',vec(phone_pos));phone_body.set('quat',vec(phone_quat));xml.find('worldbody').append(phone_body)
    ET.SubElement(xml.find('worldbody'),'geom',name='stage1_single_table',type='box',size='.35 .35 .025',pos='0 0 -.025',contype='1',conaffinity='1',friction='.6 .005 .0001',condim='4',solref='.02 1',solimp='.9 .95 .001 .5 2',rgba='.5 .55 .6 1')
    option=xml.find('option')
    if option is None:option=ET.SubElement(xml,'option')
    for key,value in phone_xml.find('option').attrib.items():option.set(key,value)
    xml.set('model','fixed_existing_hand2_phone_stage1');ET.indent(xml,space='  ')
    scene=ROOT/'hand2_phone_scene.xml'
    with scene.open('x',encoding='utf-8') as f:f.write('<?xml version="1.0" encoding="utf-8"?>\n'+ET.tostring(xml,encoding='unicode')+'\n')
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    data.qpos[:20]=hand_data.qpos;data.qvel[:]=0;data.ctrl[:]=0;mujoco.mj_forward(model,data)
    pid=model.body('right_palm_link').id;oid=model.body('hot3d_phone24').id;table_id=model.geom('stage1_single_table').id
    require(int(model.body_jntnum[pid])==0 and int(model.body_mocapid[pid])==-1 and np.array_equal(data.xpos[pid],hand_data.xpos[palm_id]) and np.array_equal(data.xquat[pid],hand_data.xquat[palm_id]),'Fixed calibrated wrist unchanged after scene assembly')
    joints,actuators,tips=state_tables(model,data,pid,mujoco,np)
    require(joints==joints_before and actuators==actuators_before,'Original named joint/actuator hard limits, torque ranges and gains unchanged')
    require(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.xpos).all() and np.isfinite(data.actuator_force).all(),'Finite zero-step forward')
    contacts=[]
    for k in range(data.ncon):
        contact=data.contact[k];a,b=(int(x) for x in contact.geom);names=[model.geom(g).name for g in (a,b)]
        category='phone_hand' if (model.geom_bodyid[a]==oid or model.geom_bodyid[b]==oid) and table_id not in (a,b) else 'phone_table' if (model.geom_bodyid[a]==oid or model.geom_bodyid[b]==oid) else 'hand_table' if table_id in (a,b) else 'hand_self'
        contacts.append({'index':k,'geom_ids':[a,b],'geom_names':names,'category':category,'distance_m':float(contact.dist),'position_world_m':contact.pos.tolist(),'generated_contact_not_global_clearance':True})
    require(not any(c['category'] in ('phone_hand','phone_table','hand_table') and c['distance_m']<0 for c in contacts),'Staging scene must not begin with phone/table penetration; do not repair with scaling or wrist changes')
    np.savez_compressed(ROOT/'zero_step_state.npz',qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy(),body_xpos=data.xpos.copy(),body_xquat=data.xquat.copy(),site_xpos=data.site_xpos.copy(),T_world_from_palm=T(Rp,pp,np),T_palm_from_phone_model=T(np.eye(3),local_phone_root,np),T_world_from_phone_model=T(Rp,phone_pos,np))
    write(ROOT/'joint_actuator_interface.json',{'joint_order':[j['name'] for j in joints],'joints':joints,'actuators':actuators,'hand_qpos_adr':list(range(20)),'hand_dof_adr':list(range(20)),'phone_freejoint':{'name':'phone24_freejoint','qpos_adr':int(model.jnt_qposadr[model.joint('phone24_freejoint').id]),'dof_adr':int(model.jnt_dofadr[model.joint('phone24_freejoint').id])}})
    write(ROOT/'palm_keypoints_and_placement.json',{'palm_body':'right_palm_link','fixed_source_wrist_pos_quat_wxyz':list(constants.RIGHT_FIXED_WRIST_POSE_W),'T_world_from_palm':T(Rp,pp,np).tolist(),'robot_tip_sites':tips,'hand_collision_enclosure_palm_m':bounds.tolist(),'enclosure_source_geoms':bound_rows,'phone_COM_palm_m':desired_COM_palm.tolist(),'T_palm_from_phone_model':T(np.eye(3),local_phone_root,np).tolist(),'T_world_from_phone_model':T(Rp,phone_pos,np).tolist(),'placement_rule':'Keep original fixed wrist and original phone scale/frame. Stage phone lower localY support plane20mm beyond the complete hand collision enclosure; X/Z COM at enclosure center. No fitted/held/grasp claim.','phone_model_COM_offset_m':center.tolist(),'table_top_world_z_m':0.})
    extract_reference()
    for value in sources.values():require(pin(value['path'])==value,'Frozen source changed during build')
    cuda=False
    if 'torch' in sys.modules:cuda=sys.modules['torch'].cuda.is_initialized()
    require(not cuda,'No CUDA initialization')
    report={'schema':'fixed_hand2_phone_stage1_cpu_zero_step_preparation:1','status':'COMPILED_AND_ZERO_STEP_FORWARD_NOT_GRASP_OR_DYNAMICS','source_pins':sources,'software':{'mujoco':mujoco.__version__,'wuji-description':importlib.metadata.version('wuji-description')},'factory':'get_pen_spin_hand2_cfg().spec_fn(); apply existing cfg.geoms.edit_spec, existing identified actuator parameters and capsule overlay','counts':{'CPU_MjModel_compiles':2,'MjData_instances':2,'mj_forward_calls':2,'mj_step_integration_Env_NN_training_GPU':0},'model':{'nq':model.nq,'nv':model.nv,'nu':model.nu,'nbody':model.nbody,'ngeom':model.ngeom,'copied_hand_resource_files':len(copied)},'existing_world_geoms_before_addition':original_world_geoms,'existing_table_or_jig_count':0,'new_table_count':1,'fixed_palm_has_joint':False,'hand_limits_actuator_values_unchanged':True,'phone_visual_unscaled':True,'phone_mass_kg':float(model.body_mass[oid]),'phone_raw_sliding_friction':float(model.geom_friction[model.geom('phone24_collision_convex').id,0]),'phone_mass_inertia_and_friction_are_assumptions':True,'HO_effective_friction_caveat':'Existing hand priority1/friction0.7 overrides phone priority0/raw0.6 on hand-phone pairs. No calibrated effective friction or new contact policy is claimed.','initial_contacts':contacts,'generated_empty_contact_does_not_prove_global_clearance':True,'CUDA_initialized':cuda,'unused_left_to_right_mapping_pending':True,'physical_holding_retargeting_or_training_qualified':False,'elapsed_seconds':time.perf_counter()-started,'output_pins':{p.relative_to(ROOT).as_posix():pin(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}}
    write(ROOT/'report.json',report);print(json.dumps({'status':report['status'],'report':pin(ROOT/'report.json'),'scene':pin(scene),'nq':model.nq,'nv':model.nv,'nu':model.nu,'initial_contact_count':len(contacts)}));return 0

if __name__=='__main__':raise SystemExit(main())
