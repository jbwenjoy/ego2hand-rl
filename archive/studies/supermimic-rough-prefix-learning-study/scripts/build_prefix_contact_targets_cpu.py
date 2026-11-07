"""SavedR16 geometry -> declared-intent target; no native query/rollout/solve."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,json,sys,re
import numpy as np
from scipy.spatial.transform import Rotation
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from rough_prefix.io import sha,pin,checked,read,write,loadnp
from rough_prefix.contact_binding import VARIANT,KNOWN_SEMANTICS,HYPER,code_hashes
from frozen_contact_math.contacts import wrench_support,sample_basis

RROOT=Path('D:/workspace/supermimic-observation-restoration-prefix-study')
CANDIDATE=RROOT/'outputs/restoration_filter_prefix_v1/candidate.npz'
CANDIDATE_SHA='6364a6e262f7fbedac049d2d9e789b2c9077cbd6ef7656ef2e9b394161bab81a'
CONTACTS=RROOT/'outputs/restoration_filter_prefix_v1/native_contacts.json'
ANCHORS=Path('D:/workspace/supermimic-observation-contact-intent-study/outputs/frame0_h123_seeded_baseline_v3/frozen_anchor_bindings.json')
LEGACY_BASIS=Path('D:/workspace/supermimic-chord-study/outputs/reference_contact_targets_v1/contact_targets.npz')


def windows(labels,t):
    edge=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
    return [{'phase':int(labels[a]),'begin_frame':int(a),'end_frame_exclusive':int(b),'time_first_s':float(t[a]),'time_last_s':float(t[b-1])} for a,b in zip(edge[:-1],edge[1:])]


def build(reference,output):
    import mujoco
    forbidden={name:0 for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_collision','mj_geomDistance','mj_contactForce','mj_kinematics','mj_comPos')};original={k:getattr(mujoco,k) for k in forbidden}
    def deny(k):
        def fail(*a,**kw):forbidden[k]+=1;raise AssertionError('Saved-target producer forbids '+k)
        return fail
    for k in original:setattr(mujoco,k,deny(k))
    try:
        output.mkdir(parents=True,exist_ok=False);meta=read(reference.with_suffix('.json'));ref=loadnp(reference);candidate=loadnp(CANDIDATE);raw=read(CONTACTS);anchors=read(ANCHORS)
        if sha(CANDIDATE)!=CANDIDATE_SHA or str(candidate['retarget_variant_id'])!=VARIANT or meta['retarget_variant_id']!=VARIANT or meta['source_window_half_open']!=[200,216]:raise ValueError('Wrong actualR/new16 variant')
        if meta['human_contact_observation_known'] is not False or meta['source_robot_labels_used'] is not False or anchors['hypothesis_id']!='H123':raise ValueError('Only declared H123 hypothesis, no observed human labels')
        for v in meta['source_pins'].values():checked(v)
        if set(ref)!={'time_s','joint_pos_rad','object_pose_w','wrist_pose_w'} or not np.array_equal(ref['time_s'],np.arange(16)*.02) or not np.array_equal(ref['joint_pos_rad'],candidate['q_goal_rad']) or not np.array_equal(ref['object_pose_w'],candidate['mapped_object_pose_w']):raise ValueError('Newbridge must match exactR16 q/object')
        source=meta['source_pins'];modelpin=source['export_source__model'];ctxpin=source['export_source__actual_context'];model=mujoco.MjModel.from_binary_path(str(checked(modelpin)));obj=model.body('object/cube').id;og=set(np.flatnonzero(model.geom_bodyid==obj).tolist());hand={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])}
        com=model.body_ipos[obj].copy();radius=max(float(np.linalg.norm(model.geom_pos[g]-com)+model.geom_rbound[g]) for g in og)
        if abs(radius-meta['object_radius_m'])>1e-12 or not np.array_equal(com,meta['object_com_offset_o_m']):raise ValueError('Frozen nominal radius/COM identity')
        R=Rotation.from_quat(ref['object_pose_w'][:,[4,5,6,3]]).as_matrix();frames=[];known=np.zeros(16,bool);finger=np.zeros((16,6),bool);jids=np.array([model.joint('robot/'+name).id for name in meta['joint_names']]);lo,hi=model.jnt_range[jids].T
        for i in range(16):
            contacts=[];reasons=[];allcontacts=[c for c in raw['fit16'] if c['frame']==i]
            if any(c['phase']!='native_CPU_geometry_not_solved_force' for c in allcontacts):raise ValueError('Only saved unsolved reference geometry, not rollout forces')
            if any(max(0.,-c['distance_m'])>.003 for c in allcontacts):reasons.append('saved_generated_overlap_over3mm')
            if np.any(ref['joint_pos_rad'][i]<lo) or np.any(ref['joint_pos_rad'][i]>hi):reasons.append('q_goal_outside_original_hard_range')
            if any(c['category']=='object_environment' and c['distance_m']<=0 for c in allcontacts):reasons.append('saved_object_environment_contact')
            for c in allcontacts:
                if c['category']!='hand_object':continue
                g0,g1=c['geom'];other=g0 if g1 in og else g1
                if (g0 in og)==(g1 in og) or other not in hand or [model.geom(g0).name,model.geom(g1).name]!=c['names']:raise ValueError('Saved contact geometry mapping differs')
                sign=1. if g1 in og else -1.;nw=sign*np.asarray(c['frame_rows_geom0_to_geom1_w'])[0];pw=np.asarray(c['position_w']);po=R[i].T@(pw-ref['object_pose_w'][i,:3]);no=R[i].T@nw
                if not np.isfinite(np.r_[pw,nw,po,no,c['distance_m']]).all() or abs(np.linalg.norm(no)-1)>1e-6:reasons.append('invalid_native_normal_or_point')
                fmatch=re.search(r'right_finger([1-5])_',model.geom(other).name);f=int(fmatch.group(1)) if fmatch else 0
                item={'point_w_m':pw.tolist(),'point_o_m':po.tolist(),'point_com_o_m':(po-com).tolist(),'normal_inward_w':nw.tolist(),'normal_inward_o':no.tolist(),'signed_distance_m':float(c['distance_m']),'geometry_contact':bool(c['distance_m']<=0),'geom_pair':[g0,g1],'finger_id':f,'source_native_contact_index':c['contact_index']}
                contacts.append(item)
            # Native absence gives no human release or positive-gap certificate.
            if not any(c['geometry_contact'] for c in contacts):reasons.append('no_saved_declared_contact_available_UNKNOWN_not_release')
            known[i]=not reasons
            if known[i]:
                for c in contacts:
                    if c['geometry_contact']:finger[i,c['finger_id']]=True
            frames.append({'frame':i,'time_s':float(ref['time_s'][i]),'declared_geometry_intent_known':bool(known[i]),'human_contact_observation_known':False,'confidence':None,'unknown_reasons':reasons,'contacts':contacts})
        K=max(map(lambda f:len(f['contacts']),frames));z=lambda tail=(),dtype=float:np.zeros((16,K)+tail,dtype=dtype)
        arrays={'time_s':ref['time_s'],'input_joint_pos_rad':ref['joint_pos_rad'],'input_object_pose_w':ref['object_pose_w'],'input_wrist_pose_w':ref['wrist_pose_w'],'object_com_offset_o_m':com,'contact_pos_w_m':z((3,)),'contact_pos_o_m':z((3,)),'contact_pos_com_o_m':z((3,)),'contact_normal_inward_w':z((3,)),'contact_normal_o':z((3,)),'raw_contact_present':z(dtype=bool),'raw_geometry_contact':z(dtype=bool),'raw_signed_distance_m':z(),'raw_geom_pair':np.full((16,K,2),-1,np.int32),'finger_id':np.full((16,K),-1,np.int32),'contact_known':known,'known_no_contact':np.zeros(16,bool),'human_contact_observation_known':np.zeros(16,bool),'contact_active':z(dtype=bool),'contact_slot_known':z(dtype=bool),'finger_active':finger,'confidence_known':np.zeros(16,bool)}
        for i,frame in enumerate(frames):
            for j,c in enumerate(frame['contacts']):
                for key,sourcekey in [('contact_pos_w_m','point_w_m'),('contact_pos_o_m','point_o_m'),('contact_pos_com_o_m','point_com_o_m'),('contact_normal_inward_w','normal_inward_w'),('contact_normal_o','normal_inward_o'),('raw_signed_distance_m','signed_distance_m'),('raw_geom_pair','geom_pair'),('finger_id','finger_id')]:arrays[key][i,j]=c[sourcekey]
                arrays['raw_contact_present'][i,j]=True;arrays['raw_geometry_contact'][i,j]=c['geometry_contact'];arrays['contact_slot_known'][i,j]=known[i];arrays['contact_active'][i,j]=known[i] and c['geometry_contact']
        n=R[:,:,0];speed=np.rad2deg(np.arccos(np.clip(np.sum(n[1:]*n[:-1],axis=1),-1,1)))/.02;speed=np.r_[speed[0],speed];signature=(finger.astype(np.int64)*(1<<np.arange(6))).sum(1);phase=np.where(known,np.where(speed<15,2,3),0).astype(np.int32)
        for k in np.flatnonzero(known[1:]&known[:-1]&(signature[1:]!=signature[:-1]))+1:
            phase[[k-1,k]]=4
        arrays.update(phase=phase,contact_finger_signature=signature,reference_shaft_speed_deg_s=speed)
        basis=sample_basis(512,0,dtype=torch.float64).numpy()
        with np.load(LEGACY_BASIS,allow_pickle=False) as old:
            if not np.array_equal(basis,old['support_basis']):raise ValueError('Same512 basis differs')
        support=wrench_support(torch.tensor(arrays['contact_pos_com_o_m']),torch.tensor(arrays['contact_normal_o']),torch.tensor(arrays['contact_active']),torch.tensor(basis),radius,mu=.1,edges=8).numpy();arrays.update(support_basis=basis,support512=support[:,None,None,:])
        np.savez_compressed(output/'contact_targets.npz',**arrays);write(output/'raw_geometry.json',frames)
        rules={'schema':'rough_prefix_contact_intent_rules:1','known_semantics':KNOWN_SEMANTICS,'human_contact_observation_known':False,'source':'SavedRfit16 native generatedcontacts plus frozenH123declaredhypothesis, no newquery','contact_distance_threshold_m':0.,'inconsistent_frame_depth_m':.003,'known_no_contact':'None. No explicit release annotation; native absence remainsUNKNOWN.','missing_finger_contact':'Not a prohibition on additional stable finger contact; score is global union support.','padding':'Zero raw distance/point in padded slots is a storage sentinel with raw_contact_present=false, never measured zero gap. Original missing-pairNaNs remain in pinned6364candidate.','support':{'basis512_seed':0,'mu':.1,'edges':8,'normal_ray':True,'unit_force_rays':True,'moment_origin':'objectCOM','radius_divisions':1,'no_force_amplitude_or_torsion_target':True},'phases':{'0':'unknown','1':'explicitknown_no_contact_notpresent','2':'declaredcontact_geometricshaftspeed<15deg_s','3':'declaredcontact_geometricshaftspeed>=15deg_s','4':'neighboringdeclaredfinger-signaturechange'},'phase_weights':{'0':0,'1':0,'2':1,'3':1,'4':.5},'human_confidence':None,'proxy_qualification':'FAILED_RETAINED_EXPLORATORY_ONLY','compiled_precision_qualification':'FAILED_RETAINED_NOT_REVALIDATED'}
        write(output/'rules.json',rules);write(output/'time_windows.json',{'phase':windows(phase,ref['time_s'])})
        compat={'schema':'supermimic_chord_contact_target_manifest:1','source_kind':'kinematic_reference_geometry','known_semantics':KNOWN_SEMANTICS,'reference_npz_file':str(reference.resolve()),'reference_npz_sha256':sha(reference),'reference_json_sha256':sha(reference.with_suffix('.json')),'model_file':modelpin['path'],'model_sha256':modelpin['sha256'],'target_file':str(output/'contact_targets.npz'),'target_npz_sha256':sha(output/'contact_targets.npz'),'rules_sha256':sha(output/'rules.json'),'raw_geometry_sha256':sha(output/'raw_geometry.json'),'time_windows_sha256':sha(output/'time_windows.json'),'target_changes_frozen_reference':False,'target_defined_from_policy_contacts':False,'no_real_contact_force_claim':True,'human_contact_observation_known':False}
        write(output/'manifest.json',compat)
        source_pins={'candidate':pin(CANDIDATE),'saved_R_native_geometry':pin(CONTACTS),'H123_hypothesis':pin(ANCHORS),'reference':pin(reference),'reference_json':pin(reference.with_suffix('.json')),'model':modelpin,'actual_context':ctxpin,'producer':pin(__file__),'legacy_basis_only':pin(LEGACY_BASIS)}
        for name,file in [('objective','contact_objective.py'),('geometry','contact_geometry.py'),('contacts','contacts.py'),('penalty_wrapper','penalty_only_objective.py')]:source_pins[name+'_math']=pin(ROOT/'vendor/frozen_contact_math'/file)
        binding={'schema':'rough_prefix_contact_intent_binding:1','retarget_variant_id':VARIANT,'source_window_half_open':[200,216],'raw_source_identity':meta['raw_source_identity'],'known_semantics':KNOWN_SEMANTICS,'human_contact_observation_known':False,'target_defined_from_policy_contacts':False,'local_proxy_qualification':'FAILED_RETAINED_EXPLORATORY_ONLY','source_kind':'derived_robot_kinematic_contact_intent','reference':pin(reference),'reference_json':pin(reference.with_suffix('.json')),'model':modelpin,'actual_context':ctxpin,'source_pins':source_pins,'known_frames':int(known.sum()),'known_no_contact_frames':0,'unknown_frames':int((~known).sum()),'phase_counts':{str(k):int(np.sum(phase==k)) for k in range(5)},'support_experiment_degenerate_to_penalty_only':not bool(known.any()),'basis_read_log':{'legacy_archive_fields':['support_basis'],'old151_support_orcontacts_consumed':False},'geometry_queries':0,'optimizer_calls':0,'physics_steps':0,'GPU_used':False}
        write(output/'binding_manifest.json',binding)
        expected={key:compat[key] for key in ('target_npz_sha256','reference_npz_sha256','reference_json_sha256','model_sha256','rules_sha256','raw_geometry_sha256','time_windows_sha256')};expected.update(manifest_sha256=sha(output/'manifest.json'),binding_manifest_sha256=sha(output/'binding_manifest.json'),actual_context_sha256=ctxpin['sha256'],**code_hashes())
        spec={'schema':'rough_prefix_contact_objective_spec:1','bundle_dir':str(output),'reference_file':str(reference.resolve()),'model_file':modelpin['path'],'actual_context_file':ctxpin['path'],'binding_manifest':pin(output/'binding_manifest.json'),'hyperparameters':HYPER,'expected_hashes':expected}
        write(output/'objective_spec.json',spec)
        summary={'status':'SAVED_R16_DECLARED_CONTACT_INTENT_BUILT','binding_manifest':pin(output/'binding_manifest.json'),'objective_spec':pin(output/'objective_spec.json'),'frames':16,'saved_contact_slots':int(arrays['raw_contact_present'].sum()),'phase_counts':binding['phase_counts'],'declared_known_frames':int(known.sum()),'human_observed_contact_known_frames':0,'unknown_declared_frames':int((~known).sum()),'support_all_positive_perframe':bool((support>0).any(1).all()),'forbidden_native_calls':forbidden,'Torch_CUDA_initialized':torch.cuda.is_initialized(),'new_geometry_queries':0,'physical_reference_qualified':False,'GPU_used':False};write(output/'build_receipt.json',summary);print(json.dumps(summary,indent=2));return summary
    finally:
        for k,value in original.items():setattr(mujoco,k,value)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--reference',type=Path,default=ROOT/'outputs/input_bundle_v1/contact_reference.npz');a.add_argument('--output',type=Path,default=ROOT/'outputs/contact_intent_v1');p=a.parse_args();build(p.reference.resolve(),p.output.resolve())
