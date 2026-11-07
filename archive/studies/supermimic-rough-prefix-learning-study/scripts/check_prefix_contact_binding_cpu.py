"""Actual16 construction/value/gradient and negative tests; CPU/no native query."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,copy,json,random,shutil,sys
import numpy as np
from scipy.spatial.transform import Rotation
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from rough_prefix.io import read,write,loadnp,pin,sha
from rough_prefix.contact_binding import build_contact_objective,build_penalty_only_objective,code_hashes
from frozen_contact_math.contacts import wrench_support
from frozen_contact_math.contact_geometry import weighted_wrench_support


def memory(frames):
    t=torch.as_tensor(frames,dtype=torch.float32).reshape(-1,1)
    return {'start':torch.zeros_like(t),'time':t,'end':torch.full_like(t,16)}


def main(specpath,out):
    import mujoco
    out.mkdir(parents=True,exist_ok=False);spec=read(specpath);checks={}
    forbidden={k:0 for k in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_collision','mj_geomDistance','mj_contactForce','mj_kinematics','mj_comPos')};orig={k:getattr(mujoco,k) for k in forbidden}
    def deny(k):
        def fail(*a,**kw):forbidden[k]+=1;raise AssertionError('CPU contact check forbids '+k)
        return fail
    for k in orig:setattr(mujoco,k,deny(k))
    try:
        py=random.getstate();nr=np.random.get_state();tr=torch.get_rng_state().clone();obj=build_contact_objective(spec,'cpu');pen=build_penalty_only_objective(spec,'cpu')
        checks['factories_preserve_all_CPU_RNG']=py==random.getstate() and np.array_equal(nr[1],np.random.get_state()[1]) and torch.equal(tr,torch.get_rng_state())
        ref=loadnp(spec['reference_file']);meta=read(Path(spec['reference_file']).with_suffix('.json'));targets=loadnp(Path(spec['bundle_dir'])/'contact_targets.npz');R=Rotation.from_quat(ref['object_pose_w'][:,[4,5,6,3]]).as_matrix();Rw=Rotation.from_quat(ref['wrist_pose_w'][:,[4,5,6,3]]).as_matrix();localp=np.einsum('tji,tj->ti',Rw,ref['object_pose_w'][:,:3]-ref['wrist_pose_w'][:,:3]);localR=Rw.transpose(0,2,1)@R;state=np.c_[ref['joint_pos_rad'],np.zeros((16,20)),localp,localR.reshape(16,9),np.zeros((16,6))]
        x=torch.tensor(state,dtype=torch.float32,requires_grad=True);mem=memory(range(16));bonus,terms=obj(x,mem);grad=torch.autograd.grad(bonus.sum(),x)[0]
        checks['actual16_value_and_gradient_finite']=torch.isfinite(bonus).all().item() and torch.isfinite(grad).all().item()
        checks['actual16_state_gradient_nonzero']=bool(grad.abs().max()>0)
        checks['actual16_phase_and_indices']=torch.equal(terms['geometry_intent_reference_frame'],torch.arange(16,dtype=torch.float32)) and torch.equal(terms['geometry_intent_phase'],torch.tensor(targets['phase'],dtype=torch.float32))
        checks['all16_declared_intent_not_human_labels']=int(targets['contact_known'].sum())==16 and not targets['human_contact_observation_known'].any() and not targets['known_no_contact'].any()
        checks['actual_support_experiment_not_Pdegenerate']=not obj.metadata['support_experiment_degenerate_to_penalty_only'] and torch.all(obj.phase_weight==1).item()
        checks['formula_unscaled_no_internal_dt']=torch.equal(bonus,terms['geometry_intent_phase_weight']*terms['geometry_intent_support_match']-.1*terms['geometry_intent_penetration_prior']) and obj.metadata['dt_applied_inside_module'] is False and obj.metadata['caller_dt_s']==.02
        checks['targets_buffers_not_optimized']=not list(obj.parameters()) and all(not a.requires_grad for a in obj.buffers()) and obj.reference_support64.grad is None
        checks['geometry21x64_no_mount']=obj.metadata['sampled_hand_surface_points']==1344 and len(obj.metadata['geometry']['hand_collision_geoms'])==21 and not any('mount' in g['name'].lower() for g in obj.metadata['geometry']['hand_collision_geoms'])
        checks['same512_training_stride8']=torch.equal(obj.reference_support64,obj.reference_support512[...,::8]) and torch.equal(obj.geometry_proxy.basis,obj.basis512[::8])
        checks['actual_context_geometry_copied_and_failed_qualification_retained']=obj.metadata['actual_geometry_host_copy_fields']==['geom_size','geom_pos','geom_friction'] and obj.metadata['actual_context_sha256']==spec['expected_hashes']['actual_context_sha256'] and obj.metadata['local_proxy_qualification']=='FAILED_RETAINED_EXPLORATORY_ONLY' and obj.metadata['dynamics_gradient_qualified'] is False
        pbonus,pterms=pen(x,mem);pg=torch.autograd.grad(pbonus.sum(),x,retain_graph=False)[0]
        y=x.detach().clone().requires_grad_();_,yt=obj(y,mem);exactP=-.1*yt['geometry_intent_penetration_prior'];eg=torch.autograd.grad(exactP.sum(),y)[0]
        checks['P_ONLY_exact_same_P_value_and_state_gradient']=torch.equal(pbonus,exactP) and torch.equal(pg,eg)
        unknown=copy.deepcopy(obj);unknown.phase.zero_();unknown.contact_known.zero_();unknown.phase_weight.zero_();u=x.detach().clone().requires_grad_();ub,ut=unknown(u,mem);ug=torch.autograd.grad(ub.sum(),u)[0]
        checks['allunknown_A2_equals_P_value_gradient']=torch.equal(ub,pbonus) and torch.equal(ug,pg)
        deep=x[:1].detach().clone();sample=obj.geometry_proxy.hand_points_palm(deep[:,:20])[0,0].detach();deep[0,40:43]=sample;deep[0,43:52]=torch.eye(3).flatten();deep.requires_grad_();db,dt=unknown(deep,memory([0]));dg=torch.autograd.grad(db.sum(),deep)[0]
        checks['unknown_attraction0_deep_penalty_still_active']=torch.equal(dt['geometry_intent_phase_weight'],torch.zeros(1)) and dt['geometry_intent_penetration_prior'].item()>0 and db.item()<0 and torch.isfinite(dg).all().item()
        absent=copy.deepcopy(obj);absent.phase.fill_(1);absent.known_no_contact.fill_(True);absent.phase_weight.zero_();ab,at=absent(deep.detach(),memory([0]));checks['synthetic_knownabsence_masks_only_support']=torch.equal(ab,db.detach()) and at['geometry_intent_penetration_prior'].item()>0
        mixed=copy.deepcopy(obj);mixed.phase[:5]=torch.arange(5);mixed.phase_weight[:5]=torch.tensor([0.,0.,1.,1.,.5]);mb,mt=mixed(x[:5].detach(),memory(range(5)))
        checks['synthetic_five_phase_weights']=torch.equal(mt['geometry_intent_phase_weight'],torch.tensor([0.,0.,1.,1.,.5])) and torch.equal(mb,mt['geometry_intent_phase_weight']*mt['geometry_intent_support_match']-.1*mt['geometry_intent_penetration_prior'])
        special=memory([-5,100,0]);special['start']=torch.tensor([[2.],[0.],[5.]]);special['end']=torch.tensor([[16.],[8.],[16.]]);_,st=obj(x[:3].detach(),special);checks['frame_clamp_matches_PenMDP']=torch.equal(st['geometry_intent_reference_frame'],torch.tensor([2.,7.,5.]))
        a=x[:2].detach().clone().requires_grad_();plain,_=obj(a,memory([0,1]));gp=torch.autograd.grad(plain.sum(),a)[0];a2=a.detach().clone().requires_grad_()
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):amp,_=obj(a2,memory([0,1]))
        ga=torch.autograd.grad(amp.sum(),a2)[0];checks['outerCPU_AMP_fullprecision_value_gradient']=torch.equal(plain,amp) and torch.equal(gp,ga)
        with torch.no_grad():full=obj.full_basis_support_for_evaluation(x[:1].detach());sub=obj.geometry_proxy(x[:1].detach(),.001)['support']
        checks['full512_diagnostic_stride64_consistency']=full.shape==(1,512) and torch.allclose(full[:,::8],sub,atol=1e-6,rtol=1e-6)
        points=torch.tensor(targets['contact_pos_com_o_m'],dtype=torch.float64);normals=torch.tensor(targets['contact_normal_o'],dtype=torch.float64);active=torch.tensor(targets['contact_active']);basis=torch.tensor(targets['support_basis'],dtype=torch.float64);radius=meta['object_radius_m']
        recompute=wrench_support(points,normals,active,basis,radius).numpy();checks['target512_from_saved_geometry_recomputed']=np.array_equal(recompute,targets['support512'][:,0,0])
        weighted=weighted_wrench_support(points,normals,active.double(),basis,radius).numpy();checks['binary_weighted_kernel_matches_contact_kernel']=np.allclose(weighted,recompute,atol=1e-12,rtol=1e-12)
        positions=np.einsum('tij,tkj->tki',R,targets['contact_pos_o_m'])+ref['object_pose_w'][:,None,:3];mask=targets['raw_contact_present'];checks['COM_point_and_normal_world_transform_roundtrip']=np.max(np.abs((positions-targets['contact_pos_w_m'])[mask]))<1e-12 and np.array_equal(targets['contact_pos_com_o_m'],targets['contact_pos_o_m']-targets['object_com_offset_o_m'])
        # Pure torch FK body/site comparison to existing native21; no native FK.
        model=mujoco.MjModel.from_binary_path(spec['model_file']);bodypos,bodyR=obj.geometry_proxy.fk(x.detach()[:,:20]);lookup={b:i for i,b in enumerate(obj.geometry_proxy.fk.body_ids)};outpoints=[bodypos[:,0]]
        for f in range(1,6):
            for link in (1,3,4):outpoints.append(bodypos[:,lookup[model.body(f'robot/right_finger{f}_link{link}').id]])
            site=model.site(f'robot/right_finger{f}_tip').id;bid=lookup[int(model.site_bodyid[site])];outpoints.append(bodypos[:,bid]+torch.einsum('tij,j->ti',bodyR[:,bid],torch.tensor(model.site_pos[site],dtype=torch.float32)))
        localpoints=torch.stack(outpoints,1).numpy();world=np.einsum('tij,tkj->tki',Rw,localpoints)+ref['wrist_pose_w'][:,None,:3];candidate=loadnp(Path('D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restoration_filter_prefix_v1/candidate.npz'));fkerr=float(np.max(np.abs(world-candidate['fk_hand_points_w'])));checks['pureTorch_palmFK_matches_saved_native21']=fkerr<3e-7
        altered=copy.deepcopy(spec);altered['expected_hashes']['reference_npz_sha256']='0'*64
        try:build_contact_objective(altered);checks['wrong_reference_hash_rejected']=False
        except ValueError:checks['wrong_reference_hash_rejected']=True
        altered=copy.deepcopy(spec);altered['hyperparameters']['bandwidth_m']=.002
        try:build_contact_objective(altered);checks['hyperparameter_drift_rejected']=False
        except ValueError:checks['hyperparameter_drift_rejected']=True
        altered=copy.deepcopy(spec);altered['expected_hashes']['binding_module_sha256']='0'*64
        try:build_contact_objective(altered);checks['binding_source_drift_rejected']=False
        except ValueError:checks['binding_source_drift_rejected']=True
        checks['zero_new_query_force_physics']=not any(forbidden.values())
        checks['CPU_noCUDA_Warp_no_GPU']=not torch.cuda.is_initialized() and 'warp' not in sys.modules and 'mujoco_warp' not in sys.modules
        np.savez_compressed(out/'actual16_diagnostics.npz',state58=state,bonus=bonus.detach().numpy(),gradient=grad.numpy(),penalty_bonus=pbonus.detach().numpy(),penalty_gradient=pg.numpy(),phase=targets['phase'],**{k:v.detach().numpy() for k,v in terms.items()})
        report={'status':'PASS_REAL16_CONTACT_BINDING_CPU' if all(checks.values()) else 'FAIL_CONTACT_BINDING_CPU','checks':{k:bool(v) for k,v in checks.items()},'passed':sum(bool(v) for v in checks.values()),'total':len(checks),'objective_spec':pin(specpath),'source_hashes':code_hashes(),'checker':pin(__file__),'geometry_metadata':obj.metadata,'native21_vs_torchFK_max_abs_m':fkerr,'actual16_bonus_range':[float(bonus.min()),float(bonus.max())],'actual16_gradient_abs_max':float(grad.abs().max()),'phase_counts':{str(k):int(np.sum(targets['phase']==k)) for k in range(5)},'unknown_all_is_penalty_only':True,'canonical_A2_degenerate_to_P':obj.metadata['support_experiment_degenerate_to_penalty_only'],'synthetic_phase_and_deepstress_cases_not_data_labels':True,'forbidden_native_calls':forbidden,'new_geometry_queries':0,'optimizer_calls':0,'physics_steps':0,'GPU_used':False,'training_or_hook_integration':'NOT_RUN; caller must apply0.02exactlyonce in imagination only; native data/reward path independent','scope':'CPU value/input-gradient/interface evidence only. Retains local-proxy and compiled-precision failures; no WM/action-gradient/contact-force qualification.'}
        write(out/'report.json',report);print(json.dumps({k:report[k] for k in ('status','passed','total','native21_vs_torchFK_max_abs_m','actual16_bonus_range','actual16_gradient_abs_max','phase_counts')},indent=2));print('failed', [k for k,v in checks.items() if not v]);return 0 if all(checks.values()) else 1
    finally:
        for k,value in orig.items():setattr(mujoco,k,value)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--spec',type=Path,default=ROOT/'outputs/contact_intent_v1/objective_spec.json');a.add_argument('--output',type=Path,default=ROOT/'outputs/contact_binding_cpu_v1');p=a.parse_args();raise SystemExit(main(p.spec,p.output))
