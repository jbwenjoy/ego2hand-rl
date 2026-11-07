"""Terminal constrained candidate -> separate reference/init/planned-control.

One explicit artifact export, no LP, optimization, collision query or physics.
Numeric qualification failures retain all bundles. No historical iterate chosen.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import bspline_control as b
PROTO_SHA='0798ea7aa6da6f56684067992906bbba3136f9e15b413f743674f54159ad2e6d'
VARIANT='H123_cubic_bspline_constrained_prefix16_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf-8')


def load_module(path,name,expected):
    if sha(path)!=expected: raise ValueError('Frozen math source drift')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def arrays_and_metadata(value,arrays,prefix=''):
    if isinstance(value,np.ndarray):
        arrays[prefix]=value
        return {'npz_array':prefix,'shape':list(value.shape),'dtype':str(value.dtype)}
    if isinstance(value,dict):return {k:arrays_and_metadata(v,arrays,prefix+'__'+k) for k,v in value.items()}
    return value


def export(manifest_path,expected_sha,output):
    start=time.perf_counter()
    if sha(manifest_path)!=expected_sha:raise ValueError('Manifest hash differs')
    m=read(manifest_path)
    if m.get('schema')!='constrained_prefix_geometry_candidate:1' or m.get('retarget_variant_id')!=VARIANT or m['protocol']['sha256']!=PROTO_SHA:raise ValueError('New constrained variant required')
    for p in [m['protocol'],m['parent_protocol'],m['candidate'],m['report'],m['q0_source'],m['rawFD'],*m['source_pins'].values()]:
        if sha(p['path'])!=p['sha256']:raise ValueError('Manifest bound artifact/source drift: '+p['path'])
    proto=read(m['protocol']['path']);parent=read(m['parent_protocol']['path'])
    if m['source_identity']!=proto['source_identity'] or proto['inputs']!=parent['inputs']:raise ValueError('Source/window/lineage differs')
    for name,p in proto['inputs'].items():
        if sha(p['path'])!=p['sha256']:raise ValueError('Frozen input drift before read: '+name)
    report=read(m['report']['path']);return_path=Path(m['candidate']['path']).parent/'optimizer_return.npz'
    with np.load(m['candidate']['path'],allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(return_path,allow_pickle=False) as z:terminal=z['x'].copy()
    with np.load(m['q0_source']['path'],allow_pickle=False) as z:q0=z['joint_pos_rad'][0].copy()
    x=data['optimizer_x'];q=data['q_goal_rad'];pose=data['mapped_object_pose_w'];t=data['time_s']
    if x.shape!=(340,) or q.shape!=(16,20) or pose.shape!=(16,7) or not np.array_equal(x,terminal) or not np.array_equal(q[0],q0):raise ValueError('Only the normal terminal340-vector and fixed source64 P0 are accepted')
    if not np.array_equal(data['control_points_rad'],b.unpack_controls(x,q0)) or not np.array_equal(t,b.BREAKS):raise ValueError('Coefficient/time layout differs')
    if report.get('canonical_selection')!='Only normally returned terminal SLSQP result.x, never a best historical iterate':raise ValueError('Missing terminal-selection contract')
    old=b.load_frozen_fp32_control()
    object_source=Path('D:/workspace/supermimic-observation-prefix-study/src/prefix_curves.py')
    c=load_module(object_source,'constrained_export_frozen_object_curve','aa03e4c8584c600875a7a6737f1c2f6d72fe63d4dd36be2e31f858ceb0bde384')
    inputs=proto['inputs']
    with np.load(inputs['rawFD_design_arrays']['path'],allow_pickle=False) as z:
        raw={k:z[k][:16].copy() for k in ('time_s','object_pose_mapped_w','raw_v_world_m_s','raw_a_world_m_s2','raw_omega_world_rad_s','raw_alpha_world_rad_s2')}
    if not np.array_equal(pose,raw['object_pose_mapped_w']) or not np.array_equal(t,raw['time_s']):raise ValueError('Frozen object motion changed')
    import mujoco
    import scipy.optimize as optimize
    forbidden={k:0 for k in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_geomDistance','mj_contactForce','mj_collision')}
    original={k:getattr(mujoco,k) for k in forbidden}
    optim_original={k:getattr(optimize,k) for k in ('linprog','minimize','least_squares')};optim_calls={k:0 for k in optim_original}
    def deny(counter,name):
        def fail(*args,**kwargs):counter[name]+=1;raise AssertionError('Pure export forbids '+name)
        return fail
    for k in forbidden:setattr(mujoco,k,deny(forbidden,k))
    for k in optim_calls:setattr(optimize,k,deny(optim_calls,k))
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    try:
        (output/'exporter_source.py').write_bytes(Path(__file__).read_bytes())
        params,pd=old.load_actual_pd(inputs['model']['path'],inputs['actual_context']['path'],model_sha256=inputs['model']['sha256'],context_sha256=inputs['actual_context']['sha256'])
        design=read(inputs['static_preload_design']['path'])
        if design.get('feasible') is not True or not np.array_equal(np.asarray(design['q0_rad']),q0):raise ValueError('Static prior not from this source q0')
        u_PDU=np.asarray(design['u0_rad'],np.float32)
        model=mujoco.MjModel.from_binary_path(inputs['model']['path']);jids=np.asarray(pd['joint_ids']);lower=model.jnt_range[jids,0];upper=model.jnt_range[jids,1]
        control=b.audit_post_C2_controls(x,q0,params,u_PDU,lower,upper)
        tau=old.static_prior_numpy(q0,u_PDU,params);linear=b.build_linear_contract(q0,params,tau,lower,upper)
        linear_res=np.asarray(linear['A_ub']@(x-linear['x0'])-linear['b_centered'])
        times=np.arange(61)*.005;qq=b.evaluate_controls(x,q0,times);qd=b.evaluate_controls(x,q0,times,1);qdd=b.evaluate_controls(x,q0,times,2)
        obj=c.build_prefix_curves(t,np.repeat(q0[None],16,axis=0),pose,raw['raw_v_world_m_s'],raw['raw_a_world_m_s2'],raw['raw_omega_world_rad_s'],raw['raw_alpha_world_rad_s2']).sample(times)
        q_error=float(np.max(np.abs(control['q_goal16_rad']-q)));q61_error=float(np.max(np.abs(qq-data['planned_q_goal_rad'])))
        object_error=max(float(np.max(np.abs(obj['object_COM_world_m']-data['planned_object_pose_w'][:,:3]))),float(np.max(np.abs(obj['object_rotation_w']-c.quaternion_matrices(data['planned_object_pose_w'][:,3:])))))
        import torch
        tp=old.torch_parameters(params);tt=old.static_prior_torch(torch.tensor(q0.copy()),torch.tensor(u_PDU.copy()),tp)
        pre_t=old.proposal_torch(torch.tensor(q),torch.tensor(control['analytic_qd16_rad_s']),tt,tp);exec_t=old.c2_torch(pre_t,tp)
        original_c2=load_module(inputs['pure_C2']['path'],'constrained_export_original_C2',inputs['pure_C2']['sha256'])
        oracle=original_c2.target_inset(pre_t,torch.tensor(params.target_lower.copy()),torch.tensor(params.target_upper.copy()),.020)
        old_export=control['fp32_old_export'];schedule=old_export['tracking_u_exec_15_FP32_rad']
        parity={'static_prior':np.array_equal(tt.numpy().view(np.uint32),tau.view(np.uint32)),
                'proposal':np.array_equal(pre_t.numpy().view(np.uint32),old_export['u_pre_16_FP32_rad'].view(np.uint32)),
                'C2':np.array_equal(exec_t.numpy().view(np.uint32),old_export['u_exec_16_diagnostic_FP32_rad'].view(np.uint32)),
                'original_C2':np.array_equal(oracle.numpy().view(np.uint32),exec_t.numpy().view(np.uint32))}
        pose32=pose[0].astype(np.float32);r32=c.quaternion_matrices(pose32[3:].astype(float));body=model.body('object/cube').id
        v0=raw['raw_v_world_m_s'][0];omega0=raw['raw_omega_world_rad_s'][0]
        init={'q_init_requested_FP32_rad':q0.astype(np.float32),'qd_init_requested_FP32_rad_s':qd[0].astype(np.float32),'object_pose_init_requested_FP32_w':pose32,
              'object_bodyorigin_v_requested_FP32_world_m_s':(v0-np.cross(omega0,r32@model.body_ipos[body])).astype(np.float32),
              'object_angular_requested_FP32_body_rad_s':(r32.T@omega0).astype(np.float32),'object_angular_requested_FP32_world_rad_s':omega0.astype(np.float32),'u_planned0_FP32_rad':schedule[0]}
        reference={'time_s':t,'q_goal_rad':q,'mapped_object_pose_w':pose,'control_points_rad':data['control_points_rad'],'Bspline_knots_s':b.make_basis().knots,
            'analytic_qd16_rad_s':control['analytic_qd16_rad_s'],'analytic_qdd16_rad_s2':control['analytic_qdd16_rad_s2'],'native_FD64_rad_s':control['native_forward_FD64_rad_s'],'native_FD32_rad_s':control['native_forward_FD32_rad_s'],
            'planned_time61_s':times,'q61_rad':qq,'qd61_rad_s':qd,'qdd61_rad_s2':qdd,'saved_generated_depth61_m':data['operational_generated_maxdepth_m']}
        reference.update({k:v for k,v in obj.items() if k.startswith('object_')})
        control_arrays={};control_meta=arrays_and_metadata(control,control_arrays,'control')
        np.savez_compressed(output/'reference_bspline.npz',**reference);np.savez_compressed(output/'init_requested.npz',**init);np.savez_compressed(output/'planned_control.npz',**control_arrays)
        depth=data['operational_generated_maxdepth_m']
        gates={'terminal_and_source_P0_identity':True,'candidate_q16_q61_objectcurve_match':q_error<=1e-12 and q61_error<=1e-12 and object_error<=1e-12,
            'controlpoint_convexhull':control['q_convexhull_certificate']['passed'],'linear_planning_inequalities':bool(np.all(linear_res<=0)),
            'NumPy_Torch_originalC2_FP32_parity':all(parity.values()),'final15_C2_domain':old_export['gates']['actual15_C2_bounds'],
            'final15_proposal_C2_identity':control['proposal_postC2_identity_actual15'],'planned14_target_rate_le15':old_export['gates']['actual14_updates_rate_le15'],
            'postC2_tracking_real_affine_Bernstein':control['postC2_real_affine_Bernstein_certificate']['tracking']['passed'],
            'postC2_hold_real_affine_Bernstein':control['postC2_real_affine_Bernstein_certificate']['hold']['passed'],
            'tracking61_FP32_reference_caps':control['FP32_reference61_caps_pass']['tracking'],'hold61_FP32_reference_caps':control['FP32_reference61_caps_pass']['hold'],
            'same_new_u0':old_export['gates']['hold_and_track_same_u0'],'nonconstant_tracking_planned_targets':old_export['gates']['tracking_FP32_intervention_nonconstant'],
            'saved_generated_depth61_planning_le2_999mm':bool(np.all(depth<=.002999)),'saved_generated_depth61_qualification_le3mm':bool(np.all(depth<=.003))}
        pins={'manifest':pin(manifest_path),'candidate':m['candidate'],'report':m['report'],'terminal_optimizer_return':pin(return_path),'protocol':m['protocol'],'parent_protocol':m['parent_protocol'],
            'exporter':pin(__file__),'basis_control_module':pin(b.__file__),'frozen_FP32_control':pin(old.__file__),'frozen_object_curve':pin(object_source),**{k:inputs[k] for k in ('model','actual_context','rawFD_design_arrays','static_preload_design','pure_C2')}}
        pins.update({'frozen_input__'+name:p for name,p in inputs.items()})
        common={'schema':'constrained_prefix_separate_bundles:1','retarget_variant_id':VARIANT,'source_identity':m['source_identity'],'source_pins':pins,'physical_reference_qualified':False,'init_measured':None,'u_readback':None,'GPU_authorized':False,'training_authorized':False}
        for kind,file in [('reference','reference_bspline.npz'),('init','init_requested.npz'),('control','planned_control.npz')]:write(output/(kind+'_manifest.json'),{**common,'bundle_kind':kind,'artifact':pin(output/file)})
        figures=render(output,times,qq,lower,upper,depth,old_export,params)
        if torch.cuda.is_initialized() or any(forbidden.values()) or any(optim_calls.values()):raise AssertionError('Pure CPU export invariant failed')
        unchanged=all(sha(p['path'])==p['sha256'] for p in pins.values())
        gates['source_input_hashes_unchanged']=unchanged
        receipt={'status':'CPU_BUNDLE_GATES_PASS_PHYSICS_PENDING' if all(gates.values()) else 'CPU_BUNDLE_FAILED_GATES_RETAINED','gates':gates,'optimizer_status_record_only':report['optimizer_status'],
            'source_pins':pins,'source_input_hashes_unchanged':unchanged,'control_metadata':control_meta,'PD_model':pd,'parity':parity,
            'candidate_q16_error_rad':q_error,'candidate_q61_error_rad':q61_error,'candidate_object_curve_error':object_error,'max_linear_Ax_minus_b':float(linear_res.max()),
            'max_saved_generated_overlap_m':float(depth.max()),'geometry_query_count':0,'geometry_saved_only':True,'forbidden_calls':forbidden,'optimizer_function_calls':optim_calls,
            'physics_steps':0,'GPU_used':False,'Torch_CUDA_initialized':torch.cuda.is_initialized(),'init_measured':None,'u_applied_measured':None,'physical_reference_qualified':False,'GPU_authorized':False,
            'figures':figures,'figure_QA':'PENDING actual view','scope':'Final post-C2 planned targets and reference-state predictions only; no physical/WM/FMA/contactforce claim. 61st endpoint diagnostic is not an extra solver sample. Old source identity is same raw lineage; this is a new trajectory/control variant.',
            'wall_seconds_including_math_and_render_excluding_imports':time.perf_counter()-start}
        write(output/'receipt.json',receipt);return receipt
    finally:
        for k,v in original.items():setattr(mujoco,k,v)
        for k,v in optim_original.items():setattr(optimize,k,v)


def render(output,times,q,lower,upper,depth,control,params):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    ax[0,0].plot(times*1000,depth[:,0]*1000,label='saved hand-object61');ax[0,0].plot(times*1000,depth[:,1]*1000,label='saved hand-self61')
    ax[0,0].axhline(3,color='.3',ls=':',label='3mm qualification');ax[0,0].axhline(2.999,color='.6',ls='--',label='2.999mm planning');ax[0,0].set(title='Saved generated depth; no new collision query',xlabel='planned time [ms]',ylabel='mm');ax[0,0].legend(fontsize=7)
    excess=np.maximum(np.maximum(lower-q,q-upper),0).max(axis=1)*1000
    ax[0,1].plot(times*1000,excess);ax[0,1].set(title='Cubic coefficient hull bounds plus61 diagnostics',xlabel='reference time [ms]',ylabel='q reference excess [mrad]')
    rates=np.abs(control['actual_target_rate_14_rad_s']).max(axis=1)
    ax[1,0].scatter(np.arange(1,15)*20,rates);ax[1,0].axhline(15,color='.3',ls=':');ax[1,0].set(title='14 post-C2 planned updates; not applied targets',xlabel='planned update time [ms]',ylabel='max target rate [rad/s]')
    lo=np.maximum(params.actuator_force_bounds[:,0],params.joint_force_bounds[:,0]);hi=np.minimum(params.actuator_force_bounds[:,1],params.joint_force_bounds[:,1])
    for name,prefix in [('tracking',''),('hold','hold_')]:
        values=[control[prefix+'reference_PD_'+g+'_60_FP32_Nm'] for g in ('affine','difference')]
        ratios=[np.where(v>=0,v/hi,v/lo).max(axis=1) for v in values]
        ax[1,1].plot(np.arange(60)*5,np.maximum(*ratios),label=name)
    ax[1,1].axhline(1,color='.3',ls=':');ax[1,1].set(title='Reference-state raw PD predictions, not measured',xlabel='planned solver prestate [ms]',ylabel='max actuator/joint force ratio');ax[1,1].legend(fontsize=7)
    for a in ax.flat:a.grid(alpha=.2)
    fig.suptitle('Constrained cubic prefix | terminal iterate only | geometry still unqualified',fontsize=12)
    files=[output/'constrained_prefix_diagnostics.png',output/'constrained_prefix_diagnostics.svg']
    for p in files:fig.savefig(p,dpi=160)
    plt.close(fig);return [pin(p) for p in files]


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--expected-manifest-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    try:
        r=export(a.manifest,a.expected_manifest_sha256,a.output)
    except BaseException as exc:
        if a.output.is_dir() and not (a.output/'receipt.json').exists():write(a.output/'engineering_failure.json',{'error':repr(exc),'no_retry_or_input_repair':True})
        raise
    print(json.dumps({'status':r['status'],'gates':r['gates'],'receipt_sha256':sha(a.output/'receipt.json')},indent=2))
    raise SystemExit(0 if all(r['gates'].values()) else 2)
