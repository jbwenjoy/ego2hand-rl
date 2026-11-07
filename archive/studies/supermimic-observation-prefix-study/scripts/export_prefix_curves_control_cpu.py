"""Export one NEW prefix16 manifest to curves/init/control; never optimize/step.

The real candidate is supplied explicitly after its independent producer run.
Failed geometric/curve/control gates are preserved, never tuned or repaired.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import prefix_curves as c
import prefix_control as u
PROTOCOL=Path('D:/workspace/supermimic-observation-dynamic-wrench-study/proposals/short_dynamic_tracking_v1/protocol.json')
PROTOCOL_SHA='94ea7cbe03f3e2aaa7bb4f7c8860bae8e599cd4e8e823cbe47220f73ffe101a3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item())+'\n',encoding='utf-8')
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def checkpin(p):
    if set(p)!={'path','sha256'} or not Path(p['path']).is_absolute() or sha(p['path'])!=p['sha256']:raise ValueError('Missing or changed source/artifact pin')
def identity(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def samepin(a,b):return a['sha256']==b['sha256'] and Path(a['path']).resolve()==Path(b['path']).resolve()


def validate_manifest(manifest,protocol):
    if manifest.get('schema')!='prefix16_geometry_candidate:1':raise ValueError('Only the new16 geometry producer schema')
    if manifest['protocol']['sha256']!=PROTOCOL_SHA:raise ValueError('Wrong prefix protocol')
    checkpin(manifest['protocol']);checkpin(manifest['candidate']);checkpin(manifest['q0_source'])
    if identity(manifest['source_identity'])!=identity(protocol['source_identity']):raise ValueError('Wrong source/window/phase identity; shared007[0,151) is not this prefix')
    if not samepin(manifest['q0_source'],protocol['inputs']['canonical_H123_q0']):raise ValueError('Wrong hard q0 source')
    if manifest.get('qd_or_u_exported') is not False or manifest.get('curve_controller_qualification')!='PENDING':raise ValueError('Geometry producer cannot self-qualify curves/control')
    if type(manifest.get('optimizer_success')) is not bool or type(manifest.get('hardq0_bitexact')) is not bool:raise ValueError('Missing producer outcome')
    if 'report' in manifest:checkpin(manifest['report'])


def export(manifest_path,manifest_sha,output):
    start=time.perf_counter()
    if sha(PROTOCOL)!=PROTOCOL_SHA or sha(manifest_path)!=manifest_sha:raise ValueError('Protocol/manifest SHA mismatch')
    protocol=read(PROTOCOL);manifest=read(manifest_path);validate_manifest(manifest,protocol)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'exporter_source.py').write_bytes(Path(__file__).read_bytes())
    (output/'prefix_curves_source.py').write_bytes(Path(c.__file__).read_bytes());(output/'prefix_control_source.py').write_bytes(Path(u.__file__).read_bytes())
    pins={'protocol':pin(PROTOCOL),'producer_manifest':pin(manifest_path),'candidate':manifest['candidate'],'q0_source':manifest['q0_source'],'curves_source':pin(c.__file__),'control_source':pin(u.__file__),'exporter_source':pin(__file__)}
    for k in ('rawFD_design_arrays','model','actual_context','static_preload_design','pure_C2'):checkpin(protocol['inputs'][k]);pins[k]=protocol['inputs'][k]
    write(output/'inputs_before_export.json',pins)
    with np.load(manifest['candidate']['path'],allow_pickle=False) as z:
        q=c.finite(z['q_goal_rad'],(16,20));t=c.finite(z['time_s'],(16,));pose=c.finite(z['mapped_object_pose_w'],(16,7))
    with np.load(manifest['q0_source']['path'],allow_pickle=False) as z:q0=z['joint_pos_rad'][0].copy()
    if not np.array_equal(q[0],q0):raise ValueError('New q0 must remain original source64 hard-fixed; no projection/splice')
    with np.load(protocol['inputs']['rawFD_design_arrays']['path'],allow_pickle=False) as z:
        raw={k:z[k][:16].copy() for k in ('time_s','object_pose_mapped_w','raw_v_world_m_s','raw_a_world_m_s2','raw_omega_world_rad_s','raw_alpha_world_rad_s2')}
    if not np.array_equal(t,raw['time_s']) or not np.array_equal(pose,raw['object_pose_mapped_w']):raise ValueError('Object/time changed from frozen full151 rawFD prefix')
    curves=c.build_prefix_curves(t,q,pose,raw['raw_v_world_m_s'],raw['raw_a_world_m_s2'],raw['raw_omega_world_rad_s'],raw['raw_alpha_world_rad_s2'])
    mp,cp=protocol['inputs']['model'],protocol['inputs']['actual_context'];params,pdproof=u.load_actual_pd(mp['path'],cp['path'],model_sha256=mp['sha256'],context_sha256=cp['sha256'])
    design=read(protocol['inputs']['static_preload_design']['path'])
    if design.get('schema')!='seeded_static_PD_preload:1' or design.get('feasible') is not True or not np.array_equal(design['q0_rad'],q0):raise ValueError('No hash-bound static load prior for this canonical q0')
    for k,expected in [('model',mp),('context',cp)]:
        if not any(samepin(expected,v) for v in design['source_pins'].values()):raise ValueError('Static prior physics identity mismatch')
    u_PDU=np.asarray(design['u0_rad'],np.float32);control=u.export_control(curves,params,q0,u_PDU)
    knots=curves.sample(t);ticks=curves.sample(np.arange(61)*.005)
    import mujoco
    model=mujoco.MjModel.from_binary_path(mp['path']);body=model.body('object/cube').id
    if np.any(model.body_ipos[body]):raise ValueError('This COM=bodyorigin initialization contract requires actual body_ipos0')
    hardlower=model.jnt_range[pdproof['joint_ids'],0];hardupper=model.jnt_range[pdproof['joint_ids'],1]
    qrange=curves.q_range_check(hardlower,hardupper)
    expected={'q_goal_rad':q,'q_curve_qd_rad_s':curves.q_curve_qd_rad_s,'q_curve_qdd_rad_s2':curves.q_curve_qdd_rad_s2,'object_COM_world_m':pose[:,:3],'object_v_world_m_s':raw['raw_v_world_m_s'],'object_a_world_m_s2':raw['raw_a_world_m_s2'],'object_rotation_w':c.quaternion_matrices(pose[:,3:]),'object_omega_world_rad_s':raw['raw_omega_world_rad_s'],'object_alpha_world_rad_s2':raw['raw_alpha_world_rad_s2']}
    tolerance={'q_goal_rad':1e-10,'q_curve_qd_rad_s':1e-8,'q_curve_qdd_rad_s2':1e-6,'object_COM_world_m':1e-10,'object_v_world_m_s':1e-8,'object_a_world_m_s2':1e-6,'object_rotation_w':1e-10,'object_omega_world_rad_s':1e-8,'object_alpha_world_rad_s2':1e-6}
    errors={k:float(np.max(np.abs(knots[k]-v))) for k,v in expected.items()}
    import torch
    tp=u.torch_parameters(params);tf=u.static_prior_torch(torch.tensor(q0.copy()),torch.tensor(u_PDU.copy()),tp)
    pre_t=u.proposal_torch(torch.tensor(knots['q_goal_rad']),torch.tensor(knots['q_curve_qd_rad_s']),tf,tp);exec_t=u.c2_torch(pre_t,tp)
    parity={'static_prior_FP32_bitexact':np.array_equal(tf.numpy().view(np.uint32),control['tau_ff_static0_FP32_Nm'].view(np.uint32)),'proposal_FP32_bitexact':np.array_equal(pre_t.numpy().view(np.uint32),control['u_pre_16_FP32_rad'].view(np.uint32)),'C2_FP32_bitexact':np.array_equal(exec_t.numpy().view(np.uint32),control['u_exec_16_diagnostic_FP32_rad'].view(np.uint32))}
    import importlib.util
    purepin=protocol['inputs']['pure_C2'];spec=importlib.util.spec_from_file_location('prefix_export_pinned_C2',purepin['path']);pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
    oracle=pure.target_inset(torch.tensor(control['u_pre_16_FP32_rad']),torch.tensor(params.target_lower.copy()),torch.tensor(params.target_upper.copy()),.020).numpy()
    parity['actual_original_C2_FP32_bitexact']=np.array_equal(oracle.view(np.uint32),control['u_exec_16_diagnostic_FP32_rad'].view(np.uint32))
    pose32=pose[0].astype(np.float32);r_init=c.quaternion_matrices(pose32[3:].astype(float));omega0=raw['raw_omega_world_rad_s'][0];v0=raw['raw_v_world_m_s'][0]
    init={'q_init_requested_FP32_rad':q[0].astype(np.float32),'qd_init_requested_FP32_rad_s':knots['q_curve_qd_rad_s'][0].astype(np.float32),'object_pose_init_requested_FP32_w':pose32,'object_bodyorigin_v_requested_FP32_w_m_s':(v0-np.cross(omega0,r_init@model.body_ipos[body])).astype(np.float32),'object_angular_requested_FP32_body_rad_s':(r_init.T@omega0).astype(np.float32),'object_angular_requested_FP32_world_rad_s':omega0.astype(np.float32),'u_executed0_FP32_rad':control['tracking_u_exec_15_FP32_rad'][0]}
    a0,d0=u.affine_predictions_numpy(init['q_init_requested_FP32_rad'],init['qd_init_requested_FP32_rad_s'],init['u_executed0_FP32_rad'],params)
    init['reference_predicted_initial_PD_affine_FP32_Nm']=a0;init['reference_predicted_initial_PD_difference_FP32_Nm']=d0
    reference={'time_s':t,'q_goal_rad':q,'mapped_object_pose_w':pose,'q_curve_qd_rad_s':knots['q_curve_qd_rad_s'],'q_curve_qdd_rad_s2':knots['q_curve_qdd_rad_s2'],'q_stencil_knot_qd_rad_s':curves.q_curve_qd_rad_s,'q_stencil_knot_qdd_rad_s2':curves.q_curve_qdd_rad_s2,'native_FD_goal_qd_rad_s':curves.native_FD_goal_qd_rad_s,'native_FD_goal_qd_FP32_rad_s':curves.native_FD_goal_qd_FP32_rad_s,'q_quintic_coefficients':curves.q_coefficients,'p_quintic_coefficients':curves.p_coefficients,'phi_quintic_coefficients':curves.phi_coefficients,'rotation_knots':curves.rotation_knots}
    reference.update({'tick61__'+k:v for k,v in ticks.items()});reference.update({'knot16__'+k:v for k,v in knots.items()})
    np.savez_compressed(output/'reference_curves.npz',**reference);np.savez_compressed(output/'init_requested.npz',**init);np.savez_compressed(output/'control_schedule.npz',**{k:v for k,v in control.items() if isinstance(v,np.ndarray)})
    gates={'producer_optimizer_success':manifest['optimizer_success'],'producer_hardq0_bitexact':manifest['hardq0_bitexact'],'whole_q_curve_hardrange':qrange['passed'],'endpoint_derivatives_match':all(errors[k]<=tolerance[k] for k in errors),'pure_NumPy_Torch_and_original_C2_parity':all(parity.values()),**control['gates']}
    write(output/'whole_q_range.json',qrange)
    figures=render_diagnostics(output,t,q,knots,ticks,qrange,hardlower,hardupper,control,params,gates)
    common={'schema':'prefix16_separate_bundles:1','source_identity':protocol['source_identity'],'protocol':pin(PROTOCOL),'producer':pin(manifest_path),'candidate':manifest['candidate'],'source_pins':pins,'physical_reference_qualified':False,'init_measured':None,'training_authorized':False,'GPU_authorized':False}
    for name,filename,scope in [('reference','reference_curves.npz','Goal/continuouscurve derivatives plus separately native FD. Object v/a/worldomega/alpha are original full151 rawFD at first16, not prefix re-estimates.'),('init','init_requested.npz','Designed nonzero FP32 initializer only; worldomega converted using normalized requested q32 orientation to freejoint bodyangular. Actual readback remains PENDING.'),('control','control_schedule.npz','15 commands/14 updates, intervalk at[.02k,.02(k+1));16th diagnostic never executed. Fixedstaticload transported plus reference qd damping, not inverse dynamics or measuredforce.')]:
        write(output/(name+'_manifest.json'),{**common,'bundle_kind':name,'artifact':pin(output/filename),'scope':scope})
    allpins=all(sha(v['path'])==v['sha256'] for v in pins.values());assert allpins and not torch.cuda.is_initialized()
    receipt={'status':'CPU_EXPORT_GATES_PASS_PHYSICS_PENDING' if all(gates.values()) else 'CPU_EXPORT_FAILED_GATES_RETAINED_NO_TUNING','gates':gates,'source_pins':pins,'source_input_hashes_unchanged':allpins,'q_stencils':c.Q_STENCILS,'PD_grouping':u.GROUPING,'actual_PD_model':pdproof,'knot_endpoint_max_abs_errors':errors,'numeric_endpoint_tolerances':tolerance,'NumPy_Torch_parity':parity,'control_diagnostics':{k:v for k,v in control.items() if not isinstance(v,np.ndarray)},'whole_q_range':pin(output/'whole_q_range.json'),'figures':figures,'figure_QA':'PENDING actual image view after this real export','bundles':{k:pin(output/(k+'_manifest.json')) for k in ('reference','init','control')},'nonzero_init_is_not_previous_zero_rest':True,'relative_initial_contact_velocity':'PENDING independent geometry diagnostics; this pure exporter does not query contacts','all16_and61_native_geometry':'PENDING separate geometric qualification','rawFD_recomputed':False,'producer_candidate_optimized_here':False,'physics_steps':0,'LP_solves':0,'GPU_used':False,'Torch_CUDA_initialized':torch.cuda.is_initialized(),'WMgradient_or_Warp_dynamics_qualified':False,'physical_reference_qualified':False,'new_initialization_measured':None,'GPU_authorized':False,'wall_s_excluding_imports':time.perf_counter()-start}
    write(output/'receipt.json',receipt);print(json.dumps({'status':receipt['status'],'gates':gates,'receipt_sha256':sha(output/'receipt.json')},indent=2))
    return receipt


def render_diagnostics(output,time,q,knots,ticks,qrange,lower,upper,control,params,gates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    def excess(values):return np.maximum(np.maximum(lower-values,values-upper),0).max(axis=1)*1000
    ax[0,0].plot(np.arange(61)*5,excess(ticks['q_goal_rad']),label='61 planned ticks',color='#31688e')
    ax[0,0].scatter(time*1000,excess(q),label='16 knots',color='#35b779',s=18,zorder=3)
    extremetimes=[];extremeexcess=[]
    for row in qrange['all300_segments_joints']:
        for s,v in zip(row['candidate_s'],row['candidate_values_rad']):
            e=max(lower[row['joint']]-v,v-upper[row['joint']],0.)
            if e>0:extremetimes.append((row['segment']+s)*20);extremeexcess.append(e*1000)
    ax[0,0].scatter(extremetimes,extremeexcess,color='#b63679',marker='x',s=22,label='all polynomial extrema with excess')
    ax[0,0].set(title='Continuous q hard-range audit (zero slack)',xlabel='reference time [ms]',ylabel='maximum excess [mrad]');ax[0,0].legend(fontsize=7)
    rates=np.abs(control['actual_target_rate_14_rad_s']).max(axis=1)
    ax[0,1].step(np.arange(1,15)*20,rates,where='mid',label='tracking14 post-init updates');ax[0,1].axhline(15,color='0.4',ls=':',label='15 rad/s cap')
    ax[0,1].set(title='Actual15 commands: target update rate',xlabel='update time [ms]',ylabel='max joint rate [rad/s]');ax[0,1].legend(fontsize=7)
    lo=np.maximum(params.actuator_force_bounds[:,0],params.joint_force_bounds[:,0]);hi=np.minimum(params.actuator_force_bounds[:,1],params.joint_force_bounds[:,1])
    for name,color,prefix in [('track','#31688e',''),('hold','#b63679','hold_')]:
        ratios=[]
        for grouping in ['affine','difference']:
            force=control[prefix+'reference_PD_'+grouping+'_60_FP32_Nm'];ratios.append(np.where(force>=0,force/hi,force/lo).max(axis=1))
        ax[1,0].plot(np.arange(60)*5,np.maximum(*ratios),label=name+' max of two FP32 groupings',color=color)
    ax[1,0].axhline(1,color='0.4',ls=':',label='raw PD cap');ax[1,0].set(title='Reference-state raw PD prediction; no dynamics',xlabel='planned solver prestate [ms]',ylabel='max actuator/joint force ratio');ax[1,0].legend(fontsize=7)
    def shaft(rotation):
        v=rotation[:,:,0];return np.rad2deg(np.arccos(np.clip(v@v[0],-1,1)))
    ax[1,1].plot(np.arange(61)*5,shaft(ticks['object_rotation_w']),label='61 continuous reference ticks');ax[1,1].scatter(time*1000,shaft(knots['object_rotation_w']),color='#35b779',s=18,label='16 frozen pose knots')
    ax[1,1].set(title='Reference directed shaft excursion (not tracking)',xlabel='reference time [ms]',ylabel='deg from initial shaft');ax[1,1].legend(fontsize=7)
    for a in ax.flat:a.grid(alpha=.2)
    fig.suptitle('Prefix16 CPU export | '+('all numeric gates pass; physics pending' if all(gates.values()) else 'failed gates retained; no tuning or physics'),fontsize=12)
    files=[Path(output)/'prefix_cpu_diagnostics.png',Path(output)/'prefix_cpu_diagnostics.svg']
    for file in files:fig.savefig(file,dpi=160)
    plt.close(fig)
    return [pin(p) for p in files]


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--candidate-manifest',type=Path,required=True);ap.add_argument('--expected-manifest-sha256',required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    try:result=export(args.candidate_manifest,args.expected_manifest_sha256,args.output)
    except BaseException as exc:
        if args.output.is_dir() and not (args.output/'receipt.json').exists():write(args.output/'engineering_failure.json',{'error':repr(exc),'GPU_used':False,'physics_steps':0,'no_retry_or_input_repair':True})
        raise
    raise SystemExit(0 if all(result['gates'].values()) else 2)
