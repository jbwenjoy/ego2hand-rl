"""Posthoc moving-reference comparison and saved q0 contact-cone screens.

No interpolation, time warping, new contact query, optimization, physics or GPU.
Original static-rest verdicts stay immutable; no motion PASS grade is assigned.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
INTENT=Path('D:/workspace/supermimic-observation-contact-intent-study')
GROUND=Path('D:/workspace/supermimic-observation-warmseed-grounding-study')
FULL=Path('D:/workspace/supermimic-observation-retarget-study/outputs/native_contact_refined_geometry_v1/geometric_retarget.npz')
LP=Path('D:/workspace/supermimic-observation-grounding-study/scripts/diagnose_rest_wrench_feasibility_cpu_v2.py')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rot(p):return Rotation.from_quat(np.asarray(p)[..., [4,5,6,3]]).as_matrix()
def summary(a):return {'mean':float(a.mean()),'max':float(a.max()),'endpoint':float(a[-1])}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'outputs/motion_reference_diagnostic_v1');args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    pins={};checks={};arrays={};rows=[];movement={};matrix_arrays={}
    def bind(name,path):
        path=Path(path);pins[name]={'path':str(path),'sha256':sha(path)};return path
    bind('frozen_full151_geometry',FULL)
    if sha(FULL)!='539d6247576b9b2e3fb1b143cf17b6ad507e3baea73c5fafcda4e221d9230c8f':raise ValueError('Not frozen539d')
    with np.load(FULL,allow_pickle=False) as z:desired=z['mapped_object_pose_w'][:16].copy();t=z['time_s'][:16].copy()
    Rd=rot(desired);desired_pos=np.linalg.norm(desired[:,:3]-desired[0,:3],axis=1);desired_shaft=np.rad2deg(np.arccos(np.clip(Rd[:,:,0]@Rd[0,:,0],-1,1)));desired_SO3=np.rad2deg(Rotation.from_matrix(Rd@Rd[0].T).magnitude())
    arrays.update(reference_time_s=t,reference_object_pose_w=desired,reference_position_change_m=desired_pos,reference_directed_shaft_progress_deg=desired_shaft,reference_SO3_progress_deg=desired_SO3)
    checks['desired16_canonical_20ms']=bool(t.shape==(16,) and np.max(np.abs(t-np.arange(16)*.02))<1e-12)
    comparison={};run_dirs={'baseline':GROUND/'outputs/h123_rest_baseline_v1','midpoint':GROUND/'outputs/h123_rest_midpoint_v1'}
    for name,directory in run_dirs.items():
        for f in ('raw_states.npz','evaluation.json','runtime.json','capacity_audit.json'):bind(name+'_'+f,directory/f)
        evaluation=read(directory/'evaluation.json');cap=read(directory/'capacity_audit.json');runtime=read(directory/'runtime.json')
        bind(name+'_frozen_rest_plan',directory.with_name(directory.name+'_watchdog')/'plan_snapshot.json')
        with np.load(directory/'raw_states.npz',allow_pickle=False) as z:
            index=z['index'][::4].copy();actual=z['object_pose_w'][::4].copy();rawtime=z['raw_time_s'][::4].copy();ctrl=z['ctrl_rad'].copy();fixed=z['fixed_target_FP32'].copy()
        checks[name+'_raw61_complete_capacity0']=evaluation['complete'] and evaluation['raw_state_count']==61 and evaluation['solver_sample_count']==60 and runtime['physics_steps']==60 and cap['status']=='OK' and cap['overflow_bitmask']==0
        checks[name+'_every4_exact_indices_no_interpolation']=bool(np.array_equal(index,np.arange(0,61,4)))
        checks[name+'_constantqref_original_rest_NOT_tracking_command']=bool(evaluation['control_condition']=='QREF_C2' and np.array_equal(ctrl.astype(np.float32),np.broadcast_to(fixed,ctrl.shape)))
        checks[name+'_original_static_rest_FAIL_preserved']=evaluation['rest_hold_pass'] is False
        clock=rawtime-t;Ra=rot(actual);position=np.linalg.norm(actual[:,:3]-desired[:,:3],axis=1);shaft=np.rad2deg(np.arccos(np.clip(np.sum(Ra[:,:,0]*Rd[:,:,0],axis=1),-1,1)));SO3=np.rad2deg(Rotation.from_matrix(Ra@np.swapaxes(Rd,-1,-2)).magnitude())
        arrays.update({name+'__'+k:v for k,v in {'raw_state_index':index,'raw_time_s':rawtime,'raw_minus_reference_clock_s':clock,'actual_object_pose_w':actual,'position_error_m':position,'directed_shaft_error_deg':shaft,'SO3_error_deg':SO3}.items()})
        comparison[name]={'position_error_m':summary(position),'directed_shaft_error_deg':summary(shaft),'SO3_error_deg':summary(SO3),'clock_difference_max_abs_s':float(np.abs(clock).max()),'original_static_rest_grade':evaluation,'new_motion_grade':'NOT_ASSIGNED_POSTHOC_EXPLORATORY_ONLY','raw_quaternion_norm_error_max':float(np.max(np.abs(np.linalg.norm(actual[:,3:],axis=1)-1)))}
        for i in range(16):rows.append({'variant':name,'tick':i,'saved_state_index':int(index[i]),'reference_time_s':float(t[i]),'raw_time_s':float(rawtime[i]),'raw_minus_reference_clock_s':float(clock[i]),'position_error_m':float(position[i]),'directed_shaft_error_deg':float(shaft[i]),'SO3_error_deg':float(SO3[i]),'original_static_rest_pass':False,'motion_grade':'NOT_ASSIGNED'})
    bind('dynamic_design_arrays',ROOT/'outputs/dynamic_wrench_v2/designed_wrench_151.npz');bind('dynamic_design_report',ROOT/'outputs/dynamic_wrench_v2/report.json')
    dynamic_report=read(ROOT/'outputs/dynamic_wrench_v2/report.json');modelpin=dynamic_report['inputs']['model'];ctxpin=dynamic_report['inputs']['context']
    if sha(modelpin['path'])!=modelpin['sha256'] or sha(ctxpin['path'])!=ctxpin['sha256']:raise ValueError('Known physical input drift')
    bind('model',modelpin['path']);bind('context',ctxpin['path']);bind('LP_helper',LP)
    if sha(LP)!='ba3f0f934ff7cc2ce6d674c92f3a755487bcc0994f04736664c5ceb87f31fdca':raise ValueError('LP helper drift')
    with np.load(ROOT/'outputs/dynamic_wrench_v2/designed_wrench_151.npz',allow_pickle=False) as z:
        required=z['raw_required_wrench_world_N_Nm'][0].copy();static=z['static_required_wrench_world_N_Nm'][0].copy();target_pose=z['object_pose_mapped_w'][0].copy()
    checks['demand_same_mapped_initial_object_frame']=np.array_equal(target_pose,desired[0])
    model=mujoco.MjModel.from_binary_path(modelpin['path']);obj=model.body('object/cube').id;og=set(np.flatnonzero(model.geom_bodyid==obj).tolist())
    import re
    hand={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])}
    with np.load(ctxpin['path'],allow_pickle=False) as z:friction3=z['geom_friction'][0].copy()
    forbidden={k:0 for k in ('mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_collision','mj_contactForce','mj_geomDistance')}
    def deny(k):
        def f(*a,**kw):forbidden[k]+=1;raise AssertionError('No new physics/contactquery '+k)
        return f
    for k in forbidden:setattr(mujoco,k,deny(k))
    spec=importlib.util.spec_from_file_location('saved_cone_fixed_LP',LP);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    def make_contacts(records):
        result=[]
        for i,r in enumerate(records):
            a,b=map(int,r['geom']);assert (a in og)!=(b in og)
            if (b if a in og else a) not in hand:continue
            pa,pb=int(model.geom_priority[a]),int(model.geom_priority[b])
            if pa==pb:dim=max(int(model.geom_condim[a]),int(model.geom_condim[b]));f3=np.maximum(friction3[a],friction3[b])
            else:selected=a if pa>pb else b;dim=int(model.geom_condim[selected]);f3=friction3[selected]
            f5=np.array([f3[0],f3[0],f3[1],f3[2],f3[2]],dtype=float)
            if 'effective_condim' in r:assert dim==r['effective_condim']
            if 'friction5' in r:assert np.array_equal(f5,r['friction5'])
            native=np.array(r.get('frame_rows_geom0_to_geom1_w',r.get('frame_rows_w')));frame=native*(1. if b in og else -1.)
            point=np.array(r.get('position_w',r.get('pos_w')))
            result.append({'id':i,'frame':frame,'point':point,'friction':f5,'condim':dim})
        return result
    cone_cases={};cone_rows=[]
    def solve_case(name,records,pose,backend,precision,source,com=None):
        contacts=make_contacts(records);com=np.asarray(pose[:3])+rot(np.asarray(pose)[None])[0]@model.body_ipos[obj] if com is None else np.asarray(com)
        checks[name+'_same_initial_pose_numerical_binding']=bool(np.array_equal(pose,target_pose) if precision=='q64' else np.array_equal(np.asarray(pose,dtype=np.float32),target_pose.astype(np.float32)))
        cases={}
        for label,target in (('static',static),('rawFD_t0_dynamic_design',required)):
            value,A=helper.solve(contacts,com,target);value['sum_normal_force_candidate_N']=float(sum(value['edge_weights_N'])) if value.get('feasible_in_native_object_wrench_screen') else None
            cases[label]=value;matrix_arrays[name+'__native_wrench_matrix_world']=A
        entry={'status':'SAVED_CONE_SCREEN_COMPLETE','precision':precision,'backend_and_phase':backend,'source':source,'object_pose_w':np.asarray(pose).tolist(),'COM_world_m':com.tolist(),'COM_minus_canonical_initial_m':(com-target_pose[:3]).tolist(),'contacts':len(contacts),'dynamic_demand_world_N_Nm':required.tolist(),'static_demand_world_N_Nm':static.tolist(),'conditions':cases,'scope':'Object-only saved-contact cone; no new geometry/force query or warm-q trajectory splice. RawFD t0 demand is the frozen full539d task-redesign model demand, not observed force or a physical control command.'}
        cone_cases[name]=entry
        cone_rows.append({'case':name,'status':entry['status'],'precision':precision,'backend':backend,'contacts':len(contacts),'static_feasible':cases['static']['feasible_in_native_object_wrench_screen'],'dynamic_rawFD_t0_feasible':cases['rawFD_t0_dynamic_design']['feasible_in_native_object_wrench_screen'],'static_sum_fn_candidate_N':cases['static']['sum_normal_force_candidate_N'],'dynamic_sum_fn_candidate_N':cases['rawFD_t0_dynamic_design']['sum_normal_force_candidate_N']})
    # Four q64 native-geometry contact sets, all single-frame hypotheses only.
    for hypothesis in ('h23','h123'):
        for variant in ('baseline','midpoint'):
            name=hypothesis+'_'+variant+'_q64_nativeCPU_saved';d=INTENT/f'outputs/frame0_{hypothesis}_seeded_{variant}_v3';npz=bind(name+'_candidate',d/'initialization_candidate.npz');record=bind(name+'_contact_json',d/'native_contacts.json');rep=read(bind(name+'_report',d/'report.json'))
            if sha(npz)!=rep['output_npz_sha256']:raise ValueError('Saved q64 candidate binding differs')
            with np.load(npz,allow_pickle=False) as z:pose64=z['mapped_object_pose_w'][0].copy()
            solve_case(name,read(record),pose64,'nativeCPU saved geometric contactlist, no solvedforce','q64',pins[name+'_contact_json'])
    # q32 nativeCPU canonical H23 was independently saved by the blocked static
    # preload design. Its failure remains immutable; no target u0 is generated.
    preload_report=bind('H23_q32_nativeCPU_blocked_preload_report',INTENT/'outputs/seeded_preload_cpu_v1/report.json');preload_design=read(bind('H23_blocked_preload_design',INTENT/'outputs/seeded_preload_cpu_v1/designed_target_u0.json'))
    pr=read(preload_report);assert preload_design['feasible'] is False and preload_design['u0_rad'] is None
    solve_case('h23_baseline_q32_nativeCPU_saved',pr['q32_native_contacts'],np.asarray(preload_design['object_pose_w_FP32']), 'nativeCPU q32 reconstructed contactlist saved before rest','q32',pins['H23_q32_nativeCPU_blocked_preload_report'])
    # All available q32 actualWarp startup contact sets. We use geometry only,
    # not active/force magnitudes, and never the subsequent moving contact sets.
    phys=[('h23_baseline',GROUND/'outputs/warmseed_rest_qref_v1'),('h123_baseline',run_dirs['baseline']),('h123_midpoint',run_dirs['midpoint'])]
    for prefix,d in phys:
        rawfile=bind(prefix+'_q32_Warp_raw_contacts',d/'raw_contacts.jsonl');start=read(bind(prefix+'_q32_Warp_startup',d/'startup_contact_measurement.json'))
        records=[r for r in (json.loads(line) for line in rawfile.read_text().splitlines() if line.strip()) if r['step']==0 and ((r['geom'][0] in og and r['geom'][1] in hand) or (r['geom'][1] in og and r['geom'][0] in hand))]
        measurement=start['measured_hand_object'];checks[prefix+'_startup_contact_geometry_order_friction_binding']=len(records)==len(measurement['geom_pair']) and [r['geom'] for r in records]==measurement['geom_pair']
        for r,f in zip(records,measurement['friction']):r['friction5']=f
        solve_case(prefix+'_q32_Warp_startup',records,np.array(start['solver_object_body_pose_w']),'actualWarp saved startupforward atzero designed velocity; all generated contacts, not measured-force selection','q32',pins[prefix+'_q32_Warp_raw_contacts'],com=start['solver_object_COM_w'])
    absent=[('h23_midpoint_q32_nativeCPU_saved','No saved nativeCPU q32 contact set'),('h123_baseline_q32_nativeCPU_saved','No saved nativeCPU q32 contact set'),('h123_midpoint_q32_nativeCPU_saved','No saved nativeCPU q32 contact set'),('h23_midpoint_q32_Warp_startup','No authorized/executed H23 midpoint rest probe')]
    for name,why in absent:
        cone_cases[name]={'status':'NOT_MEASURED_UNKNOWN','static_feasible':None,'dynamic_feasible':None,'reason':why};cone_rows.append({'case':name,'status':'NOT_MEASURED_UNKNOWN','precision':'q32','backend':None,'contacts':None,'static_feasible':None,'dynamic_rawFD_t0_feasible':None,'static_sum_fn_candidate_N':None,'dynamic_sum_fn_candidate_N':None})
    checks['all_available_LP_statuses_known']=all(v['solver_status'] in (0,2) for c in cone_cases.values() if c['status']=='SAVED_CONE_SCREEN_COMPLETE' for v in c['conditions'].values())
    checks['no_new_physics_contact_or_force_query']=not any(forbidden.values());checks['no_Torch_Warp_import']=not any(k.split('.')[0] in ('torch','warp','mujoco_warp') for k in sys.modules)
    for table,name in ((rows,'motion_reference_ticks.csv'),(cone_rows,'saved_q0_cone_screens.csv')):
        with (out/name).open('w',newline='',encoding='utf-8-sig') as f:writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    np.savez_compressed(out/'motion_reference_arrays.npz',**arrays);np.savez_compressed(out/'saved_q0_cone_matrices.npz',**matrix_arrays,rawFD_t0_required_world_wrench_N_Nm=required)
    write(out/'saved_q0_cone_screens.json',cone_cases)
    plt.rcParams.update({'svg.fonttype':'none','font.size':10});fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,layout='constrained')
    for name,color in (('baseline','#4c72b0'),('midpoint','#dd8452')):
        for ax,key,scale,label in zip(axes,('position_error_m','directed_shaft_error_deg','SO3_error_deg'),(1000,1,1),('Position error (mm)','Directed shaft error (deg)','Complete SO3 error (deg)')):
            ax.plot(t,arrays[name+'__'+key]*scale,'o-',markersize=3,label=name,color=color);ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Frozen reference tick time (s); saved state indices0,4,...60');axes[-1].set_xlim(0,.3);fig.suptitle('Posthoc moving-reference comparison: constant-qref rest commands\nOriginal static-rest FAIL unchanged; no motion PASS grade assigned',fontsize=12)
    for ext in ('png','svg'):fig.savefig(out/('moving_reference_errors.'+ext),dpi=180)
    plt.close(fig)
    result={'status':'POSTHOC_CPU_DIAGNOSTIC_COMPLETE' if all(checks.values()) else 'CPU_CHECK_FAILED','checks':{k:bool(v) for k,v in checks.items()},'reference_self_motion_first_0p3s':{'position_change_m':summary(desired_pos),'directed_shaft_progress_deg':summary(desired_shaft),'complete_SO3_progress_deg':summary(desired_SO3)},'physical_comparisons':comparison,'saved_q0_cone_screens':cone_cases,'input_pins':pins,'source_sha256':sha(__file__),'output_sha256':{n:sha(out/n) for n in ('motion_reference_arrays.npz','motion_reference_ticks.csv','saved_q0_cone_matrices.npz','saved_q0_cone_screens.csv','saved_q0_cone_screens.json','moving_reference_errors.png','moving_reference_errors.svg')},'confidence':None,'posthoc_exploratory':True,'motion_PASS_assigned':False,'original_static_grades_changed':False,'new_physics_steps':0,'new_contact_queries':0,'GPU_used':False,'forbidden_calls':forbidden,'limits':['The539d reference predates physics but moving-reference evaluation was not the commanded rest experiment or original declared restgrade.','Fixed-index20ms subsampling retains the original raw clock mismatch; no interpolation/time alignment/retiming.','Full-reference t0 dynamic wrench is designed, not measured, and does not imply the warm q0 is a complete151q trajectory.','q64 nativeCPU/q32 nativeCPU/q32 actualWarpstartup cones are separate; no missing artifacts converted to infeasible.','Startup cone includes all generated geometric contacts, not just contacts carrying actual solved force. Mathematical unbounded-normal LP feasibility ignores robot/PD/compliance/dynamics.','RawFD endpoints and unknown source metric/calibration/confidence remain unqualified.']}
    write(out/'report.json',result)
    print(json.dumps({'status':result['status'],'checks_passed':int(sum(checks.values())),'checks_total':len(checks),'reference_motion':result['reference_self_motion_first_0p3s'],'moving':{k:{'position_m':v['position_error_m'],'shaft_deg':v['directed_shaft_error_deg'],'SO3_deg':v['SO3_error_deg']} for k,v in comparison.items()},'cone_results':cone_rows},indent=2));raise SystemExit(not all(checks.values()))


if __name__=='__main__':main()
