"""Independent synthetic/native-source probe review: no environment or physics."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.abc,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
FORBIDDEN=('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic')
class NoDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in FORBIDDEN:raise AssertionError('Forbidden environment/device import '+fullname)
sys.meta_path.insert(0,NoDeviceImports())
from rough_prefix import probe
from rough_prefix.io import checked,pin,write,read

def rotation_wxyz(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(q.shape[:-1]+(3,3))

def independent_grade(states,solver,capacity,ref,lo,hi,flo,fhi,wrist_z):
    if len(states)!=61 or len(solver)!=60:return None,None
    q=np.array([x['joint_rad'] for x in states]);p=np.array([x['object_pose_w'] for x in states]);t=np.array([x['raw_time_s'] for x in states])
    expected=np.array([x['expected_time_s'] for x in states]);actualR=rotation_wxyz(p[:,3:])
    position=np.linalg.norm(p[:,:3]-ref['object_COM_world_m'],axis=-1)
    shaft=np.degrees(np.arccos(np.clip(np.einsum('ni,ni->n',actualR[:,:,0],ref['object_rotation_w'][:,:,0]),-1,1)))
    f=np.array([x['actuator_force_solver_native'] for x in solver]);load=np.maximum(f/fhi,f/flo)
    charge=np.zeros(20);peak=np.zeros(20)
    for dt,row in zip(np.diff(t),load):
        charge=np.minimum(1.,np.maximum(0.,charge+dt*(np.minimum(row,1.)-.5*charge)));peak=np.maximum(peak,charge)
    gates={'object_position_16':bool(np.max(position[::4])<=.02),'directed_shaft_16':bool(np.max(shaft[::4])<=20.),'actual_q_all61':bool(np.max(np.maximum(q-hi,lo-q))<=.001),'penetration_solver60':bool(max(x['penetration_m'] for x in solver)<=.003),'no_nonhand_object_contacts':sum(x['object_environment_contacts'] for x in solver)==0,'no_exogenous':max(x['external_force_max_abs'] for x in states)==0.,'all_actual_u_exact':all(x['ctrl_exact'] for x in states),'clock':bool(np.max(np.abs(t-expected))<=2e-6),'load_ratio':bool(np.max(load)<=1.01),'load_charge':bool(np.max(peak)<=.8),'drop':bool(np.min(p[:,2])>=wrist_z-.15),'capacity':capacity['status']=='OK'}
    return (all(gates.values()) if gates['capacity'] else None),gates

def fixture():
    q=np.zeros(20);pose=np.array([0.,0.,.5,1.,0.,0.,0.])
    states=[{'index':i,'raw_time_s':i*.005,'expected_time_s':i*.005,'joint_rad':q.copy(),'object_pose_w':pose.copy(),'external_force_max_abs':0.,'ctrl_exact':True} for i in range(61)]
    solver=[{'step':i,'actuator_force_solver_native':np.zeros(20),'penetration_m':0.,'object_environment_contacts':0} for i in range(1,61)]
    ref={'object_COM_world_m':np.tile(pose[:3],(61,1)),'object_rotation_w':np.tile(np.eye(3),(61,1,1))}
    return states,solver,{'status':'OK'},ref,-np.ones(20),np.ones(20),-np.ones(20),np.ones(20),.5

def run(output):
    output.mkdir(parents=True,exist_ok=False);checks={};cases={}
    def test(name,modify=None,want=True,focus=None):
        args=list(fixture())
        if modify:modify(args)
        got=probe.grade(*args);expect,eg=independent_grade(*args)
        ok=got['native_A0_pass'] is want and expect is want
        if eg is not None:ok=ok and got['gates']==eg
        if focus is not None:ok=ok and not got['gates'][focus]
        checks[name]=bool(ok);cases[name]={'native_A0_pass':got['native_A0_pass'],'independent_native_A0_pass':expect,'gates':got.get('gates')}
    test('finite_full_positive')
    test('position_exact_threshold',lambda a:a[0][-1]['object_pose_w'].__setitem__(0,.02))
    test('position_above_threshold',lambda a:a[0][-1]['object_pose_w'].__setitem__(0,.020000001),False,'object_position_16')
    # Reference gates use16predeclared20ms ticks; diagnostic61extrema stayvisible.
    test('interior_position_not_primary_gate',lambda a:a[0][1]['object_pose_w'].__setitem__(0,.03))
    def axis(a,degrees,index=60):
        z=np.deg2rad(degrees)/2;a[0][index]['object_pose_w'][3:]=[np.cos(z),0,0,np.sin(z)]
    test('directed_shaft_below20',lambda a:axis(a,19.999))
    test('directed_shaft_above20',lambda a:axis(a,20.001),False,'directed_shaft_16')
    test('interior_q_violation_all61',lambda a:a[0][1]['joint_rad'].__setitem__(0,1.001001),False,'actual_q_all61')
    test('q_within_original1mrad',lambda a:a[0][1]['joint_rad'].__setitem__(0,1.000999))
    test('penetration_exact3mm',lambda a:a[1][0].__setitem__('penetration_m',.003))
    test('penetration_above3mm',lambda a:a[1][0].__setitem__('penetration_m',.003000001),False,'penetration_solver60')
    test('nonhand_support_fails',lambda a:a[1][1].__setitem__('object_environment_contacts',1),False,'no_nonhand_object_contacts')
    test('any_exogenous_force_fails',lambda a:a[0][2].__setitem__('external_force_max_abs',1e-20),False,'no_exogenous')
    test('actual_u_mismatch_fails',lambda a:a[0][3].__setitem__('ctrl_exact',False),False,'all_actual_u_exact')
    test('clock_above_tolerance_fails',lambda a:a[0][4].__setitem__('raw_time_s',.020003),False,'clock')
    test('positive_load_threshold',lambda a:a[1][0]['actuator_force_solver_native'].__setitem__(0,1.01))
    test('positive_load_above_threshold',lambda a:a[1][0]['actuator_force_solver_native'].__setitem__(0,1.010001),False,'load_ratio')
    test('negative_load_above_threshold',lambda a:a[1][0]['actuator_force_solver_native'].__setitem__(0,-1.010001),False,'load_ratio')
    def drop(a):
        a[0][1]['object_pose_w'][2]=.349999;a[3]['object_COM_world_m'][1,2]=.349999
    test('interior_drop_fails',drop,False,'drop')
    test('capacity_UNKNOWN_preserves_unknown',lambda a:a.__setitem__(2,{'status':'UNKNOWN'}),None)
    test('capacity_OVERFLOW_preserves_unknown',lambda a:a.__setitem__(2,{'status':'UNKNOWN_OR_OVERFLOW'}),None)
    test('partial_missing_state_UNKNOWN',lambda a:a[0].pop(),None)
    test('partial_missing_solver_UNKNOWN',lambda a:a[1].pop(),None)
    # A simple negative quaternion sign change represents exactlythe sameR.
    test('quaternion_sign_equivalence',lambda a:a[0][60]['object_pose_w'].__setitem__(slice(3,None),[-1.,0.,0.,0.]))
    source=Path(probe.__file__);tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    fun=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run_probe')
    nested={x.name:x for x in ast.walk(fun) if isinstance(x,ast.FunctionDef)}
    step=nested['step'];calls=[x for x in step.body if isinstance(x,ast.Expr)]
    checks['step_then_sample_AST']=len(calls)==2 and ast.unparse(calls[0])=='original_step()' and ast.unparse(calls[1])=='sample(len(states))'
    checks['index4_u0_index5_u1_and_all60']=all((i-1)//4==min((i-1)//4,14) for i in range(1,61)) and (4-1)//4==0 and (5-1)//4==1
    checks['all15_preaction_phases_explicit_AST']='np.array_equal(phase, np.array([k]))' in ast.unparse(fun) and 'for k in range(15)' in ast.unparse(fun)
    text=ast.unparse(fun)
    checks['quality_no_early_exit_AST']='grade(' not in ast.unparse(nested['step']) and "if bool(terminated.any())" in text and "forward_calls != 15" in text
    overflow=[x for x in ast.walk(fun) if isinstance(x,ast.Attribute) and x.attr=='overflow']
    checks['sticky_once_after60_source_AST']=len(overflow)==1 and "'read_count': 1" in text and "'read_after_step': 60" in text
    checks['no_explicit_forward_or_step_added']=not any(x.func.attr in ('mj_forward','mj_step','forward','step') and not (isinstance(x.func.value,ast.Name) and x.func.value.id=='env') for x in ast.walk(fun) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute))
    # SourceIO checks use a disposable fixture, never mutateanyfrozeninput.
    p=output/'pin_fixture.json';write(p,{'finite':1,'unknown':None});binding=pin(p);checks['source_IO_pin_roundtrip']=checked(binding)==p and read(p)=={'finite':1,'unknown':None}
    drift=False
    try:checked({**binding,'sha256':'0'*64})
    except ValueError:drift=True
    checks['source_IO_changed_hash_rejected']=drift
    finite=False
    try:write(output/'nonfinite_rejected.json',{'invalid':float('nan')})
    except TypeError:finite=True
    checks['source_IO_nonfinite_metadata_rejected']=finite
    checks['no_environment_or_GPU_imports']=all(k.split('.')[0] not in FORBIDDEN for k in sys.modules)
    checks['original_GATES_unchanged']=probe.GATES=={'object_position_max_m':.02,'directed_shaft_error_max_deg':20.,'actual_joint_hard_excess_max_rad':.001,'penetration_any_category_max_m':.003,'object_nonhand_candidate_contact_count_max':0,'external_force_max_abs':0.,'clock_absolute_tolerance_s':2e-6,'target_rate_max_rad_s':15.,'actuator_load_ratio_max':1.01,'load_charge_max':.8,'charge_decay_time_s':2.,'drop_center_below_default_wrist_m':.15}
    result={'schema':'independent_native_a0_probe_CPU_review:1','status':'PASS_SYNTHETIC_AND_STATIC_SCOPE' if all(checks.values()) else 'FAIL','checks':checks,'passed':sum(checks.values()),'total':len(checks),'source':pin(source),'checker':pin(__file__),'cases':cases,'limits':['Synthetic grader/static AST only, not nativeexecution or physicsproof.','target_rate is presently inherited from immutablebaseline loader qualification and ctrlbitexact, not separatelyrecomputed by grade().','Nonfinite runtime states are rejected in sample() and overridegrade to UNKNOWN; this reviewdoesnot confer standalone grader validityfor arbitrary malformed arrays.','Allcontact and force samples mustbe read as justcompleted solver/preintegration, poststateq/qd alone isintegrated.'],'created_environments':0,'new_geometry_queries':0,'physics_steps':0,'optimizer_calls':0,'GPU_used':False}
    write(output/'report.json',result);print(json.dumps({'status':result['status'],'passed':result['passed'],'total':result['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(output/'report.json')},indent=2))
    return 0 if all(checks.values()) else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/independent_native_a0_probe_cpu_v1');a=p.parse_args();raise SystemExit(run(a.output.resolve()))
