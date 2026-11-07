"""Three focused synthetic contracts. No saved physical samples or MJB loads."""
import copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import object_linear_momentum105_v1 as core


def main():
    out=ROOT/'outputs/full151105_object_linear_momentum_preparation_CPU_v1'
    core.require(not out.exists(),'Fresh synthetic proof');out.mkdir()
    pins={k:core.pin(v) for k,v in core.OWN.items()};pins['borrowed_IO']=core.pin(ROOT/'src/full151_gradient/response_calibration105_v1.py')
    core.c._write_json(out/'source_freeze_before_tests.json',pins)
    for name,value in pins.items():(out/(name+'_source.py')).write_bytes(Path(value['path']).read_bytes())
    checks={}
    def eq(value):core.require(bool(value),'Synthetic contract')
    def reject(fn):
        try:fn()
        except (ValueError,AssertionError,KeyError):return
        raise AssertionError('Expected fail-closed rejection')
    try:
        # F is local contact-frame force ON geom2. Direct torque is irrelevant
        # to this linear balance; the reordered host address is not a raw index.
        frame=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
        contact={'geom':[0,1],'efc_address':1000,'frame':frame.tolist(),'force6':[1e-12,2.,-3.,11.,12.,13.]}
        raw={'geom':[0,1],'type':3,'worldid':0,'efc_address':[0,-1,1]}
        q=core.contact_force(contact,raw,{1,2},{0},2,np)
        eq(q['constraint_force_evidence_valid'] and q['category']=='HO')
        eq(np.array_equal(q['signed_force_world_N'],np.array([-2.,1e-12,-3.])))
        reverse=core.contact_force({**contact,'geom':[1,0]},{**raw,'geom':[1,0]},{1,2},{0},2,np)
        eq(np.array_equal(reverse['signed_force_world_N'],-q['signed_force_world_N']))
        internal=core.contact_force({**contact,'geom':[1,2]},{**raw,'geom':[1,2]},{1,2},{0},2,np)
        eq(internal['internal'] and np.array_equal(internal['signed_force_world_N'],np.zeros(3)))
        for change in ({'type':2},{'efc_address':[-1,-1]},{'efc_address':[2,-1]},{'worldid':1}):
            unknown=core.contact_force(contact,{**raw,**change},{1,2},{0},2,np)
            eq(unknown['external'] and not unknown['constraint_force_evidence_valid'] and np.isnan(unknown['signed_force_world_N']).all())
        checks['frame_order_internal_cancel_weak_Fn_invalid_unknown']=True

        # This deliberately exposes F32 subtraction loss; stored operands must
        # first become F64. This is an arithmetic test, not a residual threshold.
        mass=2.;dt=np.float32(.005)
        a=np.array([1e8,1.,-1e8],np.float32);b=np.array([1.,1.0000001,-1.],np.float32)
        force=np.array([2.,-4.,1e-12]);gravity=np.array([0.,0.,np.float32(-9.81)],np.float64)
        dp,fg,jc,jg,res,ef=core.impulse_terms(mass,a,b,dt,force,gravity,np)
        expected_dp=np.array([-199999998.,2.*2.**-23,199999998.])
        eq(np.array_equal(dp,expected_dp) and not np.array_equal(dp,mass*(b-a).astype(np.float64)))
        eq(np.array_equal(fg,mass*gravity) and np.array_equal(jc,float(dt)*force) and np.array_equal(jg,float(dt)*fg))
        eq(np.array_equal(res,expected_dp-jc-jg) and np.array_equal(ef,res/float(dt)))
        checks['FP32_operands_first_to_F64_dt_gravity_impulse_units']=True

        context={'body_mass':np.array([[1.,2.,3.]],np.float32)}
        values={stage+'__body_mass':np.broadcast_to(context['body_mass'][0],(2,3)).copy() for stage in ('after_copy','after_set_const_0')}
        readbacks={'A0':copy.deepcopy(values),'A2':copy.deepcopy(values)}
        eq(core.actual_mass(context,readbacks,1,np)==2.)
        bad=copy.deepcopy(readbacks);bad['A2']['after_set_const_0__body_mass'][1,1]+=1
        reject(lambda:core.actual_mass(context,bad,1,np))
        clock=[np.float32(0.)]
        for _ in range(600):clock.append(np.float32(clock[-1]+dt))
        row={'pre_qvel':np.zeros(26).tolist(),'post_qvel':np.ones(26).tolist(),'phase':2,'substep':3,
             'canonical_pre_tick':10,'canonical_post_tick':11,'pre_time':float(clock[10]),'post_time':float(clock[11]),'external_force_max':0.}
        pre,post=core.saved_velocity_clock(row,26,20,clock,np)
        eq(pre.shape==post.shape==(3,) and pre.dtype==post.dtype==np.float32)
        for patch in ({'pre_qvel':[0.]*25},{'pre_qvel':[float('nan')]+[0.]*25},{'canonical_post_tick':12},{'canonical_pre_tick':-1},{'substep':True},{'post_time':float('inf')}):
            reject(lambda patch=patch:core.saved_velocity_clock({**row,**patch},26,20,clock,np))
        checks['both_paths_stages_mass_and_full26_finite_adjacent_clock_guards']=True
        for value in pins.values():core.checked(value)
        eq(not any(name in sys.modules for name in ('torch','mujoco','warp')))
        report={'schema':'full151105_object_linear_momentum_CPU_contract:1','status':'PASS_THREE_SYNTHETIC_SAVED_MOMENTUM_CONTRACTS',
            'checks':checks,'passed':len(checks),'total':3,'source_pins':pins,
            'actual_saved_physical_rows_read':0,'actual_MJB_loads':0,'actual_momentum_analysis_runs':0,'new_model_FW_AD_FK_physics_GPU':0}
        core.c._write_json(out/'report.json',report)
        ready={'schema':'full151105_object_linear_momentum_CPU_READY:1','status':report['status'],'source_pins':pins,
            'cpu_report':core.pin(out/'report.json'),'cpu_checks':{'passed':len(checks),'total':3},'preregistration':core.pin(core.PREREG),
            'api':{'entry':str(core.OWN['CLI']),'module':str(core.OWN['consumer']),'metadata':'metadata_preflight(protocol_path,output=None)','run':'run_analysis(protocol_path,output)'},
            'actual_execution_authorized':False}
        core.c._write_json(out/'READY.json',ready)
        print(json.dumps({'status':report['status'],'passed':len(checks),'total':3,'READY':core.pin(out/'READY.json')}))
    except BaseException as error:
        core.c._write_json(out/'failure.json',{'checks_completed':checks,'error':repr(error),'source_pins':pins,'actual_saved_rows_MJB_model_GPU':0})
        raise


if __name__=='__main__':main()
