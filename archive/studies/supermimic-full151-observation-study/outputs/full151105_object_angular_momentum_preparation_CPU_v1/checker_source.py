"""Three synthetic angular observer/evidence contracts; no actual saved arrays."""
import copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import object_angular_momentum105_v1 as core


def main():
    out=ROOT/'outputs/full151105_object_angular_momentum_preparation_CPU_v1'
    core.require(not out.exists(),'Fresh synthetic proof');out.mkdir()
    pins={k:core.pin(v) for k,v in {**core.OWN,**core.BORROWED}.items()};core.c._write_json(out/'source_freeze_before_tests.json',pins)
    for name,value in pins.items():(out/(name+'_source.py')).write_bytes(Path(value['path']).read_bytes())
    checks={}
    def eq(value):core.require(bool(value),'Synthetic contract')
    def reject(fn):
        try:fn()
        except (ValueError,AssertionError,KeyError):return
        raise AssertionError('Expected fail-closed rejection')
    try:
        # Analytic rotated inertia, not a producer-generated reference matrix.
        qi=np.array([2*np.cos(np.pi/8),0.,0.,2*np.sin(np.pi/8)],np.float32)
        context={'body_inertia':np.array([[[1.,1.,1.],[2.,4.,6.],[1.,1.,1.]]],np.float32),
                 'body_iquat':np.array([[[1.,0.,0.,0.],qi,[1.,0.,0.,0.]]],np.float32)}
        values={stage+'__'+field:np.broadcast_to(value[0],(2,)+value.shape[1:]).copy() for field,value in context.items() for stage in ('after_copy','after_set_const_0')}
        readbacks={'A0':copy.deepcopy(values),'A2':copy.deepcopy(values)};I,raw=core.actual_inertia(context,readbacks,1,np)
        eq(np.allclose(I,[[3.,-1.,0.],[-1.,3.,0.],[0.,0.,6.]],rtol=0,atol=3e-7))
        pre=np.zeros(27,np.float32);pre[3]=2.;post=pre.copy();post[3:7]=np.array([3/np.sqrt(2),0.,0.,3/np.sqrt(2)],np.float32)
        vel=np.zeros(26,np.float32);vel[3:6]=[1.,2.,3.]
        a=core.endpoint(pre,vel,0,0,I,np);b=core.endpoint(post,vel,0,0,I,np)
        eq(np.allclose(a['L_world'],[1.,5.,18.],rtol=0,atol=6e-7) and np.allclose(b['L_world'],[-5.,1.,18.],rtol=0,atol=6e-7))
        eq(np.allclose(b['L_world']-a['L_world'],[-6.,-4.,0.],rtol=0,atol=1e-6) and raw['actual_iquat_raw_FP32'].tobytes()==qi.tobytes())
        eq(a['quat_raw_norm']==2. and a['qpos'].tobytes()==pre.tobytes() and a['omega_body'].tobytes()==vel[3:6].tobytes())
        checks['nonprincipal_actual_inertia_each_endpoint_rotation_raw_preserved']=True

        frame=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
        contact={'geom':[0,1],'efc_address':999,'pos':[2.,2.,3.],'frame':frame.tolist(),'force6':[1e-12,2.,-3.,1.,2.,3.]}
        rr={'geom':[0,1],'type':3,'worldid':0,'efc_address':[0,-1,1]};COM=np.array([1.,2.,3.])
        q=core.contact_torque(contact,rr,{1,2},{0},2,COM,np)
        eq(q['constraint_force_evidence_valid'] and np.array_equal(q['lever_torque_world_Nm'],[0.,3.,1e-12]) and np.array_equal(q['direct_torque_world_Nm'],[-2.,1.,3.]))
        eq(np.array_equal(q['total_torque_world_Nm'],[-2.,4.,3.+1e-12]))
        rev=core.contact_torque({**contact,'geom':[1,0]},{**rr,'geom':[1,0]},{1,2},{0},2,COM,np)
        eq(np.array_equal(rev['total_torque_world_Nm'],-q['total_torque_world_Nm']))
        internal=core.contact_torque({**contact,'geom':[1,2]},{**rr,'geom':[1,2]},{1,2},{0},2,COM,np)
        eq(internal['internal'] and np.array_equal(internal['total_torque_world_Nm'],np.zeros(3)))
        unknown=core.contact_torque({**contact,'pos':[float('nan'),2.,3.]},rr,{1,2},{0},2,COM,np)
        eq(not unknown['constraint_force_evidence_valid'] and np.isnan(unknown['total_torque_world_Nm']).all())
        checks['direct_once_order_sign_world_lever_internal_weak_Fn_unknown']=True

        bad=copy.deepcopy(readbacks);bad['A2']['after_set_const_0__body_iquat'][1,2,1]=-0.
        reject(lambda:core.actual_inertia(context,bad,1,np))
        bad=copy.deepcopy(readbacks);bad['A0']['after_copy__body_inertia'][0,2,0]=2.
        reject(lambda:core.actual_inertia(context,bad,1,np))
        zero=pre.copy();zero[3:7]=0.;unknown=core.endpoint(zero,vel,0,0,I,np)
        eq(not unknown['valid'] and unknown['quat_raw_norm']==0 and np.isnan(unknown['L_world']).all() and unknown['qpos'].tobytes()==zero.tobytes())
        dt=np.float32(.005);clock=[np.float32(0.)]
        for _ in range(600):clock.append(np.float32(clock[-1]+dt))
        row={'pre_qvel':np.zeros(26).tolist(),'post_qvel':np.ones(26).tolist(),'phase':149,'substep':4,'canonical_pre_tick':599,'canonical_post_tick':600,
             'pre_time':float(clock[599]),'post_time':float(clock[600]),'external_force_max':0.}
        core.linear.saved_velocity_clock(row,26,0,clock,np)
        reject(lambda:core.linear.saved_velocity_clock({**row,'canonical_post_tick':601},26,0,clock,np))
        reject(lambda:core.linear.saved_velocity_clock({**row,'pre_qvel':[float('nan')]+[0.]*25},26,0,clock,np))
        checks['full_field_both_stages_byte_binding_terminal_ticks_unknown_endpoint']=True
        for value in pins.values():core.checked(value)
        eq(not any(v in sys.modules for v in ('torch','mujoco','warp')))
        report={'schema':'full151105_object_angular_momentum_CPU_contract:1','status':'PASS_THREE_SYNTHETIC_ANGULAR_OBSERVER_CONTRACTS',
            'checks':checks,'passed':len(checks),'total':3,'source_pins':pins,'actual_saved_rows_arrays_read':0,'actual_angular_diagnostic_runs':0,'MJB_Data_backend_model_FK_physics_GPU':0}
        core.c._write_json(out/'report.json',report)
        ready={'schema':'full151105_object_angular_momentum_CPU_READY:1','status':report['status'],'source_pins':pins,'cpu_report':core.pin(out/'report.json'),
            'cpu_checks':{'passed':len(checks),'total':3},'preregistration':core.pin(core.PREREG),'api':{'entry':str(core.OWN['CLI']),'module':str(core.OWN['consumer']),
            'metadata':'metadata_preflight(protocol_path,output=None)','run':'run_analysis(protocol_path,output)'},'actual_execution_authorized':False}
        core.c._write_json(out/'READY.json',ready);print(json.dumps({'status':report['status'],'passed':len(checks),'total':3,'READY':core.pin(out/'READY.json')}))
    except BaseException as error:
        core.c._write_json(out/'failure.json',{'checks_completed':checks,'error':repr(error),'source_pins':pins,'actual_saved_rows_MJB_model_GPU':0});raise


if __name__=='__main__':main()
