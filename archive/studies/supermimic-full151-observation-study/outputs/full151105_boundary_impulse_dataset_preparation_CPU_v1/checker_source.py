"""Exactly3 synthetic dataset contracts; no actual saved inputs or label runs."""
import copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import boundary_impulse_dataset105_v1 as core


def fixture_schedule():
    identities=[];metadata=[]
    for pi,path in enumerate(('A0','A2')):
        slots=[('donor',k,None) for k in range(150)]
        for si,k in enumerate((0,49,99,149)):
            s=4*pi+si
            records=[{'j':None,'epsilon_index':None,'sign':0,'joint_index':None,'type':'baseline'} for _ in range(3)]
            records += [{'j':j,'epsilon_index':ei,'sign':sign,'joint_index':joint,'type':'direction'} for ei in range(2) for j,joint in enumerate((0,4,8,12)) for sign in (-1,1)]
            for off,record in enumerate(records):
                bid=19*s+off;g=len(identities)+len(slots)
                metadata.append({'row_id':bid,'snapshot_id':s,'path':path,'k':k,'group_id':g,**record});slots.append((record['type'],k,bid))
        for mode,k,bid in slots:
            g=len(identities);identities.append({'group_id':g,'path':path,'mode':mode,'k':k,'branch_id':bid,'first_physical_pre_index':4*g,'last_physical_index':4*g+3,'post20_GPU_row_index':1808+g,
                'terminal_observation':k+1,'valid_next_policy_decision':k+1<150})
    return identities,metadata


def main():
    out=ROOT/'outputs/full151105_boundary_impulse_dataset_preparation_CPU_v1';core.require(not out.exists(),'Fresh synthetic preparation');out.mkdir()
    pins={k:core.pin(v) for k,v in core.OWN.items()};core.c._write_json(out/'source_freeze_before_tests.json',pins)
    for name,value in pins.items():(out/(name+'_source.py')).write_bytes(Path(value['path']).read_bytes())
    checks={}
    def eq(v):core.require(bool(v),'Synthetic contract')
    def reject(fn):
        try:fn()
        except (ValueError,KeyError,AssertionError):return
        raise AssertionError('Expected fail-closed rejection')
    try:
        identities,metadata=fixture_schedule();plan,gids,ids=core.group_plan(identities,metadata,np)
        eq((gids[:3]==[150,151,152]).all() and gids[76]==376 and gids[-1]==451 and ids[-1,-1]==1807 and plan[-1]['terminal_observation']==150)
        eq(sum(r['mode']=='baseline' for r in plan)==24 and sum(r['mode']=='direction' for r in plan)==128 and len(np.unique(ids))==608)
        pre=np.arange(452*58,dtype=np.float32).reshape(452,58)/np.float32(1024);target=pre+np.float32(.125);u=np.arange(452*20,dtype=np.float32).reshape(452,20)/np.float32(4096)
        h1={'pre_x452_FP32':pre,'target_x452_FP32':target,'actual_u452_FP32':u,'all_substep_u452x4x20_FP32':np.broadcast_to(u[:,None],(452,4,20)).copy()}
        pred={**h1,'physical_xi46_FP32':np.arange(46,dtype=np.float32)};context={'reader_context':np.broadcast_to(pred['physical_xi46_FP32'],(128,46)).copy()}
        raw={'pre_visible152_FP32':pre[gids].reshape(8,19,58).copy(),'target_y152_FP32':target[gids].reshape(8,19,58).copy(),'actual_u152_FP32':u[gids].reshape(8,19,20).copy()}
        fit={'pre':pre[gids].copy(),'target':target[gids].copy(),'u':u[gids].copy()};joined=core.training_join(h1,pred,raw,fit,gids,context,np)
        eq(joined['x58_FP32'].tobytes()==fit['pre'].tobytes())
        wrong=copy.deepcopy(fit);wrong['u'][151,19]+=np.float32(1.)
        reject(lambda:core.training_join(h1,pred,raw,wrong,gids,context,np))
        wrong=copy.deepcopy(metadata);wrong[-1]['group_id']=450;reject(lambda:core.group_plan(identities,wrong,np))
        checks['old152_order608_terminal_native_fit_join_wrong_u_rejected']=True

        x=np.zeros((1,58),np.float32);x[:,43:52]=np.eye(3,dtype=np.float32).reshape(1,9);x[:,55:58]=[1.,2.,3.];y=x.copy();y[:,55:58]=0
        P=np.array([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]],np.float32)
        F=np.broadcast_to(np.array([[[2.,0.,0.],[0.,3.,0.]]],np.float64)[:,None],(1,4,2,3)).copy()
        tau=np.broadcast_to(np.array([[[0.,0.,4.],[1.,0.,0.]]],np.float64)[:,None],(1,4,2,3)).copy()
        h=np.full((1,4),np.float32(.005),np.float32);J,K,dt=core.four_substep_impulses(h,F,tau,np);T=4*float(np.float32(.005))
        eq(np.array_equal(J,T*F[:,0]) and np.array_equal(K,T*tau[:,0]) and dt[0]==T)
        r=core.rotate_and_residual(x,y,P,J,K,2.,np.diag([2.,3.,4.]),np.array([0.,0.,-10.]),dt,np)
        eq(np.array_equal(r['J_palm_F64'][0],np.array([[0.,-2.,0.],[3.,0.,0.]])*T) and np.array_equal(r['K_palm_F64'][0],np.array([[0.,0.,4.],[0.,-1.,0.]])*T))
        eq(np.allclose(r['r_v_saved_m_s_F64'][0],[-1.5*T,T,10*T],rtol=0,atol=1e-15))
        eq(np.allclose(r['r_w_saved_rad_s_F64'][0],[-1.,-2.+T/3,-3.-T],rtol=0,atol=1e-15))
        checks['four_native_dt_impulse_units_frame_transpose_no_origin_shift']=True

        # Nonproper raw R makes projection distinguishable; defect must survive.
        a=np.zeros((1,58),np.float32);b=a.copy();a[:,43:52]=np.eye(3,dtype=np.float32).reshape(1,9);b[:,43:52]=np.diag([2.,1.,1.]).astype(np.float32).reshape(1,9)
        a[:,55:58]=[1.,2.,3.];b[:,55:58]=[1.,2.,3.]
        q=core.rotate_and_residual(a,b,np.eye(3,dtype=np.float32)[None],np.zeros((1,2,3)),np.zeros((1,2,3)),1.,np.diag([2.,3.,4.]),np.zeros(3),np.array([.02]),np)
        eq(np.array_equal(q['native_R1_raw_F64'][0],np.diag([2.,1.,1.])) and np.array_equal(q['I_palm1_F64'][0],np.diag([8.,3.,4.])))
        eq(np.array_equal(q['r_w_saved_rad_s_F64'][0],[.75,0.,0.]) and q['rawR_solve_nonsingular_mask'][0] and q['rawR_solve_finite_mask'][0])
        eq(np.array_equal(q['rawR_solve_residual_F64'],np.zeros((1,3))))
        b[:,43:52]=0.;unknown=core.rotate_and_residual(a,b,np.eye(3,dtype=np.float32)[None],np.zeros((1,2,3)),np.zeros((1,2,3)),1.,np.eye(3),np.zeros(3),np.array([.02]),np)
        eq(not unknown['rawR_solve_nonsingular_mask'][0] and not unknown['rawR_solve_finite_mask'][0] and np.isnan(unknown['r_w_saved_rad_s_F64']).all())
        checks['native_rawR_nonzero_defect_no_projection_singular_mask_all_rows']=True
        for value in pins.values():core.checked(value)
        eq(not any(k in sys.modules for k in ('torch','mujoco','warp')))
        report={'schema':'full151105_boundary_impulse_dataset_CPU_contract:1','status':'PASS_THREE_SYNTHETIC_BOUNDARY_DATASET_CONTRACTS','checks':checks,'passed':len(checks),'total':3,'source_pins':pins,
            'actual_saved_array_reads':0,'actual_label_dataset_runs':0,'new_Torch_MJB_backend_model_physics_GPU':0}
        core.c._write_json(out/'report.json',report)
        ready={'schema':'full151105_boundary_impulse_dataset_CPU_READY:1','status':report['status'],'source_pins':pins,'cpu_report':core.pin(out/'report.json'),'cpu_checks':{'passed':len(checks),'total':3},
            'preregistration':core.pin(core.PREREG),'api':{'entry':str(core.OWN['CLI']),'module':str(core.OWN['consumer']),'metadata':'metadata_preflight(protocol_path,output=None)','run':'run_analysis(protocol_path,output)'},'actual_execution_authorized':False}
        core.c._write_json(out/'READY.json',ready);print(json.dumps({'status':report['status'],'passed':len(checks),'total':3,'READY':core.pin(out/'READY.json')}))
    except BaseException as error:
        core.c._write_json(out/'failure.json',{'checks_completed':checks,'error':repr(error),'source_pins':pins,'actual_saved_arrays_label_models_MJB_GPU':0});raise


if __name__=='__main__':main()
