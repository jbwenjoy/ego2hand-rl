"""One closedform synthetic152 fixture, including one undefined boundary."""
import importlib.util,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'scripts/audit_public_self15_endpoint_response_saved_cpu_v1.py'
spec=importlib.util.spec_from_file_location('_independent_endpoint_saved_audit',MODULE);A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def main():
    out=A.PREP;A.need(not out.exists(),'Fresh single synthetic proof');out.mkdir()
    sources={'auditor':A.pin(MODULE),'checker':A.pin(Path(__file__))}
    A.write(out/'source_freeze_before_fixture.json',{'source_pins':sources,'actual_array_reads':0})
    for name,v in sources.items():(out/(name+'_source.py')).write_bytes(Path(v['path']).read_bytes())
    checks={}
    try:
        import numpy as np
        ids=[];meta=[];gids=[];x=np.zeros((152,58),np.float32);u=np.zeros((152,20),np.float32)
        for s in range(8):
            schedule=[(None,None,0,b) for b in range(3)]+[(j,e,sign,None) for e in range(2) for j in range(4) for sign in (-1,1)]
            for off,(j,e,sign,repeat) in enumerate(schedule):
                bid=19*s+off;gid=226*(s//4)+150+19*(s%4)+off;path=('A0','A2')[s//4];phase=A.PHASES[s%4];base=j is None
                ids.append({'row_id':bid,'snapshot_id':s,'path':path,'mode':'baseline' if base else 'direction','k':phase,'branch_id':bid,'group_id':gid,'physical_rows':[4*gid+h for h in range(4)],'terminal_observation':phase+1})
                meta.append({'row_id':bid,'snapshot_id':s,'path':path,'k':phase,'type':'baseline' if base else 'direction','group_id':gid,'j':j,'epsilon_index':e,'epsilon_rad':None if base else A.EPS[e],
                    'sign':sign,'baseline_repeat':repeat,'joint_index':None if base else A.COORDINATES[j],'native_actuator_id':None if base else j+1,'native_joint_id':None if base else j+10})
                gids.append(gid);x[bid,0]=bid
                if not base:u[bid,A.COORDINATES[j]]=sign*.5
        # Selected control delta1rad and an explicit offaxis delta.5rad in the
        # first pair. This tests why selected denominator is not ||du20||.
        u[4,19]=.5
        target=np.zeros((152,58),np.float32);pred=np.zeros_like(target)
        for b in range(152):
            selected=float(u[b,:19].sum());offaxis=float(u[b,19])
            target[b,40]=2*selected+4*offaxis;pred[b,40]=3*selected+8*offaxis
        predictions={role:pred.copy() for role in A.ROLES};groups=np.asarray(gids,np.int64);sub=np.asarray([[4*g+h for h in range(4)] for g in gids],np.int64)
        arrays,rows,stats,baseline,diagnostics=A.independent(x,u,target,predictions,ids,meta,groups,sub,np)
        A.need(len(rows)==64 and rows[-1]['plus_branch_id']==151 and rows[-1]['plus_group_id']==451 and rows[-1]['k']==149,'All64 terminal mapping')
        checks['all152_oldphase_native_ID_and_EPS_to_jmajor_mapping']=True
        A.need(arrays['physical_response58_F64'][0,40]==4 and arrays[A.ROLES[0]+'__response58_F64'][0,40]==7 and arrays[A.ROLES[0]+'__response_error58_F64'][0,40]==3,'Closedform endpoint response/error')
        A.need(arrays['physical_actual_secant58_F64'][0,40]==4 and arrays[A.ROLES[0]+'__actual_secant_error58_F64'][0,40]==3,'Actual coordinate secants')
        A.need(arrays['physical_nominal_secant58_F64'][0,40]==1000 and arrays['actual_delta_x58_F64'][0,0]==1,'Nominal denominator separate and PRE drift retained')
        checks['independent_scalar_two_denominators_and_visible_PRE_drift']=True
        A.need(diagnostics[0]['selected_abs_delta_rad']==1 and diagnostics[0]['offaxis_L2_rad']==.5 and diagnostics[0]['delta_u20_L2_rad']==math.sqrt(1.25),'Control norm not selected slot')
        checks['offaxis_and_full_control_norm_descriptors_distinct']=True
        A.need(len(stats)==3 and all(len(v)==13 for v in stats.values()) and len(baseline)==8,'All role/cohort/baseline tables')
        A.need(baseline[0]['descriptors']['pre']['range58'][0]==2 and baseline[0]['descriptors']['pre']['centered_population_variance58'][0]==2/3,'Independent fsum population variance')
        A.need(stats[A.ROLES[0]]['ALL']['response']['blocks']['R_raw9']['unit']=='dimensionless' and stats[A.ROLES[0]]['ALL']['actual_secant']['blocks']['p']['unit']=='m/rad','Raw six units and per-control-rad')
        checks['fsum_statistics_all39_and_baseline8_units']=True
        # One boundary variant within this same closedform fixture. Nonzero
        # offaxis/state/target difference remains even if selected du is zero.
        u[4,0]=u[3,0]
        az,rz,sz,bz,dz=A.independent(x,u,target,predictions,ids,meta,groups,sub,np)
        A.need(len(rz)==64 and not az['actual_defined_mask64'][0] and np.isnan(az['physical_actual_secant58_F64'][0]).all() and az['physical_response58_F64'][0,40]==4,'Undefined does not remove or zero raw response')
        A.need(sz[A.ROLES[0]]['ALL']['actual_defined_pairs']==63 and sz[A.ROLES[0]]['ALL']['actual_undefined_pairs']==1 and dz[0]['selected_abs_delta_rad']==0 and dz[0]['delta_u20_L2_rad']==.5,'Explicit undefined denominators and offaxis distinction')
        checks['zero_selected_denominator_NaN_mask_preserves_raw64']=True
        A.need(not any(k in sys.modules for k in ('torch','mujoco','warp')),'No model/native import')
        for v in sources.values():A.checked(v)
        checks['no_actual_arrays_or_producer_import_or_model_execution']=True
        report={'schema':'full151105_public_self15_endpoint_saved_auditor_CPU_contract:1','status':'PASS_ONE_CLOSED_FORM_SCALAR_FSUM_ENDPOINT_FIXTURE',
            'checks':checks,'passed':len(checks),'total':len(checks),'source_pins':sources,'synthetic_fixture_count':1,'actual_array_reads':0,
            'model_checkpoint_FW_AD_MJB_Env_physics_GPU':0,'fixed_reduction_comparison':{'rtol':A.RTOL,'atol':A.ATOL,'scope':'F64 scalar/fsum reduction agreement, not scientific accuracy'}}
        A.write(out/'report.json',report)
        ready={'schema':'full151105_public_self15_endpoint_saved_auditor_CPU_READY:1','source_pins':sources,'cpu_report':A.pin(out/'report.json'),'cpu_checks':{'passed':len(checks),'total':len(checks)},'actual_run_authorized_by_this_proof':False}
        A.write(out/'READY.json',ready);print({'status':report['status'],'passed':report['passed'],'total':report['total'],'READY':A.pin(out/'READY.json')})
    except BaseException as error:
        A.write(out/'failure.json',{'error':repr(error),'checks_completed':checks,'source_pins':sources,'actual_array_reads_models_FW_physics':0,'automatic_retry':False});raise


if __name__=='__main__':main()
