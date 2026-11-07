"""Small arithmetic-only fixtures. No actual conversion/MJB/model/data load."""
import argparse,ast,copy,json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import converter_forensic105_v1 as f


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output).resolve()
    f.require(not out.exists(),'Fresh forensic proof only');out.mkdir(parents=True);(out/'executed_sources').mkdir()
    pins={k:f.pin(v) for k,v in f.OWN.items()}
    for k,v in pins.items():(out/'executed_sources'/(k+'_'+Path(v['path']).name)).write_bytes(Path(v['path']).read_bytes())
    f.write(out,'source_before.json',pins);checks={}
    def check(n,v):
        checks[n]=bool(v)
        if not checks[n]:raise AssertionError(n)
    def reject(n,fn):
        try:fn()
        except (ValueError,KeyError):check(n,True)
        else:check(n,False)
    try:
        for v in pins.values():ast.parse(Path(v['path']).read_text(encoding='utf-8-sig'))
        check('sources_frozen_parse_before_fixtures',True)
        check('noTorch_Warp_MuJoCo_imports','torch' not in sys.modules and 'warp' not in sys.modules and 'mujoco' not in sys.modules)
        d=f.declaration();check('default_forensic_only_no_conversion',d['physical_state_conversions']==d['scoring_calls']==d['NPZ_opened']==0)
        h={'schema':f.SCHEMA,'role':f.ROLE,'protocol':{},'output':str(ROOT/'outputs/full151105_converter_forensic_CPU_SYNTHETIC_UNEXECUTED'),'CPU_run_authorized':False,'cpu_threads':1,'source_pins':{'synthetic':{}},'cpu_ready':{}}
        f.header(h);check('exact_forensic_header_positive',True)
        for k,v in [('schema','dense_scoring'),('role','same_solver_phase_penalty_observability'),('CPU_run_authorized',1),('cpu_threads',True),('output',str(ROOT/'src'))]:reject('reject_header_'+k,lambda k=k,v=v:f.header({**h,k:v}))
        reject('unknown_header_scoring_flag',lambda:f.header({**h,'allow_proxy':True}))
        a=np.zeros((3,58),np.float32);b=a.copy();a[0,55]=np.float32(1e-4);a[1,52]=np.float32(100000.0078125);b[1,52]=np.float32(100000.)
        diag=f.old_diagnostics(a,b,True)
        check('old_failed_flag_returns_without_throwing',diag['old_all_pass'] is False and not diag['old_flags'][0,55])
        check('old_output_bound_exact_formula',diag['old_bound_F64'][0,55]==8*2**-23)
        check('old_large_output_scale_example_retained',bool(diag['old_flags'][1,52]))
        check('all_component_differences_saved',diag['difference_F64'].shape==diag['old_flags'].shape==(3,58))
        zero=np.zeros((1,58),np.float32);negativezero=zero.copy();negativezero[0,0]=np.float32(-0.)
        bit=f.old_diagnostics(negativezero,zero,True)
        check('q_bit_negativezero_mismatch_not_hidden_by_zero_diff',not bit['old_flags'][0,0] and bit['difference_F64'][0,0]==0)
        huge=f.old_diagnostics(np.full((1,3),3e38,np.float32),np.full((1,3),-3e38,np.float32))
        check('cast_to_F64_before_subtraction',np.isfinite(huge['difference_F64']).all())
        reject('reject_FP64_as_primary_actual',lambda:f.old_diagnostics(a.astype(float),b,True))
        reject('reject_nonfinite_reference',lambda:f.old_diagnostics(a,np.full((3,58),np.nan,np.float32),True))
        reject('reject_wrongstate58_shape',lambda:f.old_diagnostics(np.zeros((1,57),np.float32),np.zeros((1,57),np.float32),True))
        summary=f.block_summary(diag,[{'synthetic_row':i} for i in range(3)])
        check('summary_keeps_allblocks_and_failed_rows',set(summary)=={'q','qd','position','rotation','linear_velocity','angular_velocity'} and summary['angular_velocity']['failed_rows']==1)
        mapping={'hand_qpos':np.arange(7,27),'hand_dof':np.arange(6,26),'object_qpos_start':0,'object_dof_start':0}
        q=np.zeros((2,27));q[:,3]=1;q[:,:3]=[1,2,3];q[:,7:]=np.arange(20)/100
        v=np.zeros((2,26));v[:,:6]=[.4,-.2,.1,1,2,3];v[:,6:]=np.arange(20)/10
        pp=np.tile([.1,.2,.3],(2,1));R=np.broadcast_to(np.eye(3),(2,3,3)).copy()
        shadow=f.conditioning_shadow(q,v,pp,R,mapping)
        x=shadow['F64_shadow_native_definition_NOT_ACCEPTED_STATE']
        check('secondary_identity_frame_closedform',np.allclose(x[:,40:43],[.9,1.8,2.7],atol=1e-15) and np.array_equal(x[:,55:58],v[:,3:6]))
        check('secondary_direct_q_qd_mapping_exact',np.array_equal(x[:,:20],q[:,7:]) and np.array_equal(x[:,20:40],v[:,6:]))
        qnegative=q.copy();qnegative[:,3:7]*=-1
        sign=f.conditioning_shadow(qnegative,v,pp,R,mapping)
        check('quaternion_sign_invariance_secondary',np.array_equal(shadow['object_rotation_F64'],sign['object_rotation_F64']))
        qq=q.copy();qq[:,3]=qq[:,6]=np.sqrt(.5)
        rx=np.broadcast_to(np.array([[1,0,0],[0,0,-1],[0,1,0]],float),(2,3,3)).copy()
        s=f.conditioning_shadow(qq,v,pp,rx,mapping)
        check('nonidentity_bodyomega_to_world_to_palm_F64',np.allclose(s['F64_shadow_native_definition_NOT_ACCEPTED_STATE'][:,55:58],[-2,3,-1],atol=2e-15))
        c=np.sqrt(.5);rz=np.broadcast_to(np.array([[c,-c,0],[c,c,0],[0,0,1]]),(2,3,3)).copy();vv=v.copy();vv[:,3:6]=[1e5,1e5,0]
        cancellation=f.conditioning_shadow(q,vv,pp,rz,mapping)
        check('large_abs_operands_with_nearzero_output_are_recorded',cancellation['absolute_operand_sums_F64'][0,56]>1e5 and cancellation['cancellation_ratios_DIAGNOSTIC_ONLY'][0,56]>1e10)
        check('conditioning_does_not_replace_old_failed_flags',diag['old_all_pass'] is False and 'accepted' not in cancellation)
        reject('invalid_quat_conditioning_rejected',lambda:f.conditioning_shadow(np.zeros((2,27)),v,pp,R,mapping))
        process=subprocess.run([sys.executable,'-B',str(f.OWN['CLI'])],capture_output=True,text=True,timeout=20,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
        (out/'default_stdout.json').write_text(process.stdout,encoding='utf-8');(out/'default_stderr.txt').write_text(process.stderr,encoding='utf-8')
        f.require(process.returncode==0,process.stderr);actual=json.loads(process.stdout)
        check('actual_defaultCLI_noTorch_NPZ_MJB_conversion_scoring',actual['Torch_imported'] is False and actual['Warp_imported'] is False and actual['NPZ_opened']==actual['MJB_loads']==actual['physical_state_conversions']==actual['scoring_calls']==0)
        text=Path(f.__file__).read_text(encoding='utf-8');tree=ast.parse(text);run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_forensic')
        forbidden=('build_penalty_only_objective','build_contact_objective','load_world_model_checkpoint','run_analysis','evaluate_objective')
        check('no_factory_or_scoring_function_in_module',all(name not in text for name in forbidden))
        check('no_torchload_or_binding_import_reachable','torch.load' not in text and 'import binding' not in text and 'from full151_intent' not in text)
        check('one_exact_frozen_native_conversion_callsite',sum(isinstance(n,ast.Attribute) and n.attr=='native_state58' for n in ast.walk(run))==1)
        check('raw_saved_before_old_numeric_comparisons',text.index("npwrite(out,'raw_conversion_all2260.npz'")<text.index('endpoint=old_diagnostics'))
        check('no_old_numeric_flag_early_gate',all('require('+name not in text for name in ('endpoint','leading','pose','poseR')))
        check('oldFAIL_and_no_new_eligibility_explicit',"'original_failed_gate_unchanged':'FAIL'" in text and "'replacement_gate_defined':False" in text and "'dense_scoring_eligibility':False" in text)
        check('no_model_or_backend_imported_by_test','torch' not in sys.modules and 'mujoco' not in sys.modules and 'warp' not in sys.modules)
        for k,v in pins.items():check('executed_source_unchanged_'+k,f.pin(v['path'])==v)
        proof={'schema':'full151105_converter_forensic_CPU_proof:1','status':'PASS_ARITHMETIC_FIXTURES_NO_PHYSICAL_CONVERSION','checks':checks,'passed':sum(checks.values()),'total':len(checks),'source_pins':pins,'protocol':f.pin(f.PREREG),
          'actual_physical_state_conversions':0,'actual_MJB_or_objective_or_model_loads':0,'actual_dense_data_reads':0,'Env_physics_GPU_AD':0,'old_gate_relaxed':False,'new_dense_eligibility':False}
        f.write(out,'report.json',proof)
        ready={'schema':'full151105_converter_forensic_CPU_READY:1','status':'PASS_ARITHMETIC_ONLY_SOURCE_AND_SMALL_FIXTURES','source_pins':pins,'cpu_report':f.pin(out/'report.json'),'cpu_checks':{'passed':proof['passed'],'total':proof['total']},'protocol':f.pin(f.PREREG),
          'api':{'entry':str(f.OWN['CLI']),'module':str(f.OWN['consumer']),'schema':f.SCHEMA,'role':f.ROLE,'keys':sorted(f.KEYS),'required_sources':'required_sources(read(protocol))','default':'metadata JSON/hash only','authorized':'Root separate config; CUDA_VISIBLE_DEVICES=-1 --config PATH --run'},
          'actual_forensic_authorized':False,'objective_or_scoring_authorized':False,'old_failure_not_overwritten':True}
        f.write(out,'READY.json',ready)
        print(json.dumps({'passed':proof['passed'],'total':proof['total'],'report':f.pin(out/'report.json'),'READY':f.pin(out/'READY.json'),'source_pins':pins},indent=2))
    except BaseException as error:
        f.write(out,'failure.json',{'error':repr(error),'checks_completed':checks,'source_pins':pins,'status':'FAILED_PURE_FORENSIC_FIXTURE_PRESERVED','actual_physical_conversion':False});raise


if __name__=='__main__':main()
