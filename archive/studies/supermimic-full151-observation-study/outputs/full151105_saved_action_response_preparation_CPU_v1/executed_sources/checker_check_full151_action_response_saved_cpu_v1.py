"""Synthetic visible-state/control/pair/precision guards. No saved values/model."""
from __future__ import annotations
import argparse,ast,copy,json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import action_response452105_v1 as a


def fixture():
    pre=np.zeros((452,58),np.float32);pred=np.full((452,58),np.float32(.25));target=np.zeros_like(pre);u=np.zeros((452,20),np.float32)
    snapshot=np.zeros((8,58),np.float32);snapshot_u=np.zeros((8,20),np.float32);branch_u=np.zeros((152,20),np.float32)
    minus=np.zeros((8,4,2,20),np.float32);plus=minus.copy();records=[];identities=[]
    for pi,path in enumerate(('A0','A2')):
        for k in range(150):identities.append({'group_id':len(identities),'path':path,'mode':'donor','k':k,'branch_id':None})
        for capture,k in enumerate(a.PHASES):
            s=pi*4+capture;snapshot[s,0]=np.float32(s/64)
            for off in range(19):
                bid=s*19+off;gid=len(identities);base=off<3;j=None if base else (off-3)%8//2;ei=None if base else (off-3)//8;sign=0 if base else (-1,1)[(off-3)%2]
                pre[gid]=snapshot[s]
                if not base:
                    coordinate=a.JOINTS[j];u[gid,coordinate]=np.float32(sign*a.EPS[ei]);branch_u[bid]=u[gid]
                    target[gid,j]=np.float32(2)*u[gid,coordinate];pred[gid,j]=np.float32(.25)+np.float32(4)*u[gid,coordinate]
                    if sign==-1:minus[s,j,ei]=u[gid]
                    else:plus[s,j,ei]=u[gid]
                identities.append({'group_id':gid,'path':path,'mode':'baseline' if base else 'direction','k':k,'branch_id':bid})
                records.append({'row_id':bid,'snapshot_id':s,'path':path,'k':k,'j':j,'epsilon_index':ei,'epsilon_rad':None if base else a.EPS[ei],
                    'sign':sign,'baseline_repeat':off if base else None,'joint_index':None if base else a.JOINTS[j],'type':'baseline' if base else 'direction','actualu':u[gid].tolist()})
    physical={'snapshot_x':snapshot,'snapshot_u':snapshot_u,'branch_u':branch_u,'direction__centers':snapshot_u.copy(),'direction__minus':minus,'direction__plus':plus,
        'direction__d_eff':(plus.astype(np.float64)-minus.astype(np.float64))/(2*np.asarray(a.EPS)[None,None,:,None]),
        'direction__center_offset':(plus.astype(np.float64)+minus.astype(np.float64))*.5-snapshot_u[:,None,None,:], 'direction__clipped':np.zeros((8,4,2),np.bool_)}
    state={'pre_x452_FP32':pre,'predicted_x452_FP32':pred,'target_x452_FP32':target,'actual_u452_FP32':u}
    controls={path:{'actuator_ids':list(range(20)),'joint_ids':list(range(7,27))} for path in ('A0','A2')}
    return state,identities,records,physical,controls


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output).resolve()
    a.require(not out.exists(),'Fresh CPU proof directory');out.mkdir(parents=True);(out/'executed_sources').mkdir()
    pins={k:a.pin(v) for k,v in a.OWN.items()}
    for k,v in pins.items():(out/'executed_sources'/(k+'_'+Path(v['path']).name)).write_bytes(Path(v['path']).read_bytes())
    a.write(out,'MODULE_FREEZE.json',{'source_pins':pins,'protocol':a.pin(a.PREREG),'scope':'Before synthetic tests; no actual response arrays or model/native imports.'})
    checks={};loads={'attempted':0};original_np_load=np.load
    def check(name,v):checks[name]=bool(v);a.require(checks[name],name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):check(name,True)
        else:check(name,False)
    def deny(*args,**kwargs):loads['attempted']+=1;raise AssertionError('No actual NPZ read in fixture')
    np.load=deny
    try:
        for v in pins.values():ast.parse(Path(v['path']).read_text(encoding='utf-8-sig'))
        check('AST_and_pure_NumPy_namespace',all(k not in sys.modules for k in ('torch','mujoco','warp')))
        check('fixed_dc49_prereg',a.pin(a.PREREG)['sha256']==a.PREREG_SHA)
        header={'schema':a.SCHEMA,'role':a.ROLE,'protocol':{},'output':str(ROOT/'outputs/full151105_saved_action_response_CPU_fixture_NEVER_RUN'),
            'CPU_run_authorized':False,'cpu_threads':1,'source_pins':{'fixture':{}},'cpu_ready':{}}
        a.validate_header(header);check('strict_new_header_positive',True)
        reject('integer_authorization_not_bool',lambda:a.validate_header({**header,'CPU_run_authorized':1}))
        reject('unknown_GPU_field_rejected',lambda:a.validate_header({**header,'GPU_run_authorized':True}))
        data=fixture();raw,guards,rows=a.prepare_inputs(*data)
        check('full152_all_metadata_and_visible_control_guards',len(rows)==152 and all(v.all() for v in guards.values()))
        result=a.decompose(raw,guards);stats=a.statistics(result)
        features=a.per_feature_statistics(result)
        check('all128responses64pairs_no_filter',result['centered_physical_response_F64'].shape==(8,16,58) and result['secant_defined_mask'].shape==(8,4,2) and result['secant_defined_mask'].sum()==64)
        check('baseline_means_are_F64_statrefs',result['baseline_predicted_mean_F64'].dtype==np.float64 and np.all(result['constant_baseline_error_F64']==.25))
        check('all58_per_feature_raw_unit_statistics_retained',features['feature_indices']==list(range(58)) and len(features['state_units'])==58 and len(features['statistics']['centered_response_error']['RMS'])==58 and features['statistics']['physical_coordinate_secant_DEFINED_ONLY']['samples_per_feature']==64)
        empty=copy.deepcopy(result);empty['secant_defined_mask'][:]=False;empty['secant_undefined_mask'][:]=True
        for name in ('model_coordinate_secant_F64','physical_coordinate_secant_F64','coordinate_secant_error_F64'):empty[name][:]=np.nan
        emptystat=a.per_feature_statistics(empty)
        check('no_defined_secant_retains_None_not_zero_per_feature',emptystat['statistics']['physical_coordinate_secant_DEFINED_ONLY']['samples_per_feature']==0 and emptystat['statistics']['physical_coordinate_secant_DEFINED_ONLY']['RMS']==[None]*58)
        for j in range(4):
            check('closedform_physical_secant_gain2_joint'+str(j),np.all(result['physical_coordinate_secant_F64'][:,j,:,j]==2))
            check('closedform_model_secant_gain4_FP32endpoint_joint'+str(j),np.allclose(result['model_coordinate_secant_F64'][:,j,:,j],4,rtol=0,atol=2e-5))
        expected_dimensionless=(2*float(np.float32(.002)))/(2*.002)
        check('legacy_dimensionless_vs_actual_rad_halfstep_distinct',np.all(result['actual_coordinate_denominator_rad'][:,0,0]==2*float(np.float32(.002))) and np.all(raw['legacy_d_eff_dimensionless'][:,0,0,0]==expected_dimensionless) and np.all(result['actual_halfstep_u_rad'][:,0,0,0]==float(np.float32(.002))))
        check('FP32_endpoint_rounding_is_not_mislabeled_clipping',expected_dimensionless!=1 and not raw['legacy_clipped'].any())
        # Signedzero permitted in other18 numeric comparison, never in q/qd bits.
        changed=list(copy.deepcopy(data));gid=150;changed[0]['pre_x452_FP32'][gid,40]=np.float32(-0.)
        r,g,_=a.prepare_inputs(*changed);check('other18_signedzero_numeric_exact_allowed',all(v.all() for v in g.values()) and np.signbit(r['pre_visible152_FP32'][0,0,40]))
        changed=list(copy.deepcopy(data));changed[0]['pre_x452_FP32'][gid,1]=np.float32(-0.)
        r,g,_=a.prepare_inputs(*changed);check('q_signedzero_bit_mismatch_blocks',not g['same_original_snapshot_bits40'].all())
        reject('no_passed_snapshot_subset_on_visible_failure',lambda:a.decompose(r,g))
        changed=list(copy.deepcopy(data));changed[0]['actual_u452_FP32'][150,0]=np.float32(.1)
        r,g,_=a.prepare_inputs(*changed);check('different_baseline_u_not_averaged_away',not g['baseline_u_bitexact_original'].all())
        reject('baseline_u_mismatch_blocks_before_mean',lambda:a.decompose(r,g))
        changed=list(copy.deepcopy(data));changed[0]['actual_u452_FP32'][153,1]=np.float32(.1)
        r,g,_=a.prepare_inputs(*changed);check('nondeclared_joint_change_detected',not g['perturbation_nonj_bitexact_original'].all())
        changed=list(copy.deepcopy(data));changed[2][3]['sign']=1
        reject('original_minus_plus_sign_metadata_must_join',lambda:a.prepare_inputs(*changed))
        changed=list(copy.deepcopy(data));state,ids,metadata,phys,cm=changed
        state['actual_u452_FP32'][153],state['actual_u452_FP32'][154]=state['actual_u452_FP32'][154].copy(),state['actual_u452_FP32'][153].copy()
        phys['branch_u'][3],phys['branch_u'][4]=phys['branch_u'][4].copy(),phys['branch_u'][3].copy()
        metadata[3]['actualu']=phys['branch_u'][3].tolist();metadata[4]['actualu']=phys['branch_u'][4].tolist()
        phys['direction__minus'][0,0,0],phys['direction__plus'][0,0,0]=phys['direction__plus'][0,0,0].copy(),phys['direction__minus'][0,0,0].copy()
        phys['direction__d_eff'][0,0,0]*=-1;phys['direction__clipped'][0,0,0]=True
        r,g,_=a.prepare_inputs(*changed);check('negative_source_plus_minus_order_blocks_not_relabels',not g['source_plus_not_less_than_minus_coordinate'].all())
        reject('negative_pair_cannot_be_called_valid_coordinate_secant',lambda:a.decompose(r,g))
        changed=list(copy.deepcopy(data));changed[0]['target_x452_FP32'][153,2]=np.nan
        reject('nonfinite_source_rejected_not_repaired',lambda:a.prepare_inputs(*changed))
        # All inputs remain same recorded visible58; finite output repeats may vary.
        changed=list(copy.deepcopy(data));changed[0]['target_x452_FP32'][150:153,0]=np.array([-.01,0,.01],np.float32)
        r,g,_=a.prepare_inputs(*changed);noise=a.decompose(r,g)
        check('baseline_range_variance_are_empirical_ddof0',np.isclose(noise['baseline_physical_range_F64'][0,0],.02,atol=1e-9) and np.isclose(noise['baseline_physical_centered_variance_F64'][0,0],2e-4/3,atol=1e-11))
        # Legal zero actualdenominator survives with NaN secants, never epsilon repair.
        changed=list(copy.deepcopy(data));state,ids,metadata,phys,cm=changed
        for off in (3,4):
            bid=off;index=150+off;state['actual_u452_FP32'][index]=0;phys['branch_u'][bid]=0;metadata[bid]['actualu']=[0.]*20
        phys['direction__minus'][0,0,0]=0;phys['direction__plus'][0,0,0]=0;phys['direction__d_eff'][0,0,0]=0;phys['direction__center_offset'][0,0,0]=0;phys['direction__clipped'][0,0,0]=True
        r,g,_=a.prepare_inputs(*changed);zero=a.decompose(r,g)
        check('zero_actual_denominator_all_features_NaN_and_row_kept',zero['secant_undefined_mask'][0,0,0] and np.isnan(zero['model_coordinate_secant_F64'][0,0,0]).all() and np.isnan(zero['physical_coordinate_secant_F64'][0,0,0]).all() and zero['secant_defined_mask'].size==64)
        check('defined_count_reported_without_removing_zero_pair',a.statistics(zero)['q']['defined_pair_count']==63 and a.statistics(zero)['q']['undefined_pairs_kept']==1)
        changed=list(copy.deepcopy(data));changed[0]['predicted_x452_FP32'][:]=.25;changed[0]['target_x452_FP32'][:]=0
        r,g,_=a.prepare_inputs(*changed);flat=a.decompose(r,g)
        check('encoded_zero_response_preserved_no_floor_rescue',flat['model_pair_numeric_zero'].all() and flat['physical_pair_numeric_zero'].all() and np.all(flat['model_coordinate_secant_F64']==0) and flat['secant_defined_mask'].all())
        check('no_stability_or_accuracy_class_in_stats','label' not in stats['q'] and 'stable' not in stats['q'])
        proc=subprocess.Popen([sys.executable,'-B',str(a.OWN['CLI'])],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
        try:stdout,stderr=proc.communicate(timeout=20)
        except BaseException:proc.kill();proc.communicate(timeout=2);raise
        (out/'default_stdout.json').write_text(stdout,encoding='utf-8');(out/'default_stderr.txt').write_text(stderr,encoding='utf-8');a.require(proc.returncode==0,stderr)
        default=json.loads(stdout);check('actual_default_no_backend_no_NPZ',default['Torch_imported'] is False and default['NPZ_opened']==default['model_weight_loads_FW_AD_MJB_Env_GPU_physics']==0)
        check('no_actual_array_read_or_model_backend',loads['attempted']==0 and all(k not in sys.modules for k in ('torch','mujoco','warp')))
        for k,v in pins.items():check('source_frozen_'+k,a.pin(v['path'])==v)
        proof={'schema':'full151105_saved_action_response_CPU_contract:1','status':'PASS_SYNTHETIC_RESPONSE_PAIR_GUARDS_NO_MODEL','checks':checks,'passed':sum(checks.values()),'total':len(checks),
            'source_pins':pins,'protocol':a.pin(a.PREREG),'actual_saved_response_runs':0,'actual_saved_array_reads':0,'actual_model_FW_AD_MJB_physics':0,'synthetic_snapshots':8,'synthetic_pairs':64}
        a.write(out,'report.json',proof)
        ready={'schema':'full151105_saved_action_response_CPU_READY:1','status':proof['status'],'source_pins':pins,'cpu_report':a.pin(out/'report.json'),'cpu_checks':{'passed':proof['passed'],'total':proof['total']},
            'protocol':a.pin(a.PREREG),'CPU_scientific_run_authorized':False,'GPU_run_authorized':False,
            'api':{'entry':str(a.OWN['CLI']),'module':str(a.OWN['consumer']),'schema':a.SCHEMA,'role':a.ROLE,'keys':sorted(a.KEYS),'required_sources':'required_sources(protocol_dict)','header':'validate_header(config_dict)'}}
        a.write(out,'READY.json',ready);print(json.dumps({'passed':proof['passed'],'total':proof['total'],'report':a.pin(out/'report.json'),'READY':a.pin(out/'READY.json'),'source_pins':pins},indent=2))
    except BaseException as error:
        a.write(out,'failure.json',{'status':'FAILED_SYNTHETIC_CONTRACT_PRESERVED','error':repr(error),'checks_completed':checks,'source_pins':pins,'actual_data_model_FW_AD_physics':False});raise
    finally:np.load=original_np_load


if __name__=='__main__':main()
