"""Synthetic saved-data contracts only: never load or run a WM/Env/framework."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import pin,read
from rough_prefix.physical_probe_consumer_math import (
    ARRAYS,EPS,slots,validate_arrays,object_terms_f64,native_rewards,baseline_gate,
    physical_secants,descriptive_agreement,self15_calibration,check_terms)


def fixture():
    a={k:np.zeros(shape,dtype=dtype) for k,(shape,dtype) in ARRAYS.items()}
    x=np.zeros((16,58),np.float32);x[:,43:52]=np.eye(3,dtype=np.float32).reshape(1,9);x[:,40]=.01
    a['snapshot_x'][:]=x;a['donor_next_x'][:]=x
    for k in ('goal_R','goal_R_next'):a[k][:]=np.eye(3,dtype=np.float32)
    a['snapshot_completed'][:]=True;a['completed'][:]=True;a['restore_qualified'][:]=True
    d={'centers':a['snapshot_u'].copy(),'lower':np.full(20,-1,np.float32),'upper':np.full(20,1,np.float32),
       'vectors':np.broadcast_to(np.eye(20)[:8],(16,8,20)).copy(),'free_mask':np.ones((16,20),bool),
       'valid_pair':np.ones((16,8,2),bool),'plus':np.zeros((16,8,2,20),np.float32),'minus':np.zeros((16,8,2,20),np.float32)}
    for j in range(8):
        for e,eps in enumerate(EPS):d['plus'][:,j,e,j]=eps;d['minus'][:,j,e,j]=-eps
    d['effective_vectors']=(d['plus'].astype(float)-d['minus'].astype(float))/(2*np.array(EPS)[None,None,:,None])
    rows=[]
    for r in slots():
        i,s=r['row_id'],r['snapshot_id'];u=d['centers'][s] if r['type']=='baseline' else d['plus' if r['sign']==1 else 'minus'][s,r['j'],r['epsilon_index']]
        a['branch_u'][i]=u;a['branch_x_next'][i]=x[s];a['branch_x_next'][i,40]+=np.float32(.1)*u[0]
        for key,gp,gR in [('branch_object_terms_k','goal_p','goal_R'),('branch_object_terms_kplus1','goal_p_next','goal_R_next')]:
            a[key][i]=object_terms_f64(a['branch_x_next'][i],a[gp][s],a[gR][s]).astype(np.float32)
        rows.append(dict(r,actualu=u.tolist(),r_k=float(native_rewards(a['branch_object_terms_k'][i])),
                         r_kplus1=float(native_rewards(a['branch_object_terms_kplus1'][i])),restorequalified=True,status='COMPLETED',reason=None))
    for s in range(16):
        a['donor_object_terms_k'][s]=object_terms_f64(x[s],a['goal_p'][s],a['goal_R'][s]).astype(np.float32)
        a['donor_object_terms_kplus1'][s]=object_terms_f64(x[s],a['goal_p_next'][s],a['goal_R_next'][s]).astype(np.float32)
    return a,rows,d


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    sources={'consumer':ROOT/'scripts/consume_current104_physical_probe_cpu.py','math':ROOT/'src/rough_prefix/physical_probe_consumer_math.py','checker':Path(__file__)}
    for k,p in sources.items():(out/(k+'__'+p.name)).write_bytes(p.read_bytes())
    checks={}
    def reject(name,fn):
        try:fn()
        except (ValueError,TypeError):checks[name]=True
        else:checks[name]=False
    a,rows,d=fixture();validate_arrays(a,rows,d)
    checks['synthetic_schema_positive']=True
    checks['560_fixed_order_16_snapshots']=len(rows)==560 and rows[0]['sign']==0 and rows[3]['sign']==-1 and rows[4]['sign']==1 and rows[19]['epsilon_index']==1
    checks['all48_baselines_independently_pass']=baseline_gate(a)['all16_passed']
    b=copy.deepcopy(a);b['branch_self15'][0,0]=1
    base=baseline_gate(b);checks['self15_disagreement_is_not_continuous_failure']=base['all16_passed'] and base['snapshots'][0]['self15_disagreement_count']==1
    b=copy.deepcopy(a);b['branch_x_next'][0,0]=np.float32(2.1e-5)
    checks['q_threshold_failure_not_rounded_away']=not baseline_gate(b)['all16_passed']
    b=copy.deepcopy(a);b['branch_x_next'][0,20]=np.float32(2.1e-4)
    checks['qd_threshold_failure']=not baseline_gate(b)['all16_passed']
    b=copy.deepcopy(a);b['completed'][0]=False
    checks['missing_baseline_is_not_qualified']=not baseline_gate(b)['all48_completed']
    b=copy.deepcopy(a);b['snapshot_x']=b['snapshot_x'].astype(float);reject('FP64_state_rejected',lambda:validate_arrays(b,rows,d))
    b=copy.deepcopy(a);b['unknown']=np.zeros(1);reject('unknown_array_rejected',lambda:validate_arrays(b,rows,d))
    rr=copy.deepcopy(rows);rr[0]['snapshot_id']=True;reject('bool_integer_metadata_rejected',lambda:validate_arrays(a,rr,d))
    rr=copy.deepcopy(rows);rr[3]['sign']=1;reject('branch_order_rejected',lambda:validate_arrays(a,rr,d))
    rr=copy.deepcopy(rows);rr[0]['r_k']+=1e-9;reject('reward_JSON_must_exact_FP32_sum',lambda:validate_arrays(a,rr,d))
    rr=copy.deepcopy(rows);rr[0]['extra']=0;reject('unknown_index_field_rejected',lambda:validate_arrays(a,rr,d))
    b=copy.deepcopy(a);b['branch_u'][4,0]=np.nextafter(b['branch_u'][4,0],np.float32(1));reject('oneULP_changed_target_rejected',lambda:validate_arrays(b,rows,d))
    b=copy.deepcopy(a);b['branch_self15'][0,0]=.5;reject('nonbinary_native_label_rejected',lambda:validate_arrays(b,rows,d))
    b=copy.deepcopy(a);b['branch_object_terms_k'][0,0]+=.01;reject('F64_geometry_arithmetic_bound_reject',lambda:validate_arrays(b,rows,d))
    b=copy.deepcopy(a);rr=copy.deepcopy(rows);b['completed'][0]=False;rr[0].update(actualu=None,r_k=None,r_kplus1=None,status='NOT_RUN',reason='not_executed')
    reject('unexecuted_fake_finite_data_rejected',lambda:validate_arrays(b,rr,d))
    for k in ('branch_x_next','branch_u','branch_self15','branch_object_terms_k','branch_object_terms_kplus1'):b[k][0]=np.nan
    validate_arrays(b,rr,d);checks['unexecuted_NaN_preserved_no_fabrication']=True
    sec=physical_secants(a,d,baseline_gate(a));checks['flat_and_stable_both_retained']=sum(x['classification']=='STABLE_NONFLAT' for x in sec)==16 and sum(x['classification']=='FLAT_OR_BELOW_NOISE' for x in sec)==112
    b=copy.deepcopy(a);b['branch_object_terms_k'][20]=b['branch_object_terms_k'][19]
    checks['one_epsilon_flat_is_not_stable']=not physical_secants(b,d,baseline_gate(b))[0]['stable_nonflat']
    # Deliberate one-ULP terms: subtract promoted native sums, not F32 differences or F64 geometry.
    terms=np.array([[1.2345678,7.654321],[1.2345679,7.654321]],np.float32)
    checks['native_reward_exact_F32_sum_then_F64_subtraction']=np.array_equal(native_rewards(terms),terms.sum(-1,dtype=np.float32).astype(float))
    checks['independent_F64_is_diagnostic_only']=check_terms(a['donor_object_terms_k'],a['donor_next_x'],a['goal_p'],a['goal_R'])[0].shape==(16,2)
    cal=self15_calibration(np.zeros((16,15)),np.zeros((16,15)))
    checks['calibration_known_BCE_Brier']=abs(cal['BCE']-np.log(2))<1e-12 and cal['Brier']==.25
    checks['empty_positive_recall_null_no_fake_perfect']=cal['recall'] is None and cal['F1']==0
    agree=descriptive_agreement([dict(stable_nonflat=False)]*8)
    checks['zero_eligible_has_null_no_success']=agree['sign_agreement'] is None and agree['pearson_r'] is None
    # Actual CLI default + explicit-run blocked by synthetic failed engineering, never a WM load.
    fixture_dir=out/'synthetic_failed_attempt';fixture_dir.mkdir()
    fake=fixture_dir/'metadata.json';fake.write_text('{}')
    report=fixture_dir/'physical_report.json';report.write_text(json.dumps(dict(status='FAILED_ENGINEERING_PARTIAL',engineering_qualified=False,baseline_all_passed=False,capacity_finalized=False,capacity_qualified=False,source_lineage_N=1,independent_ego_N=0)))
    manifest=fixture_dir/'physical_manifest.json';proposal=pin(ROOT/'proposals/current104_samewarp_physical_fd_v2/protocol.json')
    manifest.write_text(json.dumps(dict(schema='current104_physical_probe_result:1',proposal=proposal,physical_report=pin(report),physical_arrays=pin(fake),branch_index=pin(fake),source_pins={'synthetic':pin(fake)},capacity_ledgers=pin(fake),lifecycle_evidence=pin(fake),postterminal_identity_qualified=False)))
    rp=fixture_dir/'runplan.json';source_files={**{k:v for k,v in sources.items() if k!='checker'},'WM_loader':ROOT/'src/rough_prefix/world_model_checkpoint_v2.py','Actor_loader':ROOT/'src/rough_prefix/checkpoint_v2.py','bundle_loader':ROOT/'src/rough_prefix/bundle_v2.py','io':ROOT/'src/rough_prefix/io.py'}
    plan=dict(schema='current104_physical_probe_CPU_consumer_plan:1',proposal=proposal,physical_manifest=pin(manifest),output=str(fixture_dir/'blocked_run'),source_pins={k:pin(v) for k,v in source_files.items()},CPU_run_authorized=True,soft_worker_seconds=180,automatic_retry=False);rp.write_text(json.dumps(plan))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'};cmd=[sys.executable,str(sources['consumer']),'--run-plan',str(rp)]
    default=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=30);(out/'default_stdout.txt').write_text(default.stdout+default.stderr)
    checks['actual_default_CPU_metadata_no_WM']=default.returncode==0 and json.loads(default.stdout)['WM_forwards']==0
    blocked=subprocess.run(cmd+['--run'],capture_output=True,text=True,env=env,timeout=30);(out/'blocked_stdout.txt').write_text(blocked.stdout+blocked.stderr)
    saved=read(fixture_dir/'blocked_run/report.json') if (fixture_dir/'blocked_run/report.json').exists() else {}
    checks['actual_failed_physics_failclosed_before_framework']=blocked.returncode==2 and saved.get('WM_forwards')==0 and saved.get('models')==[] and all(v=='NOT_RUN' for v in saved.get('model_statuses',{}).values())
    for name,change in [('unknown_manifest',lambda z:z.update(extra=1)),('unknown_report_bool',lambda z:z.update(postterminal_identity_qualified='true'))]:
        m=read(manifest);change(m);mp=fixture_dir/(name+'.json');mp.write_text(json.dumps(m));pp={**plan,'physical_manifest':pin(mp)};fp=fixture_dir/(name+'_plan.json');fp.write_text(json.dumps(pp));res=subprocess.run([sys.executable,str(sources['consumer']),'--run-plan',str(fp)],capture_output=True,text=True,env=env,timeout=30);checks[name+'_rejected']=res.returncode!=0
    checks['checker_has_no_framework']=not any(x in sys.modules for x in ('torch','warp','mujoco_warp'))
    result={'schema':'physical_probe_CPU_consumer_synthetic_contracts:1','status':'PASS' if all(checks.values()) else 'FAIL','passed':sum(checks.values()),'total':len(checks),'checks':checks,'source_pins':{k:pin(v) for k,v in sources.items()},'actual_WM_loads':0,'WM_forwards':0,'Env_or_physics_or_GPU':False,'scope':'Synthetic arithmetic/schema/actual CLI failclosed only, not real physical or checkpoint accuracy qualification'}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if all(checks.values()) else 1


if __name__=='__main__':sys.exit(main())
