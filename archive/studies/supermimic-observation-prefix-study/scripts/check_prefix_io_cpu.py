"""Bounded same-payload IO failure tests; no geometry, solver or entry run."""
from pathlib import Path
import argparse,hashlib,os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prefix_study import io

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'outputs/producer_IO_checks_v1');args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    original=io.replace_with_bounded_retry;attempts=[]
    def retry_same(source,target,**kwargs):
        def replace(a,b):
            attempts.append((str(a),hashlib.sha256(Path(a).read_bytes()).hexdigest()))
            if len(attempts)<=2:raise PermissionError('Simulated metadata reader handle denial')
            os.replace(a,b)
        return original(source,target,replace_fn=replace,**kwargs)
    checks={}
    try:
        io.replace_with_bounded_retry=retry_same
        target=out/'checkpoint_test.npz';io.atomic_npz(target,q_goal_rad=np.ones((16,20)),unknown_gap=np.array([np.nan]),checkpoint_plan_sha256=np.array('f'*64),checkpoint_pid=np.array(1234),checkpoint_source=np.array('jac'))
        with np.load(target,allow_pickle=False) as z:
            checks['NPZ_numeric_and_unknown_NaN_roundtrip']=np.array_equal(z['q_goal_rad'],np.ones((16,20))) and np.isnan(z['unknown_gap']).all()
            checks['NPZ_self_describing_header_roundtrip']=str(z['checkpoint_plan_sha256'])=='f'*64 and int(z['checkpoint_pid'])==1234 and str(z['checkpoint_source'])=='jac'
        checks['same_temp_and_bytes_all_permission_retries']=len(attempts)==3 and len(set(attempts))==1
        checks['successful_atomic_commit_no_temp_remains']=not Path(attempts[0][0]).exists()
        before=io.sha(target);failed_temps=[]
        def bounded_failure(source,target,**kwargs):
            failed_temps.append(Path(source));clock=[0.]
            def now():return clock[0]
            def sleep(dt):clock[0]+=dt
            def always(a,b):raise PermissionError('Exhaust simulated .1s budget')
            return original(source,target,max_seconds=.1,replace_fn=always,clock=now,sleep=sleep)
        io.replace_with_bounded_retry=bounded_failure
        try:io.atomic_npz(target,q_goal_rad=np.zeros((16,20)));raised=False
        except PermissionError:raised=True
        checks['exhaustion_propagates_no_silent_success']=raised
        checks['failed_commit_preserves_prior_checkpoint']=io.sha(target)==before
        checks['failed_temp_evidence_retained']=len(failed_temps)==1 and failed_temps[0].exists()
    finally:io.replace_with_bounded_retry=original
    # JSON uses the already reviewed same-payload writer verbatim.
    value={'plan_sha256':'f'*64,'pid':1234,'source':'jac','finite':True};io.write_json(out/'header_test.json',value)
    checks['JSON_header_exact_roundtrip']=io.read(out/'header_test.json')==value
    checks['159c_vendor_original_hash']=io.sha(ROOT/'vendor/runtime_io.py')=='159c0d11e50017a4b5fab511e6594efb290955c5ee4a543f67374b7ed19675d1'
    report={'status':'BOUNDED_IO_CHECKS_PASS' if all(checks.values()) else 'IO_CHECK_FAIL','checks':{k:bool(v) for k,v in checks.items()},'source_sha256':io.sha(__file__),'io_source_sha256':io.sha(ROOT/'src/prefix_study/io.py'),'geometry_queries':0,'optimization_calls':0,'entry_check_runs':0,'physics_steps':0,'GPU_used':False,'scope':'Same already serialized bytes/temp retry only; no model/geometry/solver callback, process retry, input mutation or physical qualification.'}
    io.write_json(out/'report.json',report);print({'status':report['status'],'passed':sum(checks.values()),'total':len(checks)});raise SystemExit(not all(checks.values()))
if __name__=='__main__':main()
