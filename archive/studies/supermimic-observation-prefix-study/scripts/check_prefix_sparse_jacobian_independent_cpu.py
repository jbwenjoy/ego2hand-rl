"""Tiny independent directional FD receipt for actual producer CSR; no solve."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prefix_study.inputs import checked_protocol
from prefix_study.problem import Prefix16GeometryProblem
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=ROOT/'outputs/independent_sparse_Jacobian_v1';out.mkdir(parents=True,exist_ok=False)
 paths=[ROOT/'src/prefix_study'/f'{n}.py' for n in ('problem','inputs','io')]
 pins={str(p):sha(p) for p in paths};began=time.monotonic()
 def deadline():
  if time.monotonic()-began>30:raise TimeoutError('Small independent FD bound30s')
 def forbidden(*a,**kw):raise AssertionError('No optimizer/force/dynamics query in independent FD')
 for n in ('mj_forward','mj_inverse','mj_step','mj_step1','mj_step2','mj_geomDistance','mj_contactForce'):setattr(mujoco,n,forbidden)
 import scipy.optimize
 scipy.optimize.least_squares=forbidden
 problem=Prefix16GeometryProblem(checked_protocol(),deadline=deadline)
 base=problem.x0.copy();r,J,local=problem.evaluate(base);rows=[]
 smooth=np.ones(problem.rows,bool)
 for f in range(16):smooth[f*problem.local_rows+45:f*problem.local_rows+45+len(problem.g.pairs)]=False
 def signature(info):
  e=info['native'];return (e.generated_mask.copy(),e.active_mask.copy(),e.contact_count.copy(),e.tie_count.copy(),e.selected_native_geom_order.copy())
 base_sigs=[signature(a) for a in local]
 directions=[]
 for items in ([(1,0,1.)],[(2,4,1.)],[(8,8,1.),(9,12,-.5)],[(15,16,1.)]):
  d=np.zeros(300)
  for frame,joint,value in items:d[(frame-1)*20+joint]=value
  directions.append(d)
 for di,d in enumerate(directions):
  predicted=J@d
  for eps in (1e-6,5e-7):
   plus=problem.evaluate(base+eps*d);minus=problem.evaluate(base-eps*d)
   fd=(plus[0]-minus[0])/(2*eps);stable=np.zeros(problem.rows,bool);changed=[]
   for f,(a,b,original) in enumerate(zip(plus[2],minus[2],base_sigs)):
    pairwise=signature(a);other=signature(b)
    ok=all(np.array_equal(v,w) and np.array_equal(v,z) for v,w,z in zip(pairwise,other,original))
    # Native ties/feature changes remain outside smooth qualifications.
    ok=ok and not np.any(a['native'].tie_count>1) and not np.any(b['native'].tie_count>1)
    if ok:stable[f*problem.local_rows+45:f*problem.local_rows+45+len(problem.g.pairs)]=True
    else:changed.append(f)
   temporal=np.arange(problem.rows)>=16*problem.local_rows
   rows.append({'direction':di,'epsilon':eps,'smooth_rows_max_abs_error':float(np.max(abs(fd[smooth]-predicted[smooth]))),
    'temporal_rows_max_abs_error':float(np.max(abs(fd[temporal]-predicted[temporal]))),
    'stable_native_rows':int(stable.sum()),'stable_native_max_abs_error':float(np.max(abs(fd[stable]-predicted[stable]))) if stable.any() else None,
    'native_branch_or_tie_unqualified_frames':changed,'all_rows_error_diagnostic_only':float(np.max(abs(fd-predicted)))})
 check={'CSR_shape6596x300':J.shape==(6596,300),'q0_no_variable_columns':J[:problem.local_rows].nnz==0,
  'initial_guess_exact_repeatedq0':np.array_equal(problem.q_from_x(base),np.repeat(problem.q0[None],16,axis=0)),
  'smooth_FD_8cases_le1e_4':all(x['smooth_rows_max_abs_error']<=1e-4 for x in rows),
  'temporal_FD_8cases_le1e_8':all(x['temporal_rows_max_abs_error']<=1e-8 for x in rows),
  'source_unchanged_during_check':all(sha(p)==v for p,v in pins.items())}
 report={'status':'PASS_NARROW_CSR_AND_SMOOTH_DERIVATIVE_RECEIPT' if all(check.values()) else 'FAIL_NARROW_CSR_RECEIPT',
  'checks':{k:bool(v) for k,v in check.items()},'cases':rows,'producer_source_sha256':pins,'checker_sha256':sha(__file__),
  'counts':problem.counters(),'wall_seconds':time.monotonic()-began,'optimization_calls':0,'physics_steps':0,'LP_queries':0,'GPU_used':False,
  'scope':'Actual producer CSR shape/frame0 columns/temporal and smooth bone-anchor-prior blocks. Native collision feature/tie rows are separately diagnostic; this does not qualify a full global contact derivative or solved trajectory.'}
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
 assert all(check.values())
if __name__=='__main__':main()
