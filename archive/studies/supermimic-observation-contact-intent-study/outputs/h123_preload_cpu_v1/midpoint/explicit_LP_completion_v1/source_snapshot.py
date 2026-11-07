"""Explicitly complete two midpoint hand LPs from frozen matrices; no geometry."""
from pathlib import Path
import hashlib,json,time,os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1];case=ROOT/'outputs/h123_preload_cpu_v1/midpoint';out=case/'explicit_LP_completion_v1';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
source=case/'static_matrices.npz';record={'source_matrices':{'path':str(source),'sha256':sha(source)},'original_report':{'path':str(case/'report.json'),'sha256':sha(case/'report.json')},'original_design':{'path':str(case/'designed_target_u0.json'),'sha256':sha(case/'designed_target_u0.json')},'source_sha256':sha(__file__),'purpose':'Original design stopped at object-only infeasibility. This artifact explicitly solves the two requested wholehand LPs using exactly those already frozen matrices; original reports remain unchanged.'}
start=time.perf_counter()
with np.load(source) as z:d={k:z[k].copy() for k in z.files}
ne=d['A_object_wrench'].shape[1];eq=np.block([[d['A_object_wrench'],np.zeros((6,40))],[d['G_hand_generalized_contact'],np.eye(20),np.eye(20)]])
b=np.r_[d['target_object_wrench'],-d['gravity_robot_tau']];scale=np.r_[np.ones(3),np.full(23,100.)];results={}
for name,bounded in [('zero_stiction',False),('bounded_stiction',True)]:
 bounds=[(0,None)]*ne+list(zip(d['tau_lower'],d['tau_upper']))+(list(zip(-d['stiction_loss'],d['stiction_loss'])) if bounded else [(0,0)]*20)
 lp=linprog(np.r_[np.ones(ne),np.zeros(40)],A_eq=scale[:,None]*eq,b_eq=scale*b,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'time_limit':10.})
 results[name]={'solver_status':int(lp.status),'solver_message':str(lp.message),'mathematical_feasible':bool(lp.success),'normal_cap_N':None,'zeta_bound':'+/- actual FP32 frictionloss' if bounded else 'exact0','complete_solution':lp.x.tolist() if lp.success else None}
record.update(status='COMPLETE_TWO_EXPLICIT_MIDPOINT_WHOLEHAND_LPs',LPs=results,new_collision_queries=0,new_physics_steps=0,GPU_used=False,physical_reference_qualified=False,wholehand_scope='Same restricted surrogate excludes joint-limit reactions and softcontact constitutive force law; infeasible does not mean all physical motions impossible.',wall_seconds_excluding_imports=time.perf_counter()-start)
(out/'report.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'LPs':results,'report_SHA256':sha(out/'report.json')},indent=2))
