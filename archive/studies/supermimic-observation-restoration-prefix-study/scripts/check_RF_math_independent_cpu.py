"""Independent synthetic R/F algebra and acceptance checks; no solve/query."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from restoration_prefix import math as rf
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 source=ROOT/'src/restoration_prefix/math.py';source_sha=sha(source);checks=[]
 def check(n,ok,d=None):checks.append({'name':n,'pass':bool(ok),'details':d})
 def rejects(f):
  try:f()
  except ValueError:return True
  return False
 # Synthetic arrays have no observation, reference, collision or force meaning.
 A=sparse.csr_matrix(([1.,-1.],([0,1],[0,0])),shape=(2,340));contract={'A_ub':A,'b_centered':np.ones(2)*.5,'x0':np.zeros(340),'lower':np.ones(340)*-1,'upper':np.ones(340)}
 seed=np.zeros(340);depth=np.ones(122);depth[0]=3.2;z=rf.restoration_z0(seed,depth)
 check('R initialslack maxzero plus1e-6mm',z.shape==(341,) and z[-1]==max(0.,float((depth-2.999).max()))+1e-6 and np.all(rf.restoration_geometry_slack(z,depth)>0))
 below=np.ones(122);check('R already-feasible seed still nonnegative buffer',rf.restoration_z0(seed,below)[-1]==1e-6)
 zz=z.copy();zz[-1]=0
 check('R slackzero/objective not primal certificate',rf.restoration_objective(zz)==0 and not rf.primal_audit(zz[:340],depth,contract)['passed'])
 C=sparse.csr_matrix((np.array([.4,-.3]),([0,1],[0,1])),shape=(122,340));D=rf.restoration_geometry_jacobian(C);direction=np.sin(np.arange(341)*.1);fd=[]
 for eps in (1e-6,5e-7):
  fp=rf.restoration_geometry_slack(z+eps*direction,depth+C@(eps*direction[:340]));fm=rf.restoration_geometry_slack(z-eps*direction,depth-C@(eps*direction[:340]));fd.append(float(abs((fp-fm)/(2*eps)-D@direction).max()))
 check('R slack Jacobian [-D,+1] twoeps',max(fd)<1e-8,fd)
 check('R objective gradient only slack',np.array_equal(rf.restoration_gradient(z),np.r_[np.zeros(340),1.]))
 lb,ub=rf.restoration_box(seed,contract['lower'],contract['upper']);check('R seed trustbox .05 only',np.array_equal(lb,np.full(340,-.05)) and np.array_equal(ub,np.full(340,.05)))
 tol=np.ones(122);tol[0]=2.999+.5e-9;fine=rf.primal_audit(seed,tol,contract);tol[0]=2.999+2e-9
 check('declared depthtol distinct from exactraw',fine['passed'] and not fine['exact_raw_all_pass'] and not rf.primal_audit(seed,tol,contract)['passed'])
 loose=seed.copy();loose[0]=.08;qp=rf.qp_contract(loose,np.ones(122),sparse.csr_matrix((122,340)),contract)
 check('F originalbox and .02 step do not inherit Rbox',np.all(qp['lower']==-.02) and np.all(qp['upper']==.02) and loose[0]>.05)
 rows=np.r_[np.arange(340),340+np.arange(340)];cols=np.r_[np.arange(340),np.arange(340)];vals=np.r_[np.ones(340),np.ones(340)*.3];J=sparse.csr_matrix((vals,(rows,cols)),shape=(6596,340));r=np.sin(np.arange(6596)*.07);g,H=rf.gauss_newton_model(r,J);step=np.cos(np.arange(340)*.11)*.001
 quad=.5*np.linalg.norm(r+J@step)**2+.5*rf.GN_MU*(step@step)-.5*(r@r);ind=float(g@step+.5*step@H@step)
 check('GN gradient/Hessian identity withmu1e-4',np.max(abs(g-J.T@r))==0 and np.max(abs(H-(J.T@J).toarray()-1e-4*np.eye(340)))<1e-15 and abs(quad-ind)<1e-10,abs(quad-ind))
 check('GN H symmetric positive definite',np.max(abs(H-H.T))==0 and np.linalg.eigvalsh(H).min()>0)
 d=np.zeros(340);d[0]=.02;g0=np.zeros(340);g0[0]=-1;current_depth=np.ones(122);current_depth[0]=2.989;zeroC=sparse.csr_matrix((122,340));qp=rf.qp_contract(seed,current_depth,zeroC,contract)
 check('inexact QP still needs explicit feasible descent direction',rf.qp_direction_audit(d,g0,qp)['eligible'] and not rf.qp_direction_audit(d,-g0,qp)['eligible'] and not rf.qp_direction_audit(np.zeros(340),g0,qp)['eligible'])
 big=d.copy();big[0]=.021;check('QP truststep/finite rejection',not rf.qp_direction_audit(big,g0,qp)['eligible'] and rejects(lambda:rf.qp_direction_audit(np.full(340,np.nan),g0,qp)))
 # A nonlinear synthetic depth that an exact zero linearization misses.
 trials=[];accepted=None
 for alpha in rf.ALPHAS:
  xx=seed+alpha*d;actual=np.ones(122);actual[0]=2.989+400*xx[0]**2;primal=rf.primal_audit(xx,actual,contract);cost=.5*(xx[0]-1)**2;decision=rf.armijo_decision(.5,cost,alpha,float(g0@d),primal,trial_x_changed=bool(np.any(xx!=seed)));trials.append({'alpha':alpha,'depth_mm':float(actual[0]),**decision})
  if decision['accepted']:accepted=alpha;break
 check('actual nonlinear recheck rejects full/half then acceptsquarter',accepted==.25 and not trials[0]['primal_pass'] and not trials[1]['primal_pass'] and trials[2]['primal_pass'],trials)
 check('accepted trial retains samplefeasible actual strict descent',trials[-1]['accepted'] and trials[-1]['actual_cost_strictly_decreases'] and trials[-1]['Armijo_pass'])
 ppass={'passed':True};pfail={'passed':False}
 check('Armijo rejects unchanged roundedRHS or infeasible decrease',not rf.armijo_decision(1.,1.,1.,-1e-300,ppass,trial_x_changed=True)['accepted'] and not rf.armijo_decision(.5,.1,1.,-.1,pfail,trial_x_changed=True)['accepted'] and not rf.armijo_decision(.5,.1,1.,-.1,ppass,trial_x_changed=False)['accepted'])
 check('strict decrease alone cannot replace Armijo',not rf.armijo_decision(.5,.49999,1.,-1.,ppass,trial_x_changed=True)['accepted'])
 check('sixalpha/shape/nonfinite negative inputs',rejects(lambda:rf.armijo_decision(.5,.4,.03,-.2,ppass,trial_x_changed=True)) and rejects(lambda:rf.restoration_z0(np.zeros(300),depth)) and rejects(lambda:rf.gauss_newton_model(r,np.zeros((3,4)))) and not rf.armijo_decision(.5,np.nan,1.,-.1,ppass,trial_x_changed=True)['accepted'])
 check('source unchanged during check',sha(source)==source_sha)
 out=ROOT/'outputs/independent_RF_math_cpu_v2';out.mkdir(parents=True,exist_ok=False);report={'status':'PASS_SYNTHETIC_RF_MATH_ONLY' if all(c['pass'] for c in checks) else 'FAIL_SYNTHETIC_RF_MATH','checks':checks,'source':{'path':str(source),'sha256':source_sha},'checker_sha256':sha(__file__),'formal_seed_or_reference_loaded':False,'SLSQP_or_LP_calls':0,'geometry_queries':0,'physics_steps':0,'GPU_used':False,'scope':'Synthetic helper validation only. No restoration point, QP direction, line-search decision or accepted scientific state was produced. Preserved feasibility means the declared arithmetic tolerance on sampled reference constraints, not exact raw/continuous/actual dynamic safety.'}
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(report,indent=2));assert all(c['pass'] for c in checks)
if __name__=='__main__':main()
