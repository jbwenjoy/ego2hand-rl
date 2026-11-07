"""Small independent basis/linear/CSR contract checks; never solve an LP."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib,json,sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import bspline_control as src
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def oracle(knots,time,order=0):
 last=int(np.flatnonzero(knots<knots[-1])[-1])
 @lru_cache(None)
 def n(i,p,d):
  if d>p:return 0.
  if p==0:return float((knots[i]<=time<knots[i+1]) or (time==knots[-1] and i==last))
  den1=knots[i+p]-knots[i];den2=knots[i+p+1]-knots[i+1]
  if d:return (p/den1*n(i,p-1,d-1) if den1 else 0.)-(p/den2*n(i+1,p-1,d-1) if den2 else 0.)
  return ((time-knots[i])/den1*n(i,p-1,0) if den1 else 0.)+((knots[i+p+1]-time)/den2*n(i+1,p-1,0) if den2 else 0.)
 return np.array([n(i,3,order) for i in range(18)])
def main():
 def forbidden(*a,**kw):raise AssertionError('Independent check cannot optimize/query geometry/physics')
 import scipy.optimize
 scipy.optimize.linprog=scipy.optimize.minimize=scipy.optimize.least_squares=forbidden
 for name in ('mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_collision','mj_geomDistance','mj_contactForce'):setattr(mujoco,name,forbidden)
 source=ROOT/'src/bspline_control.py';source_sha=sha(source);protocol=read(ROOT/'protocols/constrained-prefix-v1.json');pins=protocol['inputs'];checks=[]
 def check(n,ok,d=None):checks.append({'name':n,'pass':bool(ok),'details':d})
 def rejects(fun):
  try:fun()
  except (ValueError,TypeError):return True
  return False
 check('frozen0798 protocol',sha(ROOT/'protocols/constrained-prefix-v1.json')=='0798ea7aa6da6f56684067992906bbba3136f9e15b413f743674f54159ad2e6d')
 check('all20 original source/input pins',len(pins)==20 and all(sha(p['path'])==p['sha256'] for p in pins.values()))
 basis=src.make_basis();times=np.r_[np.arange(61)*.005,[.0037,.0771,.1533,.2972]]
 errors=[]
 for order in range(3):
  actual=basis.evaluate(times,order);ind=np.array([oracle(basis.knots,float(t),order) for t in times]);errors.append(float(abs(actual-ind).max()))
 check('Cox-deBoor independent basis derivatives0to2',errors[0]<1e-12 and errors[1]<1e-9 and errors[2]<1e-7,errors)
 B=basis.evaluate(times);check('nonnegative partition/end clamping',B.min()>=-1e-15 and abs(B.sum(axis=1)-1).max()<1e-12 and np.array_equal(basis.evaluate([0])[0],np.r_[1.,np.zeros(17)]) and np.array_equal(basis.evaluate([.3])[0],np.r_[np.zeros(17),1.]))
 check('no extrapolation/NaN/unsupported derivative',rejects(lambda:basis.evaluate([-.001])) and rejects(lambda:basis.evaluate([.301])) and rejects(lambda:basis.evaluate([np.nan])) and rejects(lambda:basis.evaluate([.1],3)))
 with np.load(pins['canonical_H123_q0']['path'],allow_pickle=False) as z:q0=z['joint_pos_rad'][0].copy()
 m=mujoco.MjModel.from_binary_path(pins['model']['path']);j=np.array([m.joint(f'robot/right_finger{f}_joint{k}').id for f in range(1,6) for k in range(1,5)]);a=np.array([np.flatnonzero(m.actuator_trnid[:,0]==k).item() for k in j]);lo,hi=m.jnt_range[j].T
 with np.load(pins['actual_context']['path'],allow_pickle=False) as z:kp=z['actuator_gainprm'][0,a,0].copy();bias=z['actuator_biasprm'][0,a,:3].copy()
 params=SimpleNamespace(kp=kp,bias=bias,target_lower=np.maximum(lo,m.actuator_ctrlrange[a,0]).astype(np.float32),target_upper=np.minimum(hi,m.actuator_ctrlrange[a,1]).astype(np.float32),actuator_force_bounds=m.actuator_forcerange[a].astype(np.float32),joint_force_bounds=m.jnt_actfrcrange[j].astype(np.float32))
 check('actual unitgear/fixedgain/affine/dyntype0',np.array_equal(m.actuator_gear[a],np.tile([1.,0,0,0,0,0],(20,1))) and np.all(m.actuator_gaintype[a]==mujoco.mjtGain.mjGAIN_FIXED) and np.all(m.actuator_biastype[a]==mujoco.mjtBias.mjBIAS_AFFINE) and np.all(m.actuator_dyntype[a]==mujoco.mjtDyn.mjDYN_NONE))
 design=read(pins['static_preload_design']['path']);pdu=np.array(design['u0_rad'],np.float32);tau=kp*pdu+(bias[:,0]+bias[:,1]*q0.astype(np.float32))
 contract=src.build_linear_contract(q0,params,tau,lo,hi);delta=.0003*np.sin(np.arange(340)*.17).reshape(17,20);delta[:,3]=abs(delta[:,3]);x=np.tile(q0,17)+delta.ravel();delta=x-contract['x0']
 check('340 free fixed P0/repeatq0 exact',np.array_equal(src.evaluate_controls(contract['x0'],q0,[0,.3]),np.repeat(q0[None],2,axis=0)) and contract['A_ub'].shape==(5960,340))
 q=np.array([q0+oracle(basis.knots,float(t))[1:]@delta.reshape(17,20) for t in src.BREAKS[:15]]);v=np.array([oracle(basis.knots,float(t),1)[1:]@delta.reshape(17,20) for t in src.BREAKS[:15]])
 command=((tau.astype(float)-bias[:,0].astype(float))-bias[:,1].astype(float)*q-bias[:,2].astype(float)*v)/kp.astype(float)
 got=(contract['command_matrix']@delta+contract['command_baseline']).reshape(15,20)
 check('CSR command linear map vs independent PD algebra',abs(command-got).max()<1e-12,float(abs(command-got).max()))
 bern=np.array([[(1-s)**3,3*s*(1-s)**2,3*s*s*(1-s),s**3] for s in (0.,.23,.71,1.)]);force_errors={}
 for name,U in [('tracking',command),('hold',np.repeat(command[:1],15,axis=0))]:
  controls=[]
  for interval in range(15):
   values=[]
   for s in (0.,.23,.71,1.):
    t=float(src.BREAKS[interval]+s*(src.BREAKS[interval+1]-src.BREAKS[interval]));Q=q0+oracle(basis.knots,t)[1:]@delta.reshape(17,20);V=oracle(basis.knots,t,1)[1:]@delta.reshape(17,20)
    values.append(kp*U[interval]+bias[:,0]+bias[:,1]*Q+bias[:,2]*V)
   controls.append(np.linalg.solve(bern,np.array(values)))
  got=(contract[name+'_force_matrix']@delta+contract['force_baseline']).reshape(15,4,20);force_errors[name]=float(abs(got-np.array(controls)).max())
 check('wholeinterval rawPD Bernstein vs four independent samples',max(force_errors.values())<1e-11,force_errors)
 centered=contract['A_ub']@delta-contract['b_centered'];absolute=contract['A_ub']@x-contract['b_ub']
 check('centered/absolute RHS equal',abs(centered-absolute).max()<1e-11,float(abs(centered-absolute).max()))
 lp=src.phase0_l1_problem(contract);epigraph=abs(delta)+1e-5;z=np.r_[x,epigraph]
 check('phase0 matrices only/absolutex/L1epigraph',lp['A_ub'].shape==(6640,680) and lp['c'].shape==(680,) and len(lp['bounds'])==680 and np.max((lp['A_ub']@z-lp['b_ub'])[5960:])<=0 and np.array_equal(lp['c'],np.r_[np.zeros(340),np.ones(340)]))
 d=np.cos(np.arange(340)*.29);FD=[]
 for eps in (1e-6,5e-7):
  def oracle_u(xi):
   dd=(xi-contract['x0']).reshape(17,20);Q=np.array([q0+oracle(basis.knots,float(t))[1:]@dd for t in src.BREAKS[:15]]);V=np.array([oracle(basis.knots,float(t),1)[1:]@dd for t in src.BREAKS[:15]]);return ((tau-bias[:,0])-bias[:,1]*Q-bias[:,2]*V)/kp
  fd=(oracle_u(x+eps*d)-oracle_u(x-eps*d))/(2*eps);FD.append(float(abs(fd.ravel()-contract['command_matrix']@d).max()))
 check('CSR command directionalFD twoeps',max(FD)<1e-7,FD)
 badparams=SimpleNamespace(**vars(params));badparams.actuator_force_bounds=np.tile(np.array([1.,0.],np.float32),(20,1))
 check('reject badx/q0 outside harddomain/emptycap',rejects(lambda:src.unpack_controls(np.zeros(300),q0)) and rejects(lambda:src.build_linear_contract(lo-.01,params,tau,lo,hi)) and rejects(lambda:src.build_linear_contract(q0,badparams,tau,lo,hi)))
 check('Root margins and units unchanged',contract['margins']['target_domain_rad']==2e-6 and contract['margins']['raw_force_Nm']==2e-6 and contract['margins']['rate_rad_s']==1e-4 and contract['margins']['planning_rate_rad_s']==14.9999 and protocol['planning_margins']['generated_depth_max_m']==.002999)
 check('source unmodified during independent check',sha(source)==source_sha)
 out=ROOT/'outputs/independent_basis_linear_cpu_v2';out.mkdir(parents=True,exist_ok=False);report={'status':'PASS_NARROW_BASIS_LINEAR_CSR_NO_SOLVE' if all(c['pass'] for c in checks) else 'FAIL_NARROW_BASIS_LINEAR_CSR','checks':checks,'source':{'path':str(source),'sha256':source_sha},'checker_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'protocols/constrained-prefix-v1.json'),'version_note':'v2 adds the explicit empty force-domain negative implied by the v1 check label; v1 numeric results retained. No source/criteria or scientific data change.','LP_calls':0,'optimizer_calls':0,'geometry_queries':0,'physics_steps':0,'GPU_used':False,'scope':'Cox-deBoor basis and actual-PD affine/Bernstein CSR algebra only. Synthetic tiny control perturbation, no phase0 solution or scientific candidate. Contact derivatives, unknown gaps and future actual dynamics remain unqualified.'}
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(report,indent=2));assert all(c['pass'] for c in checks)
if __name__=='__main__':main()
