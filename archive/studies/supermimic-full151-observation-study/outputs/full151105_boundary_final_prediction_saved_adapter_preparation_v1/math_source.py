"""Saved final-component arithmetic: scalar congruence/pivot, no model imports."""
import hashlib,importlib.util,math
from pathlib import Path
import numpy as np
P=Path(__file__).with_name('boundary_impulse_independent_math_v1.py')
if hashlib.sha256(P.read_bytes()).hexdigest()!='48eb17c899491f303d681c862fbaebb512f51f6eab5d10bc03118e01ecf8bef4':raise ValueError('Frozen scalar solver changed')
_s=importlib.util.spec_from_file_location('boundary_scalar_reference',P);S=importlib.util.module_from_spec(_s);_s.loader.exec_module(S)

def fmean(x):
    a=np.asarray(x,dtype='f8');S.require(a.size and np.isfinite(a).all(),'Finite nonempty mean operands');return math.fsum(float(v) for v in a.flat)/a.size

def probability(logits):
    a=np.asarray(logits,dtype='f8');S.require(np.isfinite(a).all(),'Finite logits');out=np.empty_like(a)
    for index in np.ndindex(a.shape):
        l=float(a[index]);e=math.exp(-abs(l));out[index]=1/(1+e) if l>=0 else e/(1+e)
    return out

def bce_elements(logits,labels):
    a=np.asarray(logits,dtype='f8');y=np.asarray(labels);S.require(a.shape==y.shape and y.dtype==np.bool_,'Matching binary labels');out=np.empty_like(a)
    for index in np.ndindex(a.shape):
        l=float(a[index]);out[index]=math.log1p(math.exp(-abs(l)))+(max(l,0.) if not y[index] else max(-l,0.))
    return out

def vector_statistics(value,unit):
    a=np.asarray(value,dtype='f8');S.require(a.ndim==2 and len(a) and np.isfinite(a).all(),'Finite per-feature statistics');squares=a*a;abs_a=abs(a)
    return {'unit':unit,'component_RMSE':math.sqrt(fmean(squares)),'vector_RMS':math.sqrt(math.fsum(math.fsum(float(v) for v in row) for row in squares)/len(a)),
      'component_MAE':fmean(abs_a),'maximum_absolute':float(abs_a.max()),'maximum_vector_norm':max(math.sqrt(math.fsum(float(v) for v in row)) for row in squares),
      'per_feature_signed_mean':[fmean(a[:,j]) for j in range(a.shape[1])],'per_feature_RMSE':[math.sqrt(fmean(squares[:,j])) for j in range(a.shape[1])],
      'per_feature_MAE':[fmean(abs_a[:,j]) for j in range(a.shape[1])],'per_feature_maximum_absolute':[float(abs_a[:,j].max()) for j in range(a.shape[1])]}

def auxiliary(x,pred,z,scale,mass,Ib,gravity,duration,Jtarget,Ktarget,rv_saved,rw_saved):
    x,pred,z,scale=np.asarray(x),np.asarray(pred),np.asarray(z),np.asarray(scale);n=len(x)
    S.require(x.shape==pred.shape==(n,58) and z.shape==(n,2,6) and scale.shape==(52,) and (scale>0).all(),'Auxiliary full row shapes/scales')
    S.require(all(v.dtype==np.float32 for v in (x,pred,z,scale)) and all(np.isfinite(v).all() for v in (x,pred,z,scale)),'Raw native FP32 operands')
    sv,sw=scale[46:49].astype('f8'),scale[49:52].astype('f8');m=float(mass);S.require(math.isfinite(m) and m>0,'Positive context mass')
    g=np.asarray(gravity,dtype='f8');t=np.asarray(duration,dtype='f8');g=np.broadcast_to(g,(n,3));t=np.broadcast_to(t,(n,))
    keys=('I_palm0_F64','I_palm1_hat_F64','z_target_F64','J_hat_palm_F64','K_hat_palm_F64','r_v_hat_F64','r_w_hat_F64','angular_RHS_F64','omega_from_predicted_impulses_F64','solve_residual_F64');out={k:[] for k in keys}
    for i in range(n):
        I0=S.congruence_native_R(x[i,43:52].astype('f8').reshape(3,3),Ib);I1=S.congruence_native_R(pred[i,43:52].astype('f8').reshape(3,3),Ib)
        target=np.asarray([np.r_[np.asarray(Jtarget[i,c],dtype='f8')/m/sv,S.solve3_partial_pivot(I0,Ktarget[i,c])/sw] for c in range(2)])
        J=np.asarray([m*sv*z[i,c,:3].astype('f8') for c in range(2)]);K=np.asarray([S.matvec(I0,sw*z[i,c,3:].astype('f8')) for c in range(2)])
        v=pred[i,52:55].astype('f8')-x[i,52:55].astype('f8')-np.asarray([math.fsum((float(J[0,j]),float(J[1,j]),m*float(g[i,j])*float(t[i])))/m for j in range(3)])
        p=S.matvec(I0,x[i,55:58].astype('f8'));rhs=np.asarray([math.fsum((float(p[j]),float(K[0,j]),float(K[1,j]))) for j in range(3)])
        omega=S.solve3_partial_pivot(I1,rhs);w=pred[i,55:58].astype('f8')-omega;residual=S.matvec(I1,omega)-rhs
        for k,val in zip(keys,(I0,I1,target,J,K,v,w,rhs,omega,residual)):out[k].append(val)
    out={k:np.asarray(v,dtype='f8') for k,v in out.items()};out['L_imp_F64']=np.asarray(fmean((z.astype('f8')-out['z_target_F64'])**2))
    out['L_cons_F64']=np.asarray(fmean(np.concatenate(((out['r_v_hat_F64']-rv_saved)/sv,(out['r_w_hat_F64']-rw_saved)/sw),axis=1)**2))
    return out

def native_fp32_reduction_evidence(captured,element_operands_FP32):
    a=np.asarray(element_operands_FP32);S.require(a.dtype==np.float32 and a.size and np.isfinite(a).all() and (a>=0).all(),'Saved nonnegative FP32 elements');ref=fmean(a);n=a.size+3;u=2**-24
    bound=n*u/(1-n*u)*abs(ref)+n*float(np.nextafter(np.float32(0),np.float32(1)))
    return {'captured':float(captured),'F64_element_mean':ref,'difference':float(captured)-ref,'conditional_FP32_reduction_envelope':bound,'within_envelope':abs(float(captured)-ref)<=bound,'kernel_bitexact_replay':False}
