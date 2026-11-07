"""Compiled-collider axes and bounded contact-manifold restoration proposals.

This stage changes initialization geometry; it never repairs a running state.
Restoration is purely kinematic. Native force scoring and qualification must
follow separately. Unknown geometry cannot be a successful restoration.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.optimize import least_squares
from .initialization import SearchBudget


OFFSETS_M=(-.08,-.06,-.04,-.02,.02,.04,.06,.08)


def collider_axes(vertices,rotation):
    vertices=np.asarray(vertices,dtype=float);rotation=np.asarray(rotation,dtype=float)
    center=vertices.mean(0);values,vectors=np.linalg.eigh((vertices-center).T@(vertices-center)/len(vertices))
    if values[0]<=0 or values[2]<=1.2*values[1]:raise ValueError('Ambiguous/full-dimensional long axis')
    axes={}
    for name,index in [('long',2),('width',1),('normal',0)]:
        v=vectors[:,index].copy()
        if v[np.argmax(abs(v))]<0:v=-v
        axes[name]=dict(local=v.tolist(),world=(rotation@v).tolist(),extent_m=float(np.ptp(vertices@v)))
    return dict(axes=axes,covariance_eigenvalues_m2=values.tolist(),source='All actual compiled convex-hull vertices, equal weights; sign canonicalized in geom-local axes. Shape descriptor, not ego gravity.')


class GeometryRestoration:
    def __init__(self,model,audit,q_seed):
        import mujoco
        self.mj=mujoco;self.model=model;self.audit=audit;self.q_seed=np.array(q_seed,dtype=float);self.data=mujoco.MjData(model)
        self.lo=np.maximum(-1.,(model.jnt_range[:12,0]+.005-self.q_seed[:12])/.35)
        self.hi=np.minimum(1.,(model.jnt_range[:12,1]-.005-self.q_seed[:12])/.35)
        if np.any(self.lo>=0) or np.any(self.hi<=0):raise ValueError('Original joints interior to restoration domain')
        self.counts=dict(MjData=1,kinematics=0,geometry_audits=0,proposal_calls=0,secant_columns=0,forward=0,integration_steps=0)

    def state(self,z):
        z=np.asarray(z,dtype=float)
        if z.shape!=(12,) or np.any(z<self.lo-1e-14) or np.any(z>self.hi+1e-14):raise ValueError('Legal restoration parameters')
        q=self.q_seed.copy();q[:12]+=.35*z
        self.mj.mj_resetData(self.model,self.data);self.data.qpos[:]=q;self.data.qvel[:]=0;self.data.ctrl[:]=0
        self.mj.mj_kinematics(self.model,self.data);self.counts['kinematics']+=1
        result=self.audit.evaluate(self.data);self.counts['geometry_audits']+=1
        return q,result

    @staticmethod
    def residual(z,geometry):
        if geometry['unknown']:raise ValueError('Unknown convex geometry in restoration')
        pad=[row['gap_m'] for row in geometry['rows'] if row['pad']]
        other=[row['lower_bound_m'] for row in geometry['rows'] if not row['pad']]
        if len(pad)!=3 or len(other)!=208:raise ValueError('Full22/189 geometry coverage')
        return np.r_[(np.array(pad)+.000075)/.0005,np.maximum(-np.array(other),0)/.0002,.002*np.asarray(z)]

    def restore(self,path,*,maximum_calls=180,maximum_seconds=12.):
        stream=Path(path).open('x',encoding='utf8');tick=time.perf_counter();best=None;last=None;stop=None
        def fun(z):
            nonlocal best,last
            if self.counts['proposal_calls']>=maximum_calls or time.perf_counter()-tick>=maximum_seconds:raise SearchBudget('Declared restoration budget')
            q,g=self.state(z);r=self.residual(z,g);score=float(r@r);self.counts['proposal_calls']+=1
            entry=dict(call=self.counts['proposal_calls'],z12=np.asarray(z).tolist(),score=score,geometry_pass=g['passed'],pad_gaps_m=[x['gap_m'] for x in g['rows'] if x['pad']])
            stream.write(json.dumps(entry,allow_nan=False)+'\n');last=(np.asarray(z).copy(),r.copy())
            if best is None or score<best['score']:best=entry.copy()
            return r
        def jac(z):
            base=last[1] if last is not None and np.array_equal(last[0],z) else fun(z)
            matrix=np.empty((len(base),12))
            for j in range(12):
                step=.001 if z[j]+.001<=self.hi[j] else -.001;p=np.asarray(z).copy();p[j]+=step
                matrix[:,j]=(fun(p)-base)/step;self.counts['secant_columns']+=1
            return matrix
        try:
            result=least_squares(fun,np.zeros(12),jac=jac,bounds=(self.lo,self.hi),max_nfev=20,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            stop=dict(kind='optimizer',status=int(result.status),message=result.message)
        except (SearchBudget,ValueError) as error:stop=dict(kind=type(error).__name__,message=str(error))
        finally:stream.close()
        if best is None:return dict(best=None,stop=stop,elapsed_s=time.perf_counter()-tick,counts=self.counts.copy(),qualified=False)
        q,g=self.state(np.array(best['z12']))
        return dict(best=best,stop=stop,elapsed_s=time.perf_counter()-tick,counts=self.counts.copy(),qpos27=q,geometry=g,qualified=g['passed'])


def compose_deviation(local,restored_q,original_q,seed_shift):
    """Compose the old initializer's local reference delta with global slide."""
    out=dict(local);R=np.array(local['proposed_reference_left_transform_R']);shift=np.asarray(seed_shift)
    out['q20_delta_rad']=(np.array(local['q20_delta_rad'])+np.asarray(restored_q)[:20]-np.asarray(original_q)[:20]).tolist()
    out['phone_root_delta_world_m']=(np.array(local['phone_root_delta_world_m'])+shift).tolist()
    out['proposed_reference_left_transform_t_m']=(np.array(local['proposed_reference_left_transform_t_m'])+R@shift).tolist()
    out['global_slide_world_m']=shift.tolist();out['scope']='Distinct initialization relative to original parent, including restoration and global slide. No rebuilt task or measured ego-gravity claim.'
    return out
