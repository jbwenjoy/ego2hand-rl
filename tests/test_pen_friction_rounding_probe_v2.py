import unittest
import numpy as np
import warp as wp
from supermimic_research.runtime.pen_friction_rounding_probe_v2 import (
    explicit_rounding,original_single,original_triple,identity_single,identity_triple,guarded_single,guarded_triple)


@wp.kernel(enable_backward=False)
def evaluate(j:wp.array[float],v:wp.array[float],d:wp.array[float],f:wp.array[float],a:wp.array[float],fields:wp.array2d[float],single:wp.array3d[float],triple:wp.array4d[float]):
    i=wp.tid(); n=j.shape[0]
    values=explicit_rounding(j[i],v[i],d[i],f[i],a[i])
    for k in range(12): fields[i,k]=values[k]
    x=original_single(i,a[i],0,n,d[i],f,j[i],v[i]); y=identity_single(i,a[i],0,n,d[i],f,j[i],v[i]); z=guarded_single(i,a[i],0,n,d[i],f,j[i],v[i])
    a0,b0,c0=original_triple(i,a[i],a[i]*0.5,a[i]*1.5,0,n,d[i],f,j[i],v[i])
    a1,b1,c1=identity_triple(i,a[i],a[i]*0.5,a[i]*1.5,0,n,d[i],f,j[i],v[i])
    a2,b2,c2=guarded_triple(i,a[i],a[i]*0.5,a[i]*1.5,0,n,d[i],f,j[i],v[i])
    for k in range(3):
        single[i,0,k]=x[k]; single[i,1,k]=y[k]; single[i,2,k]=z[k]
        triple[i,0,0,k]=a0[k]; triple[i,0,1,k]=b0[k]; triple[i,0,2,k]=c0[k]
        triple[i,1,0,k]=a1[k]; triple[i,1,1,k]=b1[k]; triple[i,1,2,k]=c1[k]
        triple[i,2,0,k]=a2[k]; triple[i,2,1,k]=b2[k]; triple[i,2,2,k]=c2[k]


class FrictionRoundingTests(unittest.TestCase):
    def fixture(self):
        wp.init()
        rows=np.asarray([[7.1270323,5.9493203,.0028098072,.036735542,1.0000001],[-1.,4.,1.,.25,1.],
                         [512.,-.001,1.,.03,1.],[-512.,.001,1.,.03,1.],[-1.,.2,1.,10.,.5]],np.float32)
        inputs=[wp.array(rows[:,i].copy(),dtype=float,device='cpu') for i in range(5)]; n=len(rows)
        out=[wp.zeros((n,12),device='cpu'),wp.zeros((n,3,3),device='cpu'),wp.zeros((n,3,3,3),device='cpu')]
        wp.launch(evaluate,dim=n,inputs=inputs,outputs=out,device='cpu')
        for i,a in enumerate(inputs): self.assertEqual(a.numpy().tobytes(),rows[:,i].tobytes())
        return rows,[x.numpy() for x in out]

    def test_explicit_rounding_fields_against_fp32_stepwise_reference(self):
        rows,(fields,_,_)=self.fixture(); F=np.float32
        for row,values in zip(rows,fields):
            j,v,d,f,a=row; x=F(float(a)*float(v)+float(j)); xu=F(j+F(a*v)); r=F(f/d)
            def terms(t):
                return (F(F(.5*d)*t),t) if -r<t<r else (f,F(F(-.5*r)+(-t if t<=-r else t)))
            p0,t0=terms(j); p1,t1=terms(x); c0=F(p0*t0); c1=F(p1*t1)
            reference=np.asarray([x,xu,r,p0,t0,c0,p1,t1,c1,F(c1-c0),F(float(c1)-float(p0)*float(t0)),F(float(p1)*float(t1)-float(c0))],np.float32)
            self.assertEqual(reference.tobytes(),values.tobytes())
        self.assertNotEqual(fields[0,9],fields[0,10])

    def test_identity_and_guarded_fallback_derivative_preservation_distinct_alphas(self):
        rows,(_,single,triple)=self.fixture()
        self.assertEqual(single[:,0].tobytes(),single[:,1].tobytes()); self.assertEqual(triple[:,0].tobytes(),triple[:,1].tobytes())
        self.assertEqual(single[:,0,1:].tobytes(),single[:,2,1:].tobytes()); self.assertEqual(triple[:,0,:,1:].tobytes(),triple[:,2,:,1:].tobytes())
        for i,(j,v,d,f,a) in enumerate(rows.astype(float)):
            for k,alpha in enumerate([a,float(np.float32(a)*np.float32(.5)),float(np.float32(a)*np.float32(1.5))]):
                r=f/d; x=j+alpha*v
                same=(-r<j<r and -r<x<r) or (j<=-r and x<=-r) or (j>=r and x>=r)
                if not same: self.assertEqual(triple[i,0,k].tobytes(),triple[i,2,k].tobytes())


if __name__=='__main__': unittest.main()
