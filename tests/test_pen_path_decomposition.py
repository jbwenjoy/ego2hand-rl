import unittest
from unittest.mock import patch
import torch
from supermimic_research.learning.pen_path_decomposition import gradients,values


class PathDecompositionTests(unittest.TestCase):
    def fixture(self):
        class Model:
            def full_prediction(self,x):return x[:,:90],1+2*x[:,81:101].square().sum(-1)
        context=dict(EMA=torch.full((2,20),.1,dtype=torch.float64),reference_joints=torch.zeros(2,20,dtype=torch.float64),encoder_bias=torch.zeros(2,20,dtype=torch.float64),lower=torch.full((2,20),-2.),upper=torch.full((2,20),2.))
        x=torch.full((2,102),.1,dtype=torch.float64);raw=torch.full((2,20),.2,dtype=torch.float64)
        def proxy(x,p,raw,*_):
            v=p[:,81:90].sum(-1)-.02*raw.square().mean(-1)
            return dict(observable_residual=v,observable_support=2*v)
        return Model(),x,raw,context,proxy

    def test_value_and_raw_partial_additivity_with_no_direct_path_term(self):
        model,x,raw,ctx,proxy=self.fixture()
        with patch('supermimic_research.learning.pen_path_decomposition.observable_proxy',side_effect=proxy):r=gradients(model,x,raw,ctx,{},dict(geometry_soft_weight=.25))
        for key in ['values','RAW','PD_partial','RAW_direct_partial']:torch.testing.assert_close(r[key][:,2],r[key][:,0]+r[key][:,1],atol=1e-14,rtol=1e-14)
        self.assertFalse(bool(r['RAW_direct_partial'][:,1].any()));torch.testing.assert_close(r['RAW'][:,1],-.25*4*x[:,81:101,None].transpose(1,2).expand(-1,2,-1)*.25)

    def test_full_direction_matches_actual_raw_and_pd_finite_difference(self):
        model,x,raw,ctx,proxy=self.fixture();d=torch.linspace(-.3,.2,20,dtype=torch.float64);e=1e-5;xp,xm=x.clone(),x.clone();xp[:,81:101]+=.25*e*d;xm[:,81:101]-=.25*e*d
        with patch('supermimic_research.learning.pen_path_decomposition.observable_proxy',side_effect=proxy):
            r=gradients(model,x,raw,ctx,{},dict(geometry_soft_weight=.25));a=values(model,xp,raw+e*d,ctx,{},dict(geometry_soft_weight=.25));b=values(model,xm,raw-e*d,ctx,{},dict(geometry_soft_weight=.25))
        torch.testing.assert_close(r['RAW']@d,(a-b)/(2*e),atol=1e-9,rtol=1e-8)


if __name__=='__main__':unittest.main()
