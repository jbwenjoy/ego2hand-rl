"""Analytic geometry fixtures, no model/physics/native-distance calls."""
import itertools
import json
from pathlib import Path
import time
import numpy as np
from convex_query import ConvexPolytope,jsonable


def run():
    started=time.perf_counter()
    box=np.array(list(itertools.product([-1.,1.],repeat=3)))
    poly=ConvexPolytope(box)
    cases=[]
    def check(name,actual,expected):
        error=abs(actual['signed_translation_margin_m']-expected)
        if error>1e-10:raise AssertionError((name,error,actual,expected))
        cases.append({'name':name,'expected_margin_m':expected,'error_m':error,'actual':jsonable(actual)})
    check('sphere_outside_face',poly.capsule([3,0,0],[3,0,0],.5),1.5)
    check('sphere_outside_corner',poly.capsule([2,2,2],[2,2,2],.25),np.sqrt(3)-.25)
    check('sphere_inside_box',poly.capsule([0,0,0],[0,0,0],.2),-1.2)
    check('sphere_touching_box',poly.capsule([1.2,0,0],[1.2,0,0],.2),0.)
    check('capsule_parallel_outside',poly.capsule([2,-.5,0],[2,.5,0],.1),.9)
    check('capsule_axis_crossing_box',poly.capsule([-2,0,0],[2,0,0],.2),-1.2)
    check('radius_zero_corner',poly.capsule([2,2,2],[2,2,2],0.),np.sqrt(3))
    small=box*.3
    check('two_boxes_separated',poly.convex(small+[3,0,0]),1.7)
    check('two_boxes_penetrating',poly.convex(small+[.9,0,0]),-.4)
    translations=np.array([[.03,-.02,.01],[-.1,.2,-.05],[.001,0,0]])
    lipschitz=[]
    for a,b,r in [(np.array([2.,-.5,0]),np.array([2.,.5,0]),.1),
                  (np.array([0.,0,0]),np.array([0.,0,0]),.2),
                  (np.array([1.2,0.,0]),np.array([1.2,0.,0]),.2)]:
        old=poly.capsule(a,b,r)['signed_translation_margin_m']
        for t in translations:
            new=poly.capsule(a+t,b+t,r)['signed_translation_margin_m']
            delta=abs(new-old);bound=float(np.linalg.norm(t))
            assert delta<=bound+1e-10
            lipschitz.append({'translation_m':t.tolist(),'absolute_margin_change_m':delta,'bound_m':bound})
    # Rigid rotation+translation invariance is independent of any engine convention.
    angle=.713;c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    t=np.array([.4,-.7,.2]);rotated=ConvexPolytope(box@R.T+t)
    check('common_rigid_transform',rotated.capsule(R@np.array([2.,-.5,0])+t,R@np.array([2.,.5,0])+t,.1),.9)
    rejected=False
    try:ConvexPolytope(np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],float))
    except Exception:rejected=True
    assert rejected
    return {'schema':'independent_convex_query_fixtures:1','passed':True,'analytic_cases':cases,
            'translation_Lipschitz_cases':lipschitz,'degenerate_coplanar_hull_rejected_not_zero':True,
            'elapsed_seconds':time.perf_counter()-started,'model_FK_nativeDistance_physics_GPU_calls':0}


if __name__=='__main__':
    result=run();path=Path(__file__).with_name('fixture_results_v1.json')
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'output':str(path),'passed':result['passed'],'analytic_cases':len(result['analytic_cases']),'Lipschitz_cases':len(result['translation_Lipschitz_cases'])}))
