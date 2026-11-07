"""Focused support-feature API fixture, not a repeat of the core geometry tests."""
import itertools,json
from pathlib import Path
import numpy as np
from query_interface_v2 import capsule_convex,convex_convex,jsonable


def main():
    box=np.array(list(itertools.product([-1.,1.],repeat=3)))
    cap=capsule_convex([2,-.5,0],[2,.5,0],.1,box)
    assert abs(cap['signed_gap_m']-.9)<1e-12 and np.allclose(cap['normal_world'],[1,0,0],atol=1e-12)
    assert len(cap['object_support_points_world_m'])==4 and len(cap['hand_support_points_world_m'])==2
    assert np.allclose(cap['object_support_points_world_m'][:,0],1.) and np.allclose(cap['hand_support_points_world_m'][:,0],1.9)
    assert cap['support_feature_nonunique'] is True and cap['closest_surface_witness_pair'] is None
    mesh=convex_convex(.3*box+[3,0,0],box)
    assert abs(mesh['signed_gap_m']-1.7)<1e-12
    assert len(mesh['object_support_points_world_m'])==len(mesh['hand_support_points_world_m'])==4
    assert np.allclose(mesh['hand_support_points_world_m'][:,0],2.7)
    path=Path(__file__).with_name('interface_fixture_v2.json')
    with path.open('x',encoding='utf-8') as f:json.dump({'schema':'convex_query_support_feature_fixture:2','passed':True,'capsule':jsonable(cap),'convex':jsonable(mesh),'model_FK_nativeDistance_physics_GPU_calls':0},f,indent=2)
    print(json.dumps({'output':str(path),'passed':True}))


if __name__=='__main__':main()
