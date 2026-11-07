"""One closed-form crossed-edge closest-witness fixture; no simulator import."""
import importlib.util,json,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('support_preparation_fixture',ROOT/'prepare_support_v1.py')
a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
P=np.array([[-.02,0,0],[.02,0,0]])
H=np.array([[.003,-.01,.00003],[.003,.01,.00003]])
r=a.closest_support_pair(P,H,np.array([0.,0.,1.]),.00003)
expected_P=np.array([.003,0,0]);expected_H=np.array([.003,0,.00003])
assert np.allclose(r['object_point_world_m'],expected_P,rtol=0,atol=1e-14)
assert np.allclose(r['hand_point_world_m'],expected_H,rtol=0,atol=1e-14)
assert r['rank']==r['unknown_count']==4
assert np.linalg.norm(P.mean(0)-expected_P)>.001  # centroid is intentionally wrong
assert 'mujoco' not in __import__('sys').modules
out=ROOT/'fixture_v1';out.mkdir(exist_ok=False)
obj={'schema':'hand2_support_preparation_closest_witness_fixture:1','passed':1,'total':1,
     'source':a.pin(ROOT/'prepare_support_v1.py'),'checker':a.pin(__file__),
     'meaning':'One crossed-edge analytic witness; verifies pairing instead of arbitrary support centroids. No actual model, LP candidate, FK or physics.',
     'result':a.plain(r)}
a.save(out/'report.json',obj)
print(json.dumps({'passed':1,'total':1,'report':a.pin(out/'report.json')}))
