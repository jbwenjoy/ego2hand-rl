import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np


class RewardSerializationTests(unittest.TestCase):
    def load(self,name):
        path=Path(__file__).resolve().parents[1]/'tools'/name
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_V2_serializes_real_numpy_summary_counts_without_numeric_changes(self):
        old=self.load('audit_backend_reward_chain.py');new=self.load('audit_backend_reward_chain_v2.py')
        values=[];meta=[];groups=[]
        for e in [.002,.005]:
            p=[];m=[]
            for sign,ids in [(1,p),(-1,m)]:
                for repeat in [0,1]:
                    ids.append(len(values));values.append(np.arange(1,11,dtype=np.float64)*sign*e);meta.append(dict(valid=True))
            groups.append(dict(source_name='numeric_fixture',direction=0,epsilon=e,rows={'1':p,'-1':m},complete=True,qualified=True))
        args=(np.asarray(values),meta,groups,['numeric_fixture'])
        a=old.summarize(*args);b=new.summarize(*args)
        with self.assertRaises(TypeError):json.dumps(a)
        actual=json.dumps(b,sort_keys=True,allow_nan=False)
        expected=json.dumps(a,sort_keys=True,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else None)
        self.assertEqual(actual,expected)
        self.assertIs(type(b['geometry_absolute_share_of_tracking_geometry_support']['share_above_10_percent']),int)
