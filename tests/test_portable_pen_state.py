import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np
import torch
from supermimic_research.runtime.portable_inventory import runtime_inventory,describe_inventory,assign_path,path_value
from supermimic_research.runtime.pen_portable_state import validate_inventory,tensor_bytes
from supermimic_research.mechanics.portable_contact_records import contacts
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from review_pen_native_resume import contact_difference


class Fixture:
    __module__='supermimic_research.testing'
    def __init__(self):
        self.buffer=torch.arange(4,dtype=torch.float32);self.alias=self.buffer
        self.count=0;self.other_count=0;self.options=(3,'fixed');self.values={'n':np.arange(3)}


@dataclass(frozen=True)
class FrozenConstants:
    __module__='supermimic_research.testing'
    hand_alpha:float=.5


def packed(inv):
    return dict(description=describe_inventory(inv),tensors={p:v.clone() for p,v in inv['tensors'].items()},
        arrays={p:v.copy() for p,v in inv['arrays'].items()},scalars=copy.deepcopy(inv['scalars']))


class Array:
    def __init__(self,v):self.value=np.asarray(v)
    def numpy(self):return self.value


def contact_fixture():
    # Reverse contact order to ensure forces index the owning WORLD, not row.
    c=dict(worldid=[1,0,99],geom=[[0,1],[0,1],[-1,-1]],dim=[1,1,999],dist=[0.,0.,np.nan],
        pos=[[0,1.,0],[0,0,0],[np.nan]*3],frame=np.tile(np.eye(3),(3,1,1)),friction=np.ones((3,5)),
        efc_address=[[0,-1,-1,-1,-1,-1],[0,-1,-1,-1,-1,-1],[999]*6])
    d=SimpleNamespace(nacon=Array([2]),naconmax=3,njmax=2,contact=SimpleNamespace(**{k:Array(v) for k,v in c.items()}),
        nefc=Array([1,1]),efc=SimpleNamespace(force=Array([[2.,np.nan],[5.,np.nan]])),
        xipos=Array(np.zeros((2,3,3))),xmat=Array(np.tile(np.eye(3),(2,3,1,1))),overflow=Array([0,0]))
    host=SimpleNamespace(ngeom=2,geom_bodyid=np.array([1,2]),geom=lambda i:SimpleNamespace(name=['robot/right_finger1_tip_sensor_col','object/cube'][i]))
    return SimpleNamespace(opt=SimpleNamespace(cone=0)),d,host


class PortablePenTests(unittest.TestCase):
    def test_aliases_and_repeated_scalar_fields_are_explicit(self):
        f=Fixture();inv=runtime_inventory(f)
        self.assertEqual(len(inv['tensors']),1)
        self.assertTrue(any(r['path']==(('attr','alias'),) and r['target']==(('attr','buffer'),) for r in inv['aliases']))
        self.assertIn((('attr','count'),),inv['scalars']);self.assertIn((('attr','other_count'),),inv['scalars'])
        self.assertFalse(inv['opaque'])

    def test_alias_signature_is_independent_of_attribute_insertion_order(self):
        a=Fixture();b=Fixture()
        a.first=a.buffer;a.second=a.buffer
        b.second=b.buffer;b.first=b.buffer
        inv=runtime_inventory(b);inv['_root']=b
        validate_inventory(inv,packed(runtime_inventory(a)))

    def test_layout_rejection_precedes_array_mutation(self):
        f=Fixture();saved=packed(runtime_inventory(f));f.new_buffer=torch.zeros(2)
        inv=runtime_inventory(f);inv['_root']=f;before=tensor_bytes(f.buffer)
        with self.assertRaisesRegex(ValueError,'closure layout'):validate_inventory(inv,saved)
        self.assertEqual(tensor_bytes(f.buffer),before)

    def test_immutable_tuple_values_cannot_be_silently_reinterpreted(self):
        f=Fixture();saved=packed(runtime_inventory(f));f.options=(4,'fixed')
        inv=runtime_inventory(f);inv['_root']=f
        with self.assertRaisesRegex(ValueError,'immutable tuple'):validate_inventory(inv,saved)

    def test_frozen_constants_are_validated_before_any_copy_and_not_reassigned(self):
        f=Fixture();f.constants=FrozenConstants();inv=runtime_inventory(f);saved=packed(inv);inv['_root']=f
        path=(('attr','constants'),('attr','hand_alpha'))
        assign_path(f,path,.5);validate_inventory(inv,saved)
        saved['scalars'][path]=.7
        with self.assertRaisesRegex(ValueError,'frozen'):validate_inventory(inv,saved)
        self.assertEqual(f.constants.hand_alpha,.5)

    def test_stable_path_restores_scalars_and_dictionary_leaves(self):
        f=Fixture();p=(('attr','values'),('item','n'))
        self.assertIs(path_value(f,p),f.values['n'])
        assign_path(f,(('attr','count'),),17);self.assertEqual(f.count,17)
        assign_path(f,(('attr','options'),('item',0)),3)
        with self.assertRaises(ValueError):assign_path(f,(('attr','options'),('item',0)),9)

    def test_nan_payloads_are_audited_as_bytes(self):
        bits=np.array([0x7fc00001,0x7fc0babe],dtype=np.uint32)
        t=torch.from_numpy(bits.view(np.float32).copy())
        self.assertEqual(tensor_bytes(t),bits.tobytes())
        changed=t.clone();changed[0]=float('nan')
        self.assertNotEqual(tensor_bytes(t),tensor_bytes(changed))

    def test_unknown_mutable_objects_are_listed_not_silently_dropped(self):
        f=Fixture();f.unknown=SimpleNamespace(dynamic=4)
        inv=runtime_inventory(f)
        self.assertEqual(inv['opaque'][0]['kind'],'opaque')
        self.assertEqual(inv['opaque'][0]['type'],'types.SimpleNamespace')

    def test_active_contacts_decode_owner_world_and_ignore_unused_garbage(self):
        model,data,host=contact_fixture();r=contacts(model,data,host,2)
        self.assertTrue(all(r['flags'].values()))
        np.testing.assert_array_equal(r['local_force6'][:,0],[5,2])
        np.testing.assert_array_equal(r['Fn6'][:,0],[2,5])
        np.testing.assert_array_equal(r['wrench_WORLD_COM'][:,0],[2,5])
        self.assertEqual(r['wrench_WORLD_COM'][1,5],-5.)

    def test_invalid_world_and_active_constraint_index_remain_failed(self):
        model,data,host=contact_fixture();data.contact.worldid.value[0]=9
        self.assertFalse(contacts(model,data,host,2)['flags']['valid_world_ids'])
        data.contact.worldid.value[0]=1;data.contact.efc_address.value[0,0]=1
        self.assertFalse(contacts(model,data,host,2)['flags']['valid_addresses'])

    def test_contact_review_matches_reordered_rows_within_owning_world(self):
        model,data,host=contact_fixture();a=contacts(model,data,host,2);b=copy.deepcopy(a)
        for key in ['worldid','geom','dim','dist','pos','frame','friction','efc_address','local_force6']:
            b[key]=b[key][::-1].copy()
        delta=contact_difference(a,b)
        self.assertTrue(delta['same_contact_groups']);self.assertEqual(delta['matched_force_max_N'],0.)
        self.assertEqual(delta['matched_position_max_m'],0.)

    def test_contact_review_cannot_match_force_from_another_world(self):
        model,data,host=contact_fixture();a=contacts(model,data,host,2);b=copy.deepcopy(a)
        b['local_force6']=b['local_force6'][::-1].copy()
        delta=contact_difference(a,b)
        self.assertEqual(delta['matched_force_max_N'],3.)


if __name__=='__main__':unittest.main()
