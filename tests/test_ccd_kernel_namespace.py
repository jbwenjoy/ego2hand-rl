import unittest
from supermimic_research.runtime.ccd_kernel_namespace import namespace_builders


class KernelNamespaceTests(unittest.TestCase):
    def test_factories_and_calls_share_unique_identity(self):
        s='@cache_kernel\ndef ccd_hfield_kernel_builder(x):\n  return x\n@cache_kernel\ndef ccd_kernel_builder(x):\n  return x\ny=ccd_kernel_builder(1)\n'
        a, mapping=namespace_builders(s,'variant_a');b, other=namespace_builders(s,'variant_b')
        self.assertNotEqual(mapping['ccd_kernel_builder'],other['ccd_kernel_builder'])
        for old,new in mapping.items():a=a.replace(new,old)
        self.assertEqual(a,s)

    def test_missing_factory_rejected(self):
        with self.assertRaises(ValueError):namespace_builders('@cache_kernel\ndef x(): pass','variant_a')

    def test_invalid_namespace_rejected(self):
        with self.assertRaises(ValueError):namespace_builders('', 'bad namespace')
