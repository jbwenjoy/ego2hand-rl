import unittest
import torch
from supermimic_research.runtime.first_overflow_observer import FirstOverflowObserver


def views(device='cpu'):
    result = {k: torch.arange(6, device=device, dtype=torch.float32).reshape(2, 3)
              for k in ['qpos', 'qvel', 'ctrl', 'qacc_warmstart']}
    result.update(overflow=torch.zeros(2, dtype=torch.int32, device=device),
                  phase=torch.tensor([67, 68], device=device), time=torch.tensor([.2, .3], device=device),
                  geom_xpos=torch.zeros(2, 4, 3, device=device), geom_xmat=torch.zeros(2, 4, 3, 3, device=device))
    return result


class ObserverTests(unittest.TestCase):
    def test_first_raw_mask_pre_post_survives_reset_and_later_event(self):
        v = views(); o = FirstOverflowObserver(v)
        before = {k: t.clone() for k, t in v.items()}
        o.before(); v['overflow'][1] = 256; v['qpos'].add_(10)
        o.after(iteration=252, kind=2, step_call=508, forward_call=31)
        captured = v['qpos'][1].clone()
        # Clear the live flag and reset every field as the real collector does.
        for t in v.values(): t.zero_()
        o.before(); v['overflow'][1] = 128; v['overflow'][0] = 3
        o.after(iteration=253, kind=1, step_call=509, forward_call=31)
        p = o.packet()
        self.assertEqual(p['raw_mask'].tolist(), [3, 256])
        self.assertEqual(p['event'][1].tolist(), [252, 2, 508, 31])
        self.assertTrue(torch.equal(p['pre']['qpos'][1], before['qpos'][1]))
        self.assertTrue(torch.equal(p['post']['qpos'][1], captured))
        self.assertEqual(p['post']['overflow'].tolist(), [3, 256])

    def test_no_event_no_live_writes_and_no_host_item_in_observer(self):
        v = views(); saved = {k: t.clone() for k, t in v.items()}; o = FirstOverflowObserver(v)
        # before/after may not read CUDA scalars through Tensor.item().
        from unittest.mock import patch
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar sync')):
            o.before(); o.after(iteration=202, kind=1, step_call=1, forward_call=0)
        for k in v: self.assertTrue(torch.equal(v[k], saved[k]))
        self.assertFalse(bool(o.packet()['raw_mask'].any()))

    def test_snapshots_do_not_alias_live_or_exported_values(self):
        v = views(); o = FirstOverflowObserver(v); o.before(); v['overflow'][0] = 256
        o.after(iteration=202, kind=1, step_call=1, forward_call=0)
        p = o.packet(); p['pre']['qpos'].zero_(); v['qpos'].zero_()
        self.assertEqual(float(o.packet()['pre']['qpos'][0, 1]), 1.)

    def test_bool_flags_and_unmatched_operations_rejected(self):
        v = views(); v['overflow'] = v['overflow'].bool()
        with self.assertRaises(ValueError): FirstOverflowObserver(v)
        o = FirstOverflowObserver(views())
        with self.assertRaises(ValueError): o.after(iteration=1, kind=1, step_call=1, forward_call=0)
        o.before()
        with self.assertRaises(ValueError): o.before()
        with self.assertRaises(ValueError): o.packet()


if __name__ == '__main__': unittest.main()
