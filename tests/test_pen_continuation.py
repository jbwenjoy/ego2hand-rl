import copy
import unittest

import torch

from supermimic_research.learning.pen_continuation import validate_parent
from supermimic_research.learning.pen_response import INPUT_SCHEMA, OUTPUT_SCHEMA, HEAD_SCHEMA


def fixture():
    cfg = dict(first_checkpoint_update=100, planned_updates=2000)
    prep = dict(outputs={'TRAIN': 'train-pin', 'CHECK': 'check-pin'}, physics={'mass': .03})
    arms = {}
    for mode in ['DIRECT', 'IMPULSE']:
        arms[mode] = dict(schema='controlled_pen_response_arm:1', input_schema=INPUT_SCHEMA,
            output_schema=OUTPUT_SCHEMA, head_schema=HEAD_SCHEMA if mode == 'IMPULSE' else OUTPUT_SCHEMA,
            mode=mode, coupled=mode == 'IMPULSE', update=100, physics=prep['physics'].copy(),
            config=cfg.copy(), sampler=torch.tensor([1, 2, 3], dtype=torch.uint8))
    saved = dict(schema='controlled_pen_response_pair_checkpoint:1', update=100, planned_updates=2000,
        arms=arms, dataset_pins=prep['outputs'].copy(), logs={m: [[0.]] * 100 for m in arms})
    health = dict(schema='controlled_pen_first100_health:1', update=100, training_running=False,
        checks={'restore': True}, passed=1, total=1)
    return saved, health, prep


class PenContinuationTests(unittest.TestCase):
    def test_only_complete_matching_parent_is_accepted(self):
        saved, health, prep = fixture()
        self.assertEqual(validate_parent(saved, health, prep)['planned_updates'], 2000)
        mutations = [
            lambda s, h, p: s.update(update=99),
            lambda s, h, p: s['arms']['DIRECT'].update(update=99),
            lambda s, h, p: s['arms']['IMPULSE'].update(output_schema='old87'),
            lambda s, h, p: s['arms']['IMPULSE'].update(coupled=False),
            lambda s, h, p: s['arms']['DIRECT']['physics'].update(mass=.04),
            lambda s, h, p: s['arms']['DIRECT']['config'].update(planned_updates=3000),
            lambda s, h, p: s['arms']['IMPULSE'].update(sampler=torch.tensor([1, 2, 4], dtype=torch.uint8)),
            lambda s, h, p: s['dataset_pins'].update(CHECK='different-pin'),
            lambda s, h, p: s['logs']['DIRECT'].pop(),
            lambda s, h, p: h.update(training_running=True),
            lambda s, h, p: h['checks'].update(restore=False),
            lambda s, h, p: h.update(total=2),
        ]
        for change in mutations:
            with self.subTest(change=change):
                s, h, p = copy.deepcopy(fixture()); change(s, h, p)
                with self.assertRaises(ValueError): validate_parent(s, h, p)
