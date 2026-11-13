from pathlib import Path
import tempfile
import unittest
from supermimic_research.runtime.controlled_evaluation import evaluate_case


class ExplicitControllerContractTests(unittest.TestCase):
    def test_ambiguous_controller_rejected_before_creating_an_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'episode'
            with self.assertRaisesRegex(ValueError, 'one explicit controller'):
                evaluate_case('hand2-pen-spin', 'INTERVENTION', None, {}, Path(tmp), None, folder,
                              constant_action=[0.] * 20, action_provider=lambda *args: None)
            self.assertFalse(folder.exists())
