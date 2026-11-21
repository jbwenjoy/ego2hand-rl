import unittest

from tools.summarize_wuji_tranche import fold


def segment(stream, round_id, passed, train, evaluation, failures=()):
    return dict(stream=stream, round=round_id, source_sequences=["A"], train_all_in_s=train,
                useful_warm_s=train/2, native_added_steps=32, failed_train_attempts=list(failures),
                checkpoint=f"{stream}_{round_id}", checks=[
                    dict(sequence="A", passed=passed, gates={"complete":passed}, eval_all_in_s=evaluation)
                ])


class LedgerTest(unittest.TestCase):
    def test_censoring_and_distinct_success(self):
        failure = dict(attempt="out/single_failed", elapsed_s=3, returncode=1)
        rows = [segment("single0",1,False,10,2,[failure]), segment("shared_mf",1,False,17,3),
                segment("shared_wm",1,True,6,2), segment("single0",2,True,5,3,[failure]),
                segment("shared_wm",2,True,9,3)]
        summary=fold({"references":[{"sequence":"A"}]},rows,verify=False)
        single=summary["methods"]["single"]
        self.assertEqual(single["references"]["A"]["first_passing_check"]["previous_checked_s"],15)
        self.assertEqual(single["first_qualified_card_s"],23)
        self.assertEqual(single["failed_s"],3)
        self.assertEqual(single["passed_at_common_budget"],0)
        self.assertEqual(summary["common_scored_card_seconds"],20)
        self.assertIsNone(summary["methods"]["shared_mf"]["gpu_h_per_qualified_clip"])
        self.assertEqual(summary["methods"]["shared_wm"]["gpu_h_per_qualified_clip"],20/3600)

    def test_missing_attempt_time_and_duplicate_segment(self):
        row=segment("single0",1,True,10,2,[dict(attempt="interrupted")])
        summary=fold({"references":[{"sequence":"A"}]},[row],verify=False)
        self.assertIsNone(summary["methods"]["single"]["gpu_h_per_qualified_clip"])
        with self.assertRaises(ValueError):
            fold({"references":[{"sequence":"A"}]},[row,row],verify=False)
