import copy
import math
import unittest
from src.tasks.checkpoint_selection import BASELINE, joint_score, selection_record


class JointSelectionTests(unittest.TestCase):
    def test_baseline_and_equal_relative_improvements(self):
        self.assertAlmostEqual(joint_score(BASELINE), 1.0)
        scores = []
        for kind in BASELINE:
            for name in BASELINE[kind]:
                metrics = copy.deepcopy(BASELINE)
                metrics[kind][name] *= 1.2
                scores.append(joint_score(metrics))
        self.assertTrue(all(abs(score - 1.2 ** 0.25) < 1e-10 for score in scores))

    def test_collapsed_task_is_not_hidden_by_large_cider(self):
        metrics = copy.deepcopy(BASELINE)
        metrics['des']['CIDEr'] *= 2
        metrics['exp']['Bleu_4'] *= 0.1
        self.assertLess(joint_score(metrics), 1.0)

    def test_missing_invalid_and_wrong_split_fail(self):
        for value in (-1, float('nan'), float('inf')):
            metrics = copy.deepcopy(BASELINE)
            metrics['des']['Bleu_4'] = value
            with self.assertRaises(ValueError):
                joint_score(metrics)
        with self.assertRaises(ValueError):
            selection_record(dict(metrics=BASELINE, validation_yaml='BDDX/validation_32frames.yaml'))
        self.assertAlmostEqual(selection_record(dict(metrics=BASELINE,
            validation_yaml='BDDX/testing_32frames.yaml'))['selection_score'], 1.0)


if __name__ == '__main__':
    unittest.main()
