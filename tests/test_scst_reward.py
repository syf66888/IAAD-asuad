import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.tasks.scst_reward import (
    BDDXScstReward, JavaPTBTokenizer, TOKENIZATION, build_training_cache,
    eos_inclusive_mask, load_training_cache, read_training_references,
)
from src.evalcap.coco_caption.pycocoevalcap.cider.cider_scorer import CiderScorer as EvalCider


class WhitespaceTokenizer:
    """Test seam only; production uses checked Java PTB without a fallback."""
    fingerprint = {"method": TOKENIZATION, "jar_sha256": "test-seam"}

    def __call__(self, sentences):
        return [sentence.lower() for sentence in sentences]


class ScstRewardTests(unittest.TestCase):
    def test_leave_one_out_excludes_self_and_stays_within_clip(self):
        # Distinct magnitudes expose accidental cross-clip mixing and /K.
        values = np.array([1., 2., 4., 8., 16., 100., 200., 400., 800., 1600.], dtype=np.float32)
        reward = self.reward()
        def scores(task, keys, hypotheses):
            self.assertEqual(len(keys), 10)  # no extra greedy rows
            return {'reward': values.copy(), 'cider': values.copy(), 'bleu4': values.copy()}
        with mock.patch.object(reward, '_task_scores', side_effect=scores):
            result = reward.score(['training_a', 'training_b'], ['a'] * 10, ['b'] * 10,
                                  baseline_type='leave_one_out')
            expected = np.array([np.mean(np.delete(row, i)) for row in values.reshape(2, 5)
                                 for i in range(5)])
            np.testing.assert_allclose(result['des']['baseline'], expected)
            np.testing.assert_allclose(result['des']['advantage'], values - expected)
            original_self_baseline = result['des']['baseline'][0]
            values[0] += 12
            changed = reward.score(['training_a', 'training_b'], ['a'] * 10, ['b'] * 10,
                                   baseline_type='leave_one_out')
            self.assertEqual(changed['des']['baseline'][0], original_self_baseline)
            np.testing.assert_allclose(changed['des']['baseline'][5:], expected[5:])
            self.assertEqual(changed['des']['baseline'][1], expected[1] + 3)

    def test_leave_one_out_real_scorer_and_invalid_inputs(self):
        result = self.reward().score(['training_a'],
            ['the car is turning right', 'unrelated words', '', 'the car is moving', 'turning right'],
            ['because the road curves right'] * 5, baseline_type='leave_one_out')
        self.assertTrue(np.isfinite(result['advantage']).all())
        self.assertAlmostEqual(result['des']['advantage'].sum(), 0., places=5)
        np.testing.assert_allclose(result['exp']['advantage'], 0., atol=1e-7)
        self.assertEqual(result['des']['greedy_cider'].size, 0)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            self.reward().score(['training_a'], ['a'], ['b'], baseline_type='leave_one_out')
        with self.assertRaisesRegex(ValueError, 'must not consume'):
            self.reward().score(['training_a'], ['a'] * 2, ['b'] * 2, ['a'], ['b'],
                                baseline_type='leave_one_out')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "BDDX"
        self.root.mkdir()
        self.yaml = self.root / "training_32frames.yaml"
        self.yaml.write_text("caption: training.caption.tsv\ncaption_linelist: training.caption.linelist.tsv\n", encoding="utf-8")
        self.rows = [
            ("training_a", [{"action": "the car is turning right", "justification": "because the road curves right"}]),
            ("training_b", [{"action": "the car is turning left", "justification": "because a person crosses the road"}]),
            ("training_c", [{"action": "the car is moving forward", "justification": "because the traffic is clear"},
                            {"action": "excluded row token", "justification": "excluded reference token"}]),
            ("training_unused", [{"action": "excluded clip token", "justification": "excluded clip explanation"}]),
        ]
        self.caption_path = self.root / "training.caption.tsv"
        self.caption_path.write_text("".join(key + "\t" + json.dumps(rows) + "\n" for key, rows in self.rows), encoding="utf-8")
        (self.root / "training.caption.linelist.tsv").write_text("0\t0\n1\t0\n2\t0\n", encoding="utf-8")
        (self.root / "testing.caption.tsv").write_text("THIS MUST NEVER BE READ", encoding="utf-8")
        self.cache_path = self.root / "scst.json.gz"
        self.tokenizer = WhitespaceTokenizer()
        self.cache = build_training_cache(self.yaml, self.cache_path, self.tokenizer)

    def tearDown(self):
        self.tmp.cleanup()

    def reward(self, **kwargs):
        return BDDXScstReward(self.cache_path, tokenizer=self.tokenizer, **kwargs)

    def test_train_only_linelist_sources_and_no_testing_dependency(self):
        self.assertEqual(self.cache["split"], "training")
        self.assertEqual(self.cache["tasks"]["des"]["ref_len"], 3)
        self.assertNotIn("training_unused", self.cache["tasks"]["des"]["references"])
        self.assertNotIn("excluded", json.dumps(self.cache["tasks"]))
        self.assertTrue(all("training" in Path(s["path"]).name for s in self.cache["sources"]))
        (self.root / "testing.caption.tsv").unlink()
        rebuilt = build_training_cache(self.yaml, self.cache_path, self.tokenizer)
        self.assertEqual(self.cache, rebuilt)
        with self.assertRaises(ValueError):
            read_training_references(self.root / "testing_32frames.yaml")
        self.yaml.write_text("caption: testing.caption.tsv\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_training_references(self.yaml)

    def test_mislabeled_testing_keys_and_stale_cache_rejected(self):
        text = self.caption_path.read_text(encoding="utf-8")
        self.caption_path.write_text(text.replace("training_a", "testing_a"), encoding="utf-8")
        with self.assertRaises(ValueError):
            read_training_references(self.yaml)
        self.caption_path.write_text(text.replace("turning right", "moving quickly"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Stale"):
            build_training_cache(self.yaml, self.cache_path, self.tokenizer)
        with self.assertRaises(KeyError):
            self.reward().score(["testing_a"], ["hello"], ["hello"], ["hello"], ["hello"])

    def test_fixed_df_matches_existing_evaluation_math(self):
        keys = [key for key, _ in self.rows[:3]]
        des = [self.rows[i][1][0]["action"] for i in range(3)]
        exp = [self.rows[i][1][0]["justification"] for i in range(3)]
        des[1] = "the car is slowing down"
        reward = self.reward().score(keys, des, exp, des, exp)
        evaluation = EvalCider()
        for key, hypothesis in zip(keys, des):
            evaluation += (hypothesis, self.cache["tasks"]["des"]["references"][key])
        _, expected = evaluation.compute_score("corpus")
        # Different NumPy/BLAS builds can retain ~1e-31 roundoff at an exact
        # zero cosine similarity; keep a tight absolute tolerance near zero.
        np.testing.assert_allclose(reward["des"]["sample_cider"], expected, rtol=1e-6, atol=1e-12)
        np.testing.assert_array_equal(reward["advantage"], np.zeros(3))

    def test_batch_order_size_and_sampling_multiplicity_do_not_change_rewards(self):
        reward = self.reward()
        single = reward.score(["training_a"], ["the car is turning right"], ["because the road curves right"],
                              ["the car is moving forward"], ["because traffic is clear"])
        multiple = reward.score(["training_a", "training_b"],
                                ["the car is turning right"] * 2 + ["the car is turning left"] * 2,
                                ["because the road curves right"] * 2 + ["because a person crosses the road"] * 2,
                                ["the car is moving forward"] * 2, ["because traffic is clear"] * 2)
        self.assertEqual(multiple["samples_per_clip"], 2)
        np.testing.assert_allclose(multiple["sample"][:2], np.repeat(single["sample"], 2))
        np.testing.assert_allclose(multiple["baseline"][:2], np.repeat(single["baseline"], 2))
        np.testing.assert_allclose(multiple["advantage"][:2], np.repeat(single["advantage"], 2))
        self.assertTrue(np.all(multiple["advantage"] > 0))
        repeat = reward.score(["training_a"], ["the car is turning right"], ["because the road curves right"],
                              ["the car is moving forward"], ["because traffic is clear"])
        np.testing.assert_array_equal(single["sample"], repeat["sample"])

    def test_metric_mixture_and_separate_task_balancing(self):
        args = (["training_a"], ["the car is turning right"], ["utterly unrelated random sentence"],
                ["utterly unrelated random sentence"], ["because the road curves right"])
        mixed = self.reward(cider_weight=0.75, bleu4_weight=0.25).score(*args)
        for task in ("des", "exp"):
            np.testing.assert_allclose(mixed[task]["sample"], 0.75 * mixed[task]["sample_cider"] / 10
                                       + 0.25 * mixed[task]["sample_bleu4"], rtol=1e-6)
        self.assertGreater(mixed["des"]["advantage"][0], 0)
        self.assertLess(mixed["exp"]["advantage"][0], 0)
        np.testing.assert_allclose(mixed["advantage"], (mixed["des"]["advantage"] + mixed["exp"]["advantage"]) / 2)
        with self.assertRaises(ValueError):
            self.reward(cider_weight=0, bleu4_weight=0)

    def test_empty_hypothesis_finite_and_eos_mask_gradient_sign(self):
        result = self.reward(bleu4_weight=0.1).score(["training_a"], [""], [""], [""], [""])
        self.assertTrue(np.isfinite(result["sample"]).all())
        np.testing.assert_array_equal(result["advantage"], [0])
        mask = eos_inclusive_mask([[8, 2, 9, 0], [8, 0, 9, 2], [2, 0, 0, 0], [8, 9, 7, 6]], 2, 0)
        np.testing.assert_array_equal(mask, [[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1]])
        # A sampled PAD before natural EOS is an action; a forced terminal EOS is not.
        np.testing.assert_array_equal(eos_inclusive_mask([[8, 0, 2]], 2, 0,
            forced_eos=[[False, False, True]]), [[True, True, False]])
        # Exact derivative d[-A * sum(mask * logp)]/dlogp; ascent is toward better captions.
        advantage = np.array([1., 0., -1., 0.5])
        derivative = -advantage[:, None] * mask
        self.assertTrue(np.all(derivative[0, :2] < 0))
        self.assertTrue(np.all(derivative[1] == 0))
        self.assertTrue(derivative[2, 0] > 0)
        self.assertTrue(np.all(derivative[~mask] == 0))

    def test_tampered_split_cache_rejected(self):
        self.cache["split"] = "testing"
        with gzip.open(self.cache_path, "wt", encoding="utf-8") as f:
            json.dump(self.cache, f)
        with self.assertRaises(ValueError):
            load_training_cache(self.cache_path)


@unittest.skipUnless(os.environ.get("SCST_TEST_JAVA") == "1", "Set SCST_TEST_JAVA=1 with evaluator PTB jar and Java")
class RealJavaPTBTests(unittest.TestCase):
    def test_matches_actual_evaluation_tokenizer(self):
        from src.evalcap.coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        sentences = ["The car isn't moving (yet).", "Because it's RED!", "", "The car\nis turning left."]
        expected = PTBTokenizer().tokenize({i: [{"caption": s}] for i, s in enumerate(sentences)})
        tokenizer = JavaPTBTokenizer()
        actual = tokenizer(sentences)
        self.assertEqual(actual, [expected[i][0] for i in range(len(sentences))])
        self.assertEqual(actual, tokenizer(sentences))


if __name__ == "__main__":
    unittest.main()
