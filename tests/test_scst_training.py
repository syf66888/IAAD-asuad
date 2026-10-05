"""CPU tests of trainer math and state transitions, without importing YOLO/Swin."""
import ast
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from easydict import EasyDict

from src.tasks.checkpoint_selection import BASELINE, selection_record
from src.tasks.dataset_protocol import validate_splits


PROJECT = Path(__file__).resolve().parents[1]


def functions_from(path, names, namespace):
    """Execute the real functions without eager heavyweight model imports."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError('Missing trainer functions: ' + str(names))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


NS = dict(torch=torch, np=np, math=math, hashlib=hashlib, os=os, Path=Path,
          selection_record=selection_record, validate_splits=validate_splits)
functions_from(PROJECT / 'src/tasks/train.py', ['accumulation_window'], NS)
functions_from(PROJECT / 'src/tasks/train_scst.py', [
    'policy_loss', 'file_sha256', 'special_token_ids', 'accumulation_scale',
    'resume_signature', 'validate_resume_state', 'termination_reason',
    'validate_completion', 'read_base_record', 'configure'], NS)


class AttrDict(dict):
    __getattr__ = dict.__getitem__


NS['EasyDict'] = EasyDict


class ScstTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / 'model.bin'
        self.model.write_bytes(b'verified supervised model')
        self.cache = self.root / 'rewards.json.gz'
        self.cache.write_bytes(b'verified training only reward cache')
        self.base = selection_record(dict(
            metrics=copy.deepcopy(BASELINE), validation_yaml='BDDX/testing_32frames.yaml',
            epoch=49, iteration=294, global_step=98, model_path=str(self.model),
            checkpoint=str(self.root), model_sha256=NS['file_sha256'](self.model)))
        self.args = AttrDict(scst_checkpoint=str(self.model), scst_base_record=self.base,
                            scst_reward_cache=str(self.cache), learning_rate=1e-6, seed=88,
                            num_train_epochs=4, scst_cider_weight=1., scst_bleu4_weight=0.,
                            per_gpu_train_batch_size=2, gradient_accumulation_steps=3)

    def record(self, epoch, factor=1.0):
        metrics = {task: {key: value * factor for key, value in values.items()}
                   for task, values in BASELINE.items()}
        return selection_record(dict(metrics=metrics, epoch=epoch, iteration=epoch * 3,
                                     global_step=epoch, validation_yaml='BDDX/testing_32frames.yaml'))

    def state(self, epochs=2):
        return dict(iteration=epochs * 3, epoch_boundary=True, world_size=1,
                    rank_rng_states=[{'rank': 0}], eval_log=[self.record(i + 1) for i in range(epochs)],
                    best_record=self.base, resume_signature=NS['resume_signature'](self.args, self.base))

    def test_policy_gradient_causal_credit_and_zero_advantage(self):
        logp = torch.full((3, 2, 4), -1., requires_grad=True)
        mask = torch.tensor([[[0, 1, 1, 0], [0, 1, 0, 0]]] * 3, dtype=torch.bool)
        des = torch.tensor([2., 0., -2.], requires_grad=True)
        exp = torch.tensor([4., 0., -4.], requires_grad=True)
        loss = NS['policy_loss'](logp, mask, des, exp)
        loss.backward()
        # Description receives 0.5 * (des + exp); explanation receives 0.5 * exp.
        torch.testing.assert_close(logp.grad[0, 0], torch.tensor([0., -1., -1., 0.]))
        torch.testing.assert_close(logp.grad[0, 1], torch.tensor([0., -2./3., 0., 0.]))
        torch.testing.assert_close(logp.grad[1], torch.zeros(2, 4))
        torch.testing.assert_close(logp.grad[2], -logp.grad[0])
        self.assertIsNone(des.grad)
        self.assertIsNone(exp.grad)

    def test_padding_negative_infinity_does_not_poison_eos_gradient(self):
        logp = torch.tensor([[[float('-inf'), -2., -3., float('-inf')],
                             [float('-inf'), -4., float('-inf'), float('-inf')]]], requires_grad=True)
        mask = torch.tensor([[[0, 1, 1, 0], [0, 1, 0, 0]]], dtype=torch.bool)
        loss = NS['policy_loss'](logp, mask, [1.], [1.])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(logp.grad[0, 0, 2].item(), -1.)  # natural EOS is still an action
        self.assertTrue(torch.equal(logp.grad[~mask], torch.zeros(5)))

    def test_tail_microbatch_gradient_equals_full_window_example_mean(self):
        parameter = torch.tensor(2., requires_grad=True)
        observations = torch.tensor([1., 2., 3., 4., 10.])
        weights = []
        for iteration, values in enumerate(observations.split(2), 1):
            weight, flush = NS['accumulation_scale'](iteration, 3, 3, 3, 2, 5, len(values))
            weights.append(weight)
            ((parameter * values).mean() * weight).backward()
            self.assertEqual(flush, iteration == 3)
        torch.testing.assert_close(parameter.grad, observations.mean())
        np.testing.assert_allclose(weights, [0.4, 0.4, 0.2])
        # At the next epoch boundary the last microbatch must not leak into a new window.
        weight, flush = NS['accumulation_scale'](6, 6, 3, 3, 2, 5, 1)
        self.assertAlmostEqual(weight, 0.2)
        self.assertTrue(flush)

    def test_short_accumulation_window_and_truncated_smoke(self):
        # Five nominal batches, last containing one clip: full window, then 2+1.
        weights, flushes = [], []
        for iteration, count in enumerate([2, 2, 2, 2, 1], 1):
            weight, flush = NS['accumulation_scale'](iteration, 5, 5, 3, 2, 9, count)
            weights.append(weight)
            flushes.append(flush)
        np.testing.assert_allclose(weights, [1/3, 1/3, 1/3, 2/3, 1/3])
        self.assertEqual(flushes, [False, False, True, False, True])
        self.assertEqual(NS['accumulation_scale'](2, 2, 5, 4, 2, 9, 2), (0.5, True))
        with self.assertRaises(ValueError):
            NS['accumulation_scale'](5, 5, 5, 3, 2, 9, 2)

    def test_legacy_tokenizer_needs_no_id_properties(self):
        class LegacyTokenizer:
            cls_token, sep_token, pad_token, mask_token = '[CLS]', '[SEP]', '[PAD]', '[MASK]'

            def convert_tokens_to_ids(self, value):
                return {'[CLS]': 101, '[SEP]': 102, '[PAD]': 0, '[MASK]': 103}[value]

        ids = NS['special_token_ids'](LegacyTokenizer())
        self.assertEqual(ids, dict(bos_token_id=101, pad_token_id=0, mask_token_id=103, eos_token_ids=[102]))

    def test_swapped_caption_order_is_rejected_before_any_training(self):
        config = dict(train_yaml='BDDX/training_32frames.yaml', val_yaml='BDDX/testing_32frames.yaml',
                      use_sep_cap=True, max_seq_length=70, max_gen_length=35,
                      add_od_labels=False, use_swap_cap=True)
        with self.assertRaisesRegex(ValueError, 'swapped'):
            NS['configure'](config)

    def valid_config(self):
        return dict(train_yaml='BDDX/training_32frames.yaml', val_yaml='BDDX/testing_32frames.yaml',
                    use_sep_cap=True, max_seq_length=70, max_gen_length=35, add_od_labels=False,
                    scst_checkpoint=str(self.model), num_train_epochs=4, gradient_accumulation_steps=2,
                    per_gpu_train_batch_size=1, learning_rate=1e-6)

    def assert_configure_defaults(self, configure, namespace):
        # Use the actual installed EasyDict, whose inherited dict.setdefault
        # does not synchronize the attribute namespace in the deployed version.
        with mock.patch.object(torch.cuda, 'is_available', return_value=False), \
                mock.patch.dict(namespace, {'dist_init': lambda args: None}):
            args = configure(self.valid_config())
        for name, expected in dict(scst_sample_n=2, scst_min_epochs=2, early_stopping_patience=2,
                                   scst_resume='', smoke=False, max_train_steps=0,
                                   scst_weight_decay=0.05, scst_warmup_ratio=0.05).items():
            self.assertEqual(args[name], expected)
            self.assertEqual(getattr(args, name), expected)
        self.assertTrue(args.scst)
        self.assertTrue(args['scst'])
        self.assertEqual(args.device.type, 'cpu')

    def test_real_easydict_defaults_and_update_expose_attributes(self):
        self.assert_configure_defaults(NS['configure'], NS)

    def test_k5_baseline_ce_and_beam_configuration(self):
        config = self.valid_config()
        config.update(scst_sample_n=5, scst_baseline_type='leave_one_out',
                      sc_baseline_type='greedy', scst_ce_weight=.05, num_beams=3)
        with mock.patch.object(torch.cuda, 'is_available', return_value=False), \
                mock.patch.dict(NS, {'dist_init': lambda args: None}):
            args = NS['configure'](config)
            self.assertEqual(args.scst_sample_n, 5)
            self.assertEqual(args.scst_baseline_type, 'leave_one_out')
            self.assertEqual(args.sc_baseline_type, 'leave_one_out')
            self.assertEqual(args.scst_ce_weight, .05)
            self.assertEqual(args.num_beams, 3)
            for changes in ({'scst_sample_n': 1}, {'scst_sample_n': 17},
                            {'scst_sample_n': 2.5}, {'scst_sample_n': True},
                            {'scst_baseline_type': 'sample_mean'}, {'scst_ce_weight': float('nan')},
                            {'scst_ce_weight': -.1}, {'num_beams': 0}):
                with self.assertRaises(ValueError):
                    NS['configure'](dict(config, **changes))
            legacy = dict(self.valid_config(), sc_baseline_type='leave_one_out')
            self.assertEqual(NS['configure'](legacy).scst_baseline_type, 'leave_one_out')

    def test_resume_identity_includes_baseline_ce_and_decoding(self):
        original = NS['resume_signature'](self.args, self.base)
        for key, value in [('scst_baseline_type', 'leave_one_out'), ('scst_ce_weight', .05),
                           ('num_beams', 3)]:
            changed = AttrDict(self.args, **{key: value})
            self.assertNotEqual(original, NS['resume_signature'](changed, self.base))

    @unittest.skipUnless(os.environ.get('SCST_TEST_FULL_IMPORT') == '1',
                         'Set SCST_TEST_FULL_IMPORT=1 in the full model runtime')
    def test_complete_trainer_import_and_real_easydict_configure(self):
        spec = importlib.util.spec_from_file_location('scst_actual_import_test', PROJECT / 'src/tasks/train_scst.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assert_configure_defaults(module.configure, module.__dict__)

    def test_baseline_is_bound_to_exact_model_and_metrics(self):
        baseline = NS['read_base_record'](self.args)
        self.assertAlmostEqual(baseline['selection_score'], 1.)
        self.model.write_bytes(b'other epoch with same filename')
        with self.assertRaisesRegex(ValueError, 'bound'):
            NS['read_base_record'](self.args)

    def test_resume_rejects_changed_reward_or_training_config(self):
        state = self.state()
        signature = NS['resume_signature'](self.args, self.base)
        NS['validate_resume_state'](state, signature, 3, 12, 1)
        self.args['scst_bleu4_weight'] = 0.25
        with self.assertRaisesRegex(ValueError, 'configuration'):
            NS['validate_resume_state'](state, NS['resume_signature'](self.args, self.base), 3, 12, 1)
        self.args['scst_bleu4_weight'] = 0.
        self.cache.write_bytes(b'changed training captions')
        with self.assertRaisesRegex(ValueError, 'configuration'):
            NS['validate_resume_state'](state, NS['resume_signature'](self.args, self.base), 3, 12, 1)

    def test_resume_rejects_incomplete_history_bad_world_and_stale_best(self):
        signature = NS['resume_signature'](self.args, self.base)
        bad = self.state()
        bad['iteration'] = 5
        with self.assertRaises(ValueError):
            NS['validate_resume_state'](bad, signature, 3, 12, 1)
        bad = self.state()
        bad['eval_log'] = bad['eval_log'][:1]
        with self.assertRaisesRegex(ValueError, 'history'):
            NS['validate_resume_state'](bad, signature, 3, 12, 1)
        with self.assertRaisesRegex(ValueError, 'world'):
            NS['validate_resume_state'](self.state(), signature, 3, 12, 2)
        self.model.write_bytes(b'overwritten winner')
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            NS['validate_resume_state'](self.state(), signature, 3, 12, 1)

    def test_terminal_resume_does_not_need_another_epoch(self):
        self.assertIsNone(NS['termination_reason'](1, 4, 1, 2, 2))
        self.assertEqual(NS['termination_reason'](2, 4, 2, 2, 2), 'early_stopping')
        self.assertEqual(NS['termination_reason'](4, 4, 0, 2, 2), 'epochs_complete')
        NS['validate_completion'](6, 12, 3, [self.record(1), self.record(2)], True)
        NS['validate_completion'](12, 12, 3, [self.record(i) for i in range(1, 5)], False)
        with self.assertRaisesRegex(RuntimeError, 'before'):
            NS['validate_completion'](6, 12, 3, [self.record(1), self.record(2)], False)
        with self.assertRaises(RuntimeError):
            NS['validate_completion'](12, 12, 3, [], False)


if __name__ == '__main__':
    unittest.main()
