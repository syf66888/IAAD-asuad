"""Policy probability, conditioning and gradient tests using a real tiny BERT."""

import unittest
import types
from unittest import mock

import torch

from src.layers.bert import BertConfig
from src.layers.bert.modeling_bert import BertForImageCaptioning
from src.modeling.scst_decode import (
    _parallel_query_logits, _rollout, caption_token_mask,
    generation_attention_mask, sampled_token_log_probs, scst_decode,
)


SPECIAL = dict(bos_token_id=1, pad_token_id=0, eos_token_ids=[2], mask_token_id=11)


def tiny_model():
    torch.manual_seed(88)
    config = BertConfig(vocab_size_or_config_json_file=12, hidden_size=16,
                        num_hidden_layers=2, num_attention_heads=4, intermediate_size=32)
    config.img_feature_dim = 3
    config.img_feature_type = 'frcnn'
    config.use_img_layernorm = False
    config.hidden_dropout_prob = 0.2
    config.attention_probs_dropout_prob = 0.2
    return BertForImageCaptioning(config)


def inputs(batch=2, slot=5):
    features = torch.randn(batch, 3, 3, requires_grad=True)
    # Text region deliberately invalid: the helper must regenerate it, while
    # preserving the differentiable learned visual block.
    visual = torch.tensor([[[1., .7, .2], [.3, 1., .4], [.8, .6, 1.]]], requires_grad=True)
    visual = visual.expand(batch, -1, -1)
    mask = generation_attention_mask(visual, 3, slot)
    return features, mask


def sequential_logits(model, features, full_mask, tokens, positions=None):
    """Independent full-prefix implementation, without the parallel trick."""
    batch, _, slot = tokens.shape
    flat = tokens.reshape(batch, 2 * slot)
    results = []
    for segment in range(2):
        per_token = []
        for position in (range(1, slot) if positions is None else positions):
            absolute = segment * slot + position
            prefix = flat[:, :absolute]
            ids = torch.cat((prefix, prefix.new_full((batch, 1), SPECIAL['mask_token_id'])), dim=1)
            mask = full_mask.clone()
            if segment == 1:
                mask[:, :, :slot] *= tokens[:, 0].ne(0)[:, None].to(mask.dtype)
            keep = torch.cat((torch.arange(absolute + 1), torch.arange(2 * slot, mask.shape[-1])))
            mask = mask.index_select(1, keep).index_select(2, keep)
            types = torch.zeros_like(ids)
            types[:, slot:] = 1
            logits = model.encode_forward(ids, features, mask, token_type_ids=types,
                                           is_training=False)[0][:, -1]
            per_token.append(logits)
        results.append(torch.stack(per_token, dim=1))
    return torch.stack(results, dim=1)


class ScstDecodeTests(unittest.TestCase):
    def teacher_inputs(self, mask):
        ids = torch.tensor([[1, 11, 5, 2, 0, 1, 6, 11, 2, 0]])
        positions = torch.zeros_like(ids)
        positions[:, [1, 7]] = 1
        return dict(input_ids=ids, attention_mask=mask,
                    token_type_ids=torch.tensor([[0] * 5 + [1] * 5]),
                    masked_pos=positions, masked_ids=torch.tensor([[4, 7, -1]]))

    def test_k5_loo_skips_greedy_preserves_sampling_and_gradients(self):
        model = tiny_model().train()
        features, mask = inputs(batch=2)
        with mock.patch('src.modeling.scst_decode._rollout', wraps=_rollout) as rollout:
            output = scst_decode(model, features, mask, SPECIAL, sample_n=5,
                                 max_length=5, baseline_type='leave_one_out')
        self.assertEqual(rollout.call_count, 1)
        self.assertTrue(rollout.call_args.args[5])  # multinomial sample, not beam
        self.assertIsNone(output['greedy_ids'])
        self.assertEqual(output['sampled_ids'].shape, (10, 2, 5))
        self.assertGreater(output['sampled_ids'].reshape(10, -1).unique(dim=0).shape[0], 1)
        (-output['sampled_log_probs'].sum()).backward()
        self.assertGreater(features.grad.abs().sum().item(), 0.)
        self.assertTrue(model.training)

    def test_masked_ce_matches_original_decoder_and_reaches_visual_gradient(self):
        model = tiny_model().eval()
        features, mask = inputs(batch=1)
        teacher = self.teacher_inputs(mask)
        expected = model(img_feats=features, **teacher)[0]
        output = scst_decode(model, features, mask, SPECIAL, sample_n=5, max_length=5,
                             baseline_type='leave_one_out', teacher_forcing=teacher)
        torch.testing.assert_close(output['ce_loss'], expected)
        (.05 * output['ce_loss']).backward()
        self.assertGreater(features.grad.abs().sum().item(), 0.)
        self.assertGreater(model.cls.predictions.bias.grad.abs().sum().item(), 0.)
        self.assertNotIn('img_feats', teacher)  # do not mutate caller data
        self.assertFalse(model.training)
        bad = dict(teacher, masked_ids=torch.tensor([[4, -1, -1]]))
        with self.assertRaisesRegex(ValueError, 'match per clip'):
            scst_decode(model, features, mask, SPECIAL, sample_n=2, max_length=5,
                        baseline_type='leave_one_out', teacher_forcing=bad)

    def test_full_wrapper_one_visual_encoding_with_optional_ce(self):
        from src.modeling.asuad_model import AsuadModel
        class Swin(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Module()
                self.backbone.norm = torch.nn.LayerNorm(4)
                self.grid = torch.nn.Parameter(torch.randn(1, 4, 1, 1, 3))
                self.calls = 0
            def forward(self, images):
                self.calls += 1
                return self.grid.expand(images.shape[0], -1, -1, -1, -1)
        args = types.SimpleNamespace(use_checkpoint=False, freeze_backbone=False,
            img_feature_dim=3, grid_feat=True, mask_prob=.5, max_img_seq_length=3,
            max_num_frames=2, feature_fusion='none', learn_mask_enabled=True,
            learn_mask_log_bias=True, sparse_mask_soft2hard=False)
        model = AsuadModel(args, None, Swin(), tiny_model()).eval()
        _, mask = inputs(batch=1)
        teacher = self.teacher_inputs(mask)
        images = torch.randn(1, 2, 3, 4, 4)
        original = model(img_feats=images, **teacher)[0]
        model.swin.calls = 0
        output = model(img_feats=images, scst=True, scst_options=dict(special_ids=SPECIAL,
            sample_n=5, max_length=5, baseline_type='leave_one_out', ce_weight=.05), **teacher)
        self.assertEqual(model.swin.calls, 1)
        torch.testing.assert_close(output['ce_loss'], original)
        (.05 * output['ce_loss'] - output['sampled_log_probs'].mean()).backward()
        for parameter in (model.swin.grid, model.fc.weight, model.learn_vid_att.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)
        # Zero CE uses no extra teacher decoder invocation and accepts the old API.
        output = model(img_feats=images, scst=True, scst_options=dict(special_ids=SPECIAL,
            sample_n=2, max_length=5), **teacher)
        self.assertEqual(output['ce_loss'].item(), 0.)

    def test_parallel_queries_equal_each_autoregressive_prefix(self):
        model = tiny_model().eval()
        features, mask = inputs()
        # PAD inside a generated action stresses the distinct des/exp masks.
        tokens = torch.tensor([[[1, 4, 0, 5, 2], [1, 6, 7, 2, 0]],
                               [[1, 3, 2, 0, 0], [1, 9, 4, 8, 2]]])
        expected = sequential_logits(model, features, mask, tokens)
        actual = _parallel_query_logits(model, features, mask, tokens, SPECIAL)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=1e-5),
                        (actual - expected).abs().max().item())

    def test_exp_depends_on_its_sampled_des_and_des_has_no_future_leak(self):
        model = tiny_model().eval()
        features, mask = inputs(batch=1)
        tokens = torch.tensor([[[1, 4, 5, 2, 0], [1, 6, 7, 2, 0]]])
        actual = _parallel_query_logits(model, features, mask, tokens, SPECIAL)
        changed = tokens.clone()
        changed[:, 0, 1] = 8
        changed_logits = _parallel_query_logits(model, features, mask, changed, SPECIAL)
        self.assertTrue(torch.equal(actual[:, 0, 0], changed_logits[:, 0, 0]))
        self.assertGreater((actual[:, 1] - changed_logits[:, 1]).abs().max().item(), 1e-8)
        changed = tokens.clone()
        changed[:, 1, 1:] = 9
        changed_logits = _parallel_query_logits(model, features, mask, changed, SPECIAL)
        self.assertTrue(torch.equal(actual[:, 0], changed_logits[:, 0]))

    def test_greedy_matches_existing_generation_with_cached_bert(self):
        model = tiny_model().eval()
        features, mask = inputs()
        for eos_bias in (-10., 10.):
            with torch.no_grad():
                model.cls.predictions.bias[2] = eos_bias
                actual, _ = _rollout(model, features, mask, SPECIAL, 5, False, 0, 1.)
                expected = model.generate(
                    img_feats=features, attention_mask=mask, masked_pos=torch.ones(2, 10).long(),
                    token_type_ids=torch.cat((torch.zeros(2, 5), torch.ones(2, 5)), dim=1).long(),
                    input_ids=torch.zeros(2, 10).long(), max_length=10,
                    do_sample=False, num_beams=1, temperature=1., top_k=0, top_p=1.,
                    repetition_penalty=1., length_penalty=1., num_return_sequences=1,
                    num_keep_best=1, is_decode=True, add_od_labels=False,
                    od_labels_start_posid=10, use_sep_cap=True, **SPECIAL)[0]
                self.assertTrue(torch.equal(actual, expected[:, 0].reshape(2, 2, 5)))

    def test_scst_backprop_reaches_decoder_and_visual_features(self):
        model = tiny_model().train()
        features, mask = inputs()
        model.cls_img_feat.eval()  # A deliberately frozen mode must survive.
        output = scst_decode(model, features, mask, SPECIAL, sample_n=2, max_length=5)
        self.assertEqual(output['sampled_ids'].shape, (4, 2, 5))
        self.assertEqual(output['greedy_ids'].shape, (2, 2, 5))
        self.assertFalse(output['greedy_ids'].requires_grad)
        self.assertTrue(model.training)
        self.assertFalse(model.cls_img_feat.training)
        scores, valid = output['sampled_log_probs'], output['sampled_mask']
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.equal(scores[~valid], torch.zeros_like(scores[~valid])))
        loss = -scores.sum() / valid.sum().clamp_min(1)
        loss.backward()
        self.assertGreater(model.cls.predictions.bias.grad.abs().sum().item(), 0.)
        self.assertGreater(features.grad.abs().sum().item(), 0.)
        self.assertFalse(valid[:, :, 0].any())

    def test_natural_eos_has_policy_gradient_and_forced_eos_does_not(self):
        model = tiny_model().eval()
        features, mask = inputs(batch=1)
        with torch.no_grad():
            model.cls.predictions.bias[2] = 100.
        natural = scst_decode(model, features, mask, SPECIAL, sample_n=1, max_length=5)
        self.assertTrue(natural['sampled_ids'][:, :, 1].eq(2).all())
        self.assertTrue(natural['sampled_mask'][:, :, 1].all())
        self.assertFalse(natural['sampled_mask'][:, :, 2:].any())
        self.assertFalse(natural['sampled_forced_eos'].any())
        with torch.no_grad():
            model.cls.predictions.bias[2] = -100.
        forced = scst_decode(model, features, mask, SPECIAL, sample_n=1, max_length=5)
        self.assertTrue(forced['sampled_forced_eos'][:, :, -1].all())
        self.assertFalse(forced['sampled_mask'][:, :, -1].any())
        self.assertTrue(torch.isfinite(forced['sampled_log_probs']).all())
        # An empty but naturally terminated caption still trains termination.
        with torch.no_grad():
            model.cls.predictions.bias[2] = 0.
        immediate_eos = torch.tensor([[[1, 2, 0, 0, 0], [1, 2, 0, 0, 0]]])
        scores, valid = sampled_token_log_probs(model, features, mask, immediate_eos, SPECIAL)
        (-scores.sum()).backward()
        self.assertTrue(valid[:, :, 1].all())
        self.assertGreater(model.cls.predictions.bias.grad[2].abs().item(), 0.)

    def test_natural_pad_is_an_action_but_post_eos_padding_is_not(self):
        model = tiny_model().eval()
        features, mask = inputs(batch=1)
        tokens = torch.tensor([[[1, 4, 0, 2, 0], [1, 0, 2, 0, 0]]])
        scores, valid = sampled_token_log_probs(model, features, mask, tokens, SPECIAL)
        self.assertEqual(valid.tolist(), [[[False, True, True, True, False],
                                          [False, True, True, False, False]]])
        (-scores[0, 0, 2] - scores[0, 1, 1]).backward()
        self.assertGreater(model.cls.predictions.bias.grad[0].abs().item(), 0.)

    def test_maximum_35_and_112_token_slots_have_independent_boundaries(self):
        for slot in (35, 112):
            with self.subTest(slot=slot):
                model = tiny_model().eval()
                features, mask = inputs(batch=1, slot=slot)
                with torch.no_grad():
                    model.cls.predictions.bias[2] = -100.
                result = scst_decode(model, features, mask, SPECIAL, sample_n=1, max_length=slot)
                self.assertEqual(result['sampled_ids'].shape, (1, 2, slot))
                self.assertEqual(result['greedy_ids'].shape, (1, 2, slot))
                self.assertTrue(result['sampled_ids'][:, :, 0].eq(1).all())
                self.assertTrue(result['sampled_ids'][:, :, -1].eq(2).all())
                self.assertTrue(result['sampled_forced_eos'][:, :, -1].all())
                self.assertFalse(result['sampled_mask'][:, :, (0, -1)].any())
                self.assertTrue(result['sampled_mask'][:, :, 1:-1].all())
                self.assertTrue(torch.isfinite(result['sampled_log_probs']).all())
                (-result['sampled_log_probs'].sum()).backward()
                self.assertGreater(features.grad.abs().sum().item(), 0.)
                self.assertGreater(model.cls.predictions.bias.grad.abs().sum().item(), 0.)

    def test_112_slot_parallel_queries_match_long_prefixes_with_safe_positions(self):
        slot = 112
        model = tiny_model().eval()
        features, mask = inputs(batch=1, slot=slot)
        tokens = (torch.arange(2 * slot).reshape(1, 2, slot) % 7) + 3
        tokens[:, :, 0], tokens[:, :, -1] = 1, 2
        tokens[:, 0, 15] = 0  # PAD before EOS must preserve the two distinct prefix rules.
        probes = [1, 2, 55, 56, slot - 2, slot - 1]
        with torch.no_grad():
            expected = sequential_logits(model, features, mask, tokens, positions=probes)
        with mock.patch.object(model.bert, 'forward', wraps=model.bert.forward) as forward:
            actual = _parallel_query_logits(model, features, mask, tokens, SPECIAL)
        self.assertEqual(actual.shape, (1, 2, slot - 1, 12))
        bert_args, bert_kwargs = forward.call_args
        # Parallel content/query packing is longer than BERT's 512-position
        # table, but reuses each query's original absolute position <= 223.
        self.assertEqual(bert_args[0].shape, (1, 5 * slot - 2))
        self.assertEqual(bert_kwargs['attention_mask'].shape,
                         (1, 5 * slot - 2 + 3, 5 * slot - 2 + 3))
        self.assertEqual(int(bert_kwargs['position_ids'].max()), 2 * slot - 1)
        self.assertLess(int(bert_kwargs['position_ids'].max()),
                        model.bert.embeddings.position_embeddings.num_embeddings)
        selected = actual.index_select(2, torch.tensor(probes) - 1)
        torch.testing.assert_close(selected, expected, atol=2e-6, rtol=1e-5)
        (-actual.log_softmax(-1)[..., 4].mean()).backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0.)
        query_gradient = model.bert.encoder.layer[0].attention.self.query.weight.grad
        self.assertGreater(query_gradient.abs().sum().item(), 0.)

    def test_112_slot_natural_and_forced_eos_have_separate_gradient_masks(self):
        slot, natural_position = 112, 73
        model = tiny_model().eval()
        features, mask = inputs(batch=1, slot=slot)
        tokens = torch.full((1, 2, slot), 4, dtype=torch.long)
        tokens[:, :, 0] = 1
        tokens[:, 0, natural_position] = 2
        tokens[:, 0, natural_position + 1:] = 0
        tokens[:, 1, -1] = 2
        forced = torch.zeros_like(tokens, dtype=torch.bool)
        forced[:, 1, -1] = True
        scores, valid = sampled_token_log_probs(
            model, features, mask, tokens, SPECIAL, forced_eos=forced)
        self.assertEqual(tuple(valid.sum(-1).flatten().tolist()), (natural_position, slot - 2))
        self.assertTrue(valid[:, 0, natural_position].all())
        self.assertFalse(valid[:, 0, natural_position + 1:].any())
        self.assertFalse(valid[:, 1, -1].any())
        self.assertEqual(float(scores[:, 1, -1].item()), 0.)
        forced_gradient = torch.autograd.grad(scores[:, 1, -1].sum(),
                                             model.cls.predictions.bias, retain_graph=True)[0]
        self.assertEqual(float(forced_gradient.abs().sum()), 0.)
        (-scores[:, 0, natural_position].sum()).backward()
        self.assertGreater(model.cls.predictions.bias.grad[2].abs().item(), 0.)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0.)

    def test_top_k_and_bfloat16_are_finite(self):
        if int(torch.__version__.split('.')[0]) < 2:
            self.skipTest('torch 1.x CPU LayerNorm lacks mixed BF16/FP32 support; CUDA is tested separately')
        model = tiny_model().train()
        features, mask = inputs(batch=1)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            result = scst_decode(model, features, mask, SPECIAL, sample_n=1,
                                 max_length=5, top_k=3, temperature=.8)
            loss = -result['sampled_log_probs'].sum()
        self.assertEqual(result['sampled_log_probs'].dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        decoder_gradient = model.bert.encoder.layer[0].attention.self.query.weight.grad
        self.assertIsNotNone(decoder_gradient)
        self.assertGreater(decoder_gradient.abs().sum().item(), 0.)

    def test_masks_preserve_visual_attention_and_natural_eos(self):
        visual = torch.eye(3)[None].requires_grad_()
        mask = generation_attention_mask(visual, 3, 35)
        self.assertEqual(mask.shape, (1, 73, 73))
        self.assertTrue(torch.equal(mask[:, 70:, 70:], visual))
        self.assertEqual(mask[0, 36, 1].item(), 1.)
        self.assertEqual(mask[0, 1, 36].item(), 0.)
        self.assertFalse(mask[:, 70:, :70].any())
        mask.sum().backward()
        self.assertTrue(torch.equal(visual.grad, torch.ones_like(visual)))
        tokens = torch.tensor([[[1, 4, 2, 9, 0], [1, 4, 0, 5, 2]]])
        self.assertEqual(caption_token_mask(tokens, [2], 0).tolist(),
                         [[[False, True, True, False, False], [False, True, True, True, True]]])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
