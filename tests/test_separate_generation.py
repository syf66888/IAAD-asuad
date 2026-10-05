"""Regression tests for action/explanation generation with real search code."""

import types
import unittest

import torch
from torch import nn

from src.layers.bert.modeling_bert import BertForImageCaptioning
from src.layers.bert.modeling_utils import BeamHypotheses


class ScriptedCaptioner(BertForImageCaptioning):
    """Deterministic logits; actual generation, masks, caching and beam search."""
    def __init__(self, cached=False, action_never_ends=False):
        nn.Module.__init__(self)
        self.config = types.SimpleNamespace(vocab_size=12)
        self.anchor = nn.Parameter(torch.zeros(1))
        self.cached = cached
        self.action_never_ends = action_never_ends
        self.calls = []

    def forward(self, input_ids, position_ids, attention_mask, img_feats=None, **kwargs):
        self.calls.append((input_ids.clone(), position_ids.clone(), attention_mask.clone()))
        batch, length = input_ids.shape
        scores = torch.full((batch, length, 12), -16., device=input_ids.device)
        for b in range(batch):
            sample = int(self.img_feats[b, 0, 0])
            for index, position in enumerate(position_ids[b].tolist()):
                if position < 5:
                    token = (3 + sample) if self.action_never_ends or position == 1 else 2
                else:
                    token = (6 + sample) if position == 6 else 2
                scores[b, index, token] = 16.
        if not self.cached:
            return (scores,)
        hidden = input_ids.float().unsqueeze(-1)
        if img_feats is not None:
            hidden = torch.cat((hidden, torch.zeros(batch, img_feats.shape[1], 1)), dim=1)
        return scores, (hidden,)


def generate(model, beams=3, separate=True):
    length = 10 if separate else 5
    batch = 2
    image_features = torch.tensor([[[0.], [0.]], [[1.], [1.]]])
    mask = torch.ones(batch, length + 2, length + 2)
    mask[:, :length, :length] = torch.tril(torch.ones(length, length))
    mask[:, length:, :length] = 0
    original_mask = mask.clone()
    output = model.generate(
        img_feats=image_features, attention_mask=mask,
        masked_pos=torch.ones(batch, length, dtype=torch.long),
        token_type_ids=torch.cat((torch.zeros(batch, 5), torch.ones(batch, length - 5)), dim=1).long(),
        input_ids=torch.zeros(batch, length, dtype=torch.long),
        max_length=length, do_sample=False, num_beams=beams, temperature=1.,
        top_k=0, top_p=1., repetition_penalty=1., bos_token_id=1,
        pad_token_id=0, eos_token_ids=[2], mask_token_id=11,
        length_penalty=1., num_return_sequences=1, num_keep_best=1,
        is_decode=True, add_od_labels=False, od_labels_start_posid=5,
        use_sep_cap=separate,
    )
    assert torch.equal(mask, original_mask)
    return output


class SeparateGenerationTests(unittest.TestCase):
    def test_greedy_and_beam_generate_both_slots_with_or_without_cache(self):
        for beams in (1, 3):
            for cached in (False, True):
                with self.subTest(beams=beams, cached=cached):
                    model = ScriptedCaptioner(cached=cached)
                    ids, scores = generate(model, beams)
                    self.assertEqual(ids.shape, (2, 1, 10))
                    self.assertTrue(torch.isfinite(scores).all())
                    for sample in range(2):
                        self.assertEqual(ids[sample, 0].tolist(), [1, 3 + sample, 2, 0, 0, 1, 6 + sample, 2, 0, 0])
                    reason_start = [call for call in model.calls if call[1][0, -1].item() == 6][0]
                    self.assertEqual(reason_start[0].shape[1], 7)
                    # Explanation attends to action content/EOS, not its pads.
                    self.assertEqual(reason_start[2][0, 6, 1].item(), 1)
                    self.assertEqual(reason_start[2][0, 6, 3].item(), 0)
                    self.assertEqual(reason_start[2][0, 6, 4].item(), 0)

    def test_action_length_limit_forces_eos_and_still_starts_explanation(self):
        for beams in (1, 3):
            model = ScriptedCaptioner(cached=True, action_never_ends=True)
            ids, _ = generate(model, beams)
            self.assertTrue(torch.equal(ids[:, 0, 4], torch.tensor([2, 2])))
            self.assertTrue(torch.equal(ids[:, 0, 5], torch.tensor([1, 1])))
            self.assertTrue(torch.equal(ids[:, 0, 6], torch.tensor([6, 7])))

    def test_reason_length_penalty_excludes_fixed_action_prefix(self):
        short = BeamHypotheses(1, 5, 1., False)
        prefixed = BeamHypotheses(1, 10, 1., False, length_offset=5)
        short.add(torch.tensor([1, 6, 7]), -3.)
        prefixed.add(torch.tensor([1, 3, 2, 0, 0, 1, 6, 7]), -3.)
        self.assertEqual(short.hyp[0][0], prefixed.hyp[0][0])
        self.assertEqual(short.is_done(-4.), prefixed.is_done(-4.))

    def test_single_caption_keeps_single_slot(self):
        for beams in (1, 3):
            ids, _ = generate(ScriptedCaptioner(cached=True), beams, separate=False)
            self.assertEqual(ids.shape, (2, 1, 5))
            self.assertEqual(ids[0, 0].tolist(), [1, 3, 2, 0, 0])

    def test_real_bert_cached_and_uncached_generation_agree(self):
        from src.layers.bert import BertConfig
        torch.manual_seed(88)
        config = BertConfig(
            vocab_size_or_config_json_file=12, hidden_size=16,
            num_hidden_layers=2, num_attention_heads=4, intermediate_size=32,
        )
        config.img_feature_dim = 1
        config.img_feature_type = 'frcnn'
        config.use_img_layernorm = False
        config.hidden_dropout_prob = 0.
        config.attention_probs_dropout_prob = 0.
        model = BertForImageCaptioning(config).eval()
        with torch.no_grad():
            # Exercise several explanation cache steps, not immediate EOS.
            model.cls.predictions.bias[2] = -10.
            for beams in (1, 3):
                model.bert.encoder.output_hidden_states = False
                plain_ids, plain_scores = generate(model, beams)
                model.bert.encoder.output_hidden_states = True
                cached_ids, cached_scores = generate(model, beams)
                self.assertTrue(torch.equal(plain_ids, cached_ids))
                self.assertTrue(torch.allclose(plain_scores, cached_scores, atol=1e-5, rtol=1e-5))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
