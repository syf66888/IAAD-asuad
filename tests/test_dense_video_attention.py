"""CPU checks for the dense-video ablation, including caption causality.

No external model weights, detector, dataset, or GPU is required.
"""

import ast
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.datasets.caption_tensorizer import CaptionTensorizer
from src.modeling.reference_fusion import attention_mask_to_bias
from src.modeling.scst_decode import generation_attention_mask, _parallel_query_logits
from src.modeling.asuad_model import AsuadModel


VISUAL = 8
SLOT = 5
TEXT = 2 * SLOT


def tensorizer_mask(train=True):
    tensorizer = CaptionTensorizer(
        tokenizer=None, max_img_seq_length=VISUAL, max_seq_length=TEXT,
        max_seq_a_length=SLOT, attn_mask_type='learn_vid_att',
        is_train=train, text_mask_type='random', use_sep_cap=True,
    )
    return tensorizer.get_attn_masks(3, 4).unsqueeze(0)


class TinySwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.norm = nn.LayerNorm(4)
        self.grid = nn.Parameter(torch.arange(32).float().reshape(1, 4, 2, 2, 2))

    def forward(self, images):
        return self.grid.expand(images.shape[0], -1, -1, -1, -1)


class References(nn.Module):
    def __init__(self, args, feature_dim=None):
        super().__init__()

    def forward(self, images, grid, motion_features=None):
        objects = grid.mean((-1, -2)).transpose(1, 2)
        motion = torch.nn.functional.interpolate(
            motion_features.transpose(1, 2), size=grid.shape[2],
            mode='linear', align_corners=False,
        ).transpose(1, 2)
        return objects, motion


class CaptionSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = types.SimpleNamespace(
            encoder=types.SimpleNamespace(output_attentions=False))
        self.last = None

    def forward(self, **kwargs):
        self.last = kwargs
        features = kwargs['img_feats']
        return features.square().mean(), features


def make_model(learned=False):
    args = types.SimpleNamespace(
        use_checkpoint=False, freeze_backbone=False, img_feature_dim=4,
        grid_feat=True, mask_prob=0.5, max_img_seq_length=VISUAL,
        max_num_frames=4, use_car_sensor=False, learn_mask_enabled=learned,
        sparse_mask_soft2hard=False, learn_mask_log_bias=False,
        feature_fusion='gated', fusion_hidden_dim=3, fusion_dropout=0,
    )
    with patch('src.modeling.visual_references.VisualReferenceExtractor', References):
        model = AsuadModel(args, None, TinySwin(), CaptionSpy())
    return model


def visual_inputs():
    return dict(
        img_feats=torch.zeros(1, 4, 3, 4, 4),
        motion_features=torch.tensor([[[0., 0., 0., 0.],
                                       [1., 0., 1., 1.],
                                       [2., 0., 2., 1.],
                                       [3., 0., 3., 1.]]]),
        attention_mask=tensorizer_mask(), input_ids=torch.zeros(1, TEXT, dtype=torch.long),
        car_info=torch.ones(1, 1, 4) * 100,
    )


class DenseVideoAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(88)

    def test_tensorizer_is_dense_visual_and_preserves_text_causality(self):
        for train in (False, True):
            mask = tensorizer_mask(train)[0]
            self.assertTrue(torch.equal(mask[TEXT:, TEXT:], torch.ones(VISUAL, VISUAL, dtype=mask.dtype)))
            self.assertEqual(int(mask[TEXT:, :TEXT].sum()), 0)
            self.assertEqual(int(mask[:3, SLOT:TEXT].sum()), 0)
            self.assertEqual(int(mask[0, 1:3].sum()), 0)
            self.assertEqual(int(mask[SLOT, SLOT + 1:SLOT + 4].sum()), 0)
            self.assertTrue(torch.equal(mask[SLOT:SLOT + 4, :3], torch.ones(4, 3, dtype=mask.dtype)))
            self.assertEqual(int(mask[:TEXT, 3:SLOT].sum()), 0)
            self.assertEqual(int(mask[:TEXT, SLOT + 4:TEXT].sum()), 0)

    def test_disabled_mask_removes_parameter_and_passes_original_mask_unchanged(self):
        dense = make_model(False)
        learned = make_model(True)
        self.assertNotIn('learn_vid_att.weight', dense.state_dict())
        self.assertEqual(sum(p.numel() for p in learned.parameters()) -
                         sum(p.numel() for p in dense.parameters()), VISUAL ** 2)
        inputs = visual_inputs()
        original_mask = inputs['attention_mask'].clone()
        outputs = dense(**inputs)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(dense.trans_encoder.last['img_feats'].shape[1], VISUAL)
        self.assertTrue(torch.equal(dense.trans_encoder.last['attention_mask'], original_mask))
        self.assertTrue(torch.equal(inputs['attention_mask'], original_mask))
        outputs[0].backward()
        self.assertGreater(float(dense.fc.weight.grad[:, -4:].abs().sum()), 0)
        self.assertGreater(float(dense.reference_fusion.aux_projection.weight.grad.abs().sum()), 0)
        self.assertIsNone(dense.expand_car_info.weight.grad)

    def test_dense_bias_has_no_hidden_offdiagonal_penalty_in_fp32_or_bf16(self):
        dense = make_model(False)
        dense(**visual_inputs())
        mask = dense.trans_encoder.last['attention_mask']
        for dtype in (torch.float32, torch.bfloat16):
            bias = attention_mask_to_bias(mask, dtype=dtype, use_log=False)
            self.assertEqual(float(bias[:, TEXT:, TEXT:].abs().sum()), 0)
            self.assertTrue((bias[:, TEXT:, :TEXT] <= -9900).all())
            probabilities = bias[:, TEXT:, TEXT:].float().softmax(-1)
            self.assertTrue(torch.allclose(probabilities, torch.full_like(probabilities, 1 / VISUAL)))

    def test_scst_outer_forward_keeps_dense_block_and_returns_zero_sparse_loss(self):
        model = make_model(False)

        def decode_spy(decoder, features, attention_mask, **options):
            return {'generation_mask': generation_attention_mask(attention_mask, features.shape[1], SLOT)}

        with patch('src.modeling.scst_decode.scst_decode', decode_spy):
            result = model(**visual_inputs(), scst=True, scst_options={})
        self.assertEqual(float(result['sparse_loss']), 0)
        mask = result['generation_mask']
        self.assertTrue((mask[:, TEXT:, TEXT:] == 1).all())
        self.assertEqual(int(mask[:, TEXT:, :TEXT].sum()), 0)
        self.assertEqual(int(torch.triu(mask[:, :TEXT, :TEXT], diagonal=1).sum()), 0)

    def test_scst_parallel_query_scoring_keeps_dense_video_and_blocks_text_feedback(self):
        class BertSpy:
            def __call__(self, input_ids, img_feats, attention_mask, **kwargs):
                self.mask = attention_mask
                return (torch.zeros(input_ids.shape[0], attention_mask.shape[-1], 4),)

        bert = BertSpy()
        decoder = types.SimpleNamespace(bert=bert, cls=nn.Linear(4, 12))
        features = torch.zeros(1, VISUAL, 4)
        mask = generation_attention_mask(tensorizer_mask(), VISUAL, SLOT)
        tokens = torch.tensor([[[1, 4, 2, 0, 0], [1, 6, 7, 2, 0]]])
        special = dict(bos_token_id=1, pad_token_id=0, eos_token_ids=[2], mask_token_id=11)
        logits = _parallel_query_logits(decoder, features, mask, tokens, special)
        self.assertEqual(tuple(logits.shape), (1, 2, SLOT - 1, 12))
        self.assertTrue((bert.mask[:, -VISUAL:, -VISUAL:] == 1).all())
        self.assertEqual(int(bert.mask[:, -VISUAL:, :-VISUAL].sum()), 0)

    def test_ce_flushes_short_epoch_tail_and_last_update(self):
        # Load the exact small scheduling helper without importing GPU trainer dependencies.
        source = Path(__file__).resolve().parents[1] / 'src/tasks/train.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                    and item.name == 'accumulation_window')
        module = ast.Module(body=[node], type_ignores=[])
        namespace = {}
        exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
        window = namespace['accumulation_window']
        per_epoch, epochs, accumulation = 5286, 49, 4
        maximum = per_epoch * epochs
        self.assertEqual(window(per_epoch - 1, maximum, per_epoch, accumulation), (2, False))
        self.assertEqual(window(per_epoch, maximum, per_epoch, accumulation), (2, True))
        self.assertEqual(window(maximum, maximum, per_epoch, accumulation), (2, True))
        updates = sum(window(i, maximum, per_epoch, accumulation)[1] for i in range(1, maximum + 1))
        self.assertEqual(updates, 64778)


if __name__ == '__main__':
    unittest.main()
