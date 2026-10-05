"""Semantic checks for temporal alignment, fusion gradients and soft masks."""

import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.modeling.reference_fusion import (
    ReferenceFeatureFusion, attention_mask_to_bias, video_grid_to_tokens,
)
from src.modeling.asuad_model import AsuadModel


class TinySwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.norm = nn.LayerNorm(4)
        self.grid = nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(1, 4, 2, 2, 2))

    def forward(self, images):
        return self.grid.expand(images.shape[0], -1, -1, -1, -1)


class TinyCaptioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = types.SimpleNamespace(encoder=types.SimpleNamespace(output_attentions=False))
        self.last_inputs = None

    def forward(self, **kwargs):
        assert 'reference_features' not in kwargs
        assert 'motion_features' not in kwargs
        self.last_inputs = kwargs
        features = kwargs['img_feats']
        mask = kwargs['attention_mask']
        bias = attention_mask_to_bias(mask, features.dtype, self.bert.use_log_attention_mask)
        attention = bias.softmax(-1)
        positions = torch.arange(mask.shape[-1], device=features.device, dtype=features.dtype)
        return (features.square().mean() + (attention * positions).sum(-1).mean(), features)


class FakeReferences(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, images, grid, motion_features=None):
        objects = grid.mean((-1, -2)).transpose(1, 2)
        self.last_motion = motion_features
        if motion_features is not None:
            motion = torch.nn.functional.interpolate(
                motion_features.transpose(1, 2), size=grid.shape[2],
                mode='linear', align_corners=False,
            ).transpose(1, 2)
        else:
            motion = grid.new_zeros(grid.shape[0], grid.shape[2], 4)
        return objects, motion


def make_model(mode='gated', sensors=False, grid_feat=True):
    args = types.SimpleNamespace(
        use_checkpoint=False, freeze_backbone=False, img_feature_dim=4,
        grid_feat=grid_feat, mask_prob=0.5, max_img_seq_length=8,
        max_num_frames=4, use_car_sensor=sensors, learn_mask_enabled=True,
        sparse_mask_soft2hard=False, learn_mask_log_bias=True,
        feature_fusion=mode, fusion_hidden_dim=3, fusion_dropout=0.0,
    )
    with patch('src.modeling.visual_references.VisualReferenceExtractor', FakeReferences):
        return AsuadModel(args, None, TinySwin(), TinyCaptioner())


class ReferenceFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(88)

    def test_noncontiguous_grid_preserves_time_and_channels(self):
        grid = torch.arange(2 * 4 * 3 * 2 * 2).reshape(2, 4, 3, 2, 2).transpose(-1, -2)
        tokens, steps = video_grid_to_tokens(grid, 4)
        self.assertEqual(steps, 3)
        self.assertTrue(torch.equal(tokens[0, 0], grid[0, :, 0, 0, 0]))
        self.assertTrue(torch.equal(tokens[1, 11], grid[1, :, 2, 1, 1]))

    def test_mismatched_auxiliary_dimensions_project_and_backpropagate(self):
        model = ReferenceFeatureFusion(32, 8, reference_dim=10, dropout=0)
        tokens = torch.randn(2, 12, 32, requires_grad=True)
        reference = torch.randn(2, 3, 10, requires_grad=True)
        output = model(tokens, reference, temporal_steps=3)
        output.square().mean().backward()
        self.assertEqual(output.shape, tokens.shape)
        for gradient in (tokens.grad, reference.grad, model.hard_gate.weight.grad, model.aux_projection.weight.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        before = model.hard_gate.weight.detach().clone()
        torch.optim.SGD(model.parameters(), lr=.1).step()
        self.assertFalse(torch.equal(before, model.hard_gate.weight))

    def test_reference_layouts_and_identity_residual(self):
        model = ReferenceFeatureFusion(16, 8, dropout=0)
        tokens = torch.randn(2, 12, 16)
        for count in (1, 3, 5, 12):
            output = model(tokens, torch.randn(2, count, 16), temporal_steps=3)
            self.assertEqual(output.shape, tokens.shape)
        with torch.no_grad():
            model.residual_scale.zero_()
        self.assertTrue(torch.equal(model(tokens, temporal_steps=3), tokens))

    def test_soft_mask_has_gradients_in_half_precision_and_binary_zeros_stay_masked(self):
        probabilities = torch.tensor([[1., .5, 0.]], requires_grad=True)
        bias = attention_mask_to_bias(probabilities, torch.float16, use_log=True)
        self.assertEqual(bias[0, 0].item(), 0)
        self.assertEqual(bias[0, 2].item(), -10000)
        attention = bias.float().softmax(-1)
        self.assertGreater(attention[0, 1].item(), 0.3)
        attention[0, 1].backward()
        self.assertGreater(probabilities.grad[0, 1].abs().item(), 0)
        binary = torch.tensor([[1., 0.]])
        self.assertTrue(torch.equal(
            attention_mask_to_bias(binary, torch.float32, True),
            attention_mask_to_bias(binary, torch.float32, False),
        ))

    def test_none_mode_preserves_channels_even_when_grid_flag_false(self):
        model = make_model('none', grid_feat=False)
        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.weight[:, :4].copy_(torch.eye(4))
            model.fc.bias.zero_()
        images = torch.randn(2, 4, 3, 8, 8)
        output = model.encode_visual_features(images)
        expected, _ = video_grid_to_tokens(model.swin(images.permute(0, 2, 1, 3, 4)), 4)
        self.assertTrue(torch.equal(output, expected))

    def test_native_checkpoint_replays_cpu_bfloat16_and_preserves_parameter_gradients(self):
        class RecordingSwin(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.norm = nn.LayerNorm(4)
                self.project = nn.Conv3d(3, 4, 1)
                self.dtypes = []

            def forward(self, images):
                output = self.project(images[:, :, :2, :2, :2])
                self.dtypes.append(output.dtype)
                return output

        model = make_model('none')
        model.swin = RecordingSwin()
        model.use_checkpoint = True
        original_keys = list(model.state_dict())
        images = torch.randn(2, 4, 3, 8, 8)
        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            output = model.encode_visual_features(images)
            loss = output.float().square().mean()
        loss.backward()
        self.assertEqual(model.swin.dtypes, [torch.bfloat16, torch.bfloat16])
        self.assertGreater(model.swin.project.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.swin.project.weight.grad).all())
        self.assertEqual(list(model.state_dict()), original_keys)

    def test_motion_broadcast_tracks_swin_time_not_first_tokens(self):
        model = make_model('gated')
        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.weight[0, 4] = 1
            model.fc.bias.zero_()
            model.reference_fusion.residual_scale.zero_()
        reference = torch.zeros(2, 2, 8)
        reference[:, 0, 4] = 10
        reference[:, 1, 4] = 20
        output = model.encode_visual_features(torch.randn(2, 4, 3, 8, 8), reference)
        self.assertTrue(torch.equal(output[0, :, 0], torch.tensor([10.] * 4 + [20.] * 4)))

    def test_mask_excludes_sensor_suffix_and_preserves_input(self):
        model = make_model('gated', sensors=True)
        with torch.no_grad():
            model.learn_vid_att.weight.zero_()
        mask = torch.ones(2, 13, 13)
        reference = torch.randn(2, 2, 8, requires_grad=True)
        outputs = model(
            img_feats=torch.randn(2, 4, 3, 8, 8), attention_mask=mask,
            car_info=torch.randn(2, 2, 4), reference_features=reference,
        )
        forwarded = model.trans_encoder.last_inputs
        self.assertEqual(forwarded['img_feats'].shape, (2, 10, 4))
        self.assertEqual(forwarded['attention_mask'][0, 3, 4].item(), .5)
        self.assertEqual(forwarded['attention_mask'][0, 3, 3].item(), 1)
        self.assertTrue(torch.equal(forwarded['attention_mask'][:, 11:, :], mask[:, 11:, :]))
        self.assertTrue(torch.equal(mask, torch.ones_like(mask)))
        outputs[0].backward()
        self.assertGreater(model.learn_vid_att.weight.grad.abs().sum().item(), 0)
        self.assertGreater(reference.grad.abs().sum().item(), 0)

    def test_cached_per_frame_motion_reaches_extractor_without_leaking_to_bert(self):
        model = make_model('gated')
        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.weight[0, 4] = 1
            model.fc.bias.zero_()
            model.reference_fusion.residual_scale.zero_()
        motion = torch.zeros(2, 4, 4)
        motion[:, :, 0] = torch.tensor([0., 2., 4., 6.])
        model(img_feats=torch.randn(2, 4, 3, 8, 8),
              attention_mask=torch.ones(2, 11, 11), motion_features=motion)
        self.assertIs(model.visual_references.last_motion, motion)
        forwarded = model.trans_encoder.last_inputs
        self.assertNotIn('motion_features', forwarded)
        self.assertTrue(torch.equal(forwarded['img_feats'][0, :, 0], torch.tensor([1.] * 4 + [5.] * 4)))

    def test_legacy_uses_configured_budget_and_backward(self):
        model = make_model('legacy')
        reference = torch.randn(2, 2, 8, requires_grad=True)
        output = model.encode_visual_features(torch.randn(2, 4, 3, 8, 8), reference)
        self.assertEqual(output.shape, (2, 8, 4))
        output.square().mean().backward()
        self.assertGreater(reference.grad.abs().sum().item(), 0)

    def test_reload_attention_can_resize_up_and_down(self):
        model = make_model('none')
        for old_size in (4, 8, 12):
            model.reload_attn_mask(torch.full((old_size ** 2, 1), 2.))
            self.assertTrue(torch.equal(model.learn_vid_att.weight, torch.full((64, 1), 2.)))

    def test_real_bert_uses_log_bias_and_trains_the_gate(self):
        from src.layers.bert import BertConfig
        from src.layers.bert.modeling_bert import BertImgModel
        config = BertConfig(
            vocab_size_or_config_json_file=24, hidden_size=16,
            num_hidden_layers=1, num_attention_heads=4, intermediate_size=32,
        )
        config.img_feature_dim = 4
        config.img_feature_type = 'frcnn'
        config.use_img_layernorm = False
        config.hidden_dropout_prob = 0.
        config.attention_probs_dropout_prob = 0.
        model = BertImgModel(config)
        model.use_log_attention_mask = True
        gate = torch.tensor(.5, requires_grad=True)
        mask = torch.ones(1, 5, 5)
        mask[:, 3, 4] = gate
        output = model(
            torch.tensor([[1, 2, 3]]), img_feats=torch.randn(1, 2, 4),
            attention_mask=mask,
        )[0]
        output[0, 3, 0].backward()
        self.assertEqual(output.shape, (1, 5, 16))
        self.assertGreater(gate.grad.abs().item(), 1e-10)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
