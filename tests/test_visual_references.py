"""Semantic regression checks for visual input, timing, ROI and flow units."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.modeling.visual_references import (
    VisualReferenceExtractor, align_temporal_features, denormalize_rgb,
    nearest_temporal_samples, pool_object_references, temporal_sample_indices,
)


def normalized(rgb):
    mean = rgb.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = rgb.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    return (rgb - mean) / std


class VisualReferenceTests(unittest.TestCase):
    def test_denormalize_preserves_black_white_and_channels(self):
        rgb = torch.zeros(1, 2, 3, 4, 6)
        rgb[:, 1] = 1
        rgb[:, 0, 0] = 0.75
        torch.testing.assert_close(denormalize_rgb(normalized(rgb)), rgb)
        with self.assertRaises(ValueError):
            denormalize_rgb(torch.full_like(rgb, float("nan")))

    def test_temporal_bins_select_correct_source_frames(self):
        selected = temporal_sample_indices(32, 8)
        self.assertEqual(selected.tolist(), [0, 4, 9, 13, 18, 22, 27, 31])
        self.assertEqual(nearest_temporal_samples(32, 16, selected).tolist(),
                         [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7])
        self.assertEqual(temporal_sample_indices(1, 8).tolist(), [0])

    def test_temporal_motion_is_aligned_then_broadcast_across_space(self):
        steps = torch.arange(8).view(1, 8, 1).float().expand(2, -1, 4)
        aligned = align_temporal_features(steps, 4)
        torch.testing.assert_close(aligned[0, :, 0], torch.tensor([0.5, 2.5, 4.5, 6.5]))
        tokens = aligned[:, :, None, None, :].expand(2, 4, 3, 5, 4).reshape(2, 60, 4)
        torch.testing.assert_close(tokens[0, :15], torch.full((15, 4), 0.5))
        torch.testing.assert_close(tokens[0, 45:], torch.full((15, 4), 6.5))

    def test_roi_preserves_batch_time_channels_and_gradient(self):
        grid = torch.arange(6).view(1, 1, 1, 1, 6).expand(2, 2, 2, 4, 6).float().clone()
        grid += torch.tensor([0, 100]).view(2, 1, 1, 1, 1)
        grid += torch.tensor([0, 1000]).view(1, 2, 1, 1, 1)
        grid += torch.tensor([0, 10]).view(1, 1, 2, 1, 1)
        grid.requires_grad_()
        left = torch.tensor([[0., 0., .5, 1., 1.]])
        right = torch.tensor([[.5, 0., 1., 1., 1.]])
        out = pool_object_references(grid, [left, right, right, left], torch.tensor([0, 3]), 4)
        self.assertEqual(tuple(out.shape), (2, 2, 2))
        torch.testing.assert_close(out[..., 1] - out[..., 0], torch.full((2, 2), 1000.))
        self.assertGreater(float(out[1, 0, 0] - out[0, 0, 0]), 102.)
        self.assertLess(float(out[1, 1, 0] - out[0, 1, 0]), 98.)
        out.sum().backward()
        self.assertGreater(float(grid.grad.abs().sum()), 0.)

    def test_empty_or_invalid_boxes_fall_back_to_spatial_mean(self):
        grid = torch.randn(2, 3, 2, 4, 6, requires_grad=True)
        empty = torch.empty(0, 5)
        invalid = torch.tensor([[.8, .8, .1, .1, 1.]])
        out = pool_object_references(grid, [empty, invalid, empty, empty], torch.tensor([0, 3]), 4)
        torch.testing.assert_close(out, grid.mean((-2, -1)).transpose(1, 2))

    def test_identical_frames_zero_motion_and_shape(self):
        extractor = VisualReferenceExtractor(SimpleNamespace(vision_yolo_weights=""))
        rgb = torch.rand(2, 1, 3, 64, 96).expand(-1, 4, -1, -1, -1)
        grid = torch.rand(2, 7, 2, 4, 6)
        objects, motion = extractor(normalized(rgb), grid)
        self.assertEqual(tuple(objects.shape), (2, 2, 7))
        torch.testing.assert_close(motion, torch.zeros(2, 2, 4))

    def test_dis_translation_has_correct_direction_and_normalized_units(self):
        rng = np.random.default_rng(41)
        image = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
        shifted = np.roll(image, 3, axis=1)
        rgb = torch.from_numpy(np.stack([image, shifted])).permute(0, 3, 1, 2).float()[None] / 255
        extractor = VisualReferenceExtractor(SimpleNamespace(vision_yolo_weights="", vision_flow_size=96))
        flow = extractor._motion(rgb)
        self.assertGreater(float(flow[0, 1, 0]), 0.015)
        self.assertLess(float(flow[0, 1, 0]), 0.05)
        self.assertLess(abs(float(flow[0, 1, 1])), 0.01)
        torch.testing.assert_close(flow[:, 0], torch.zeros(1, 4))

    def test_missing_checkpoint_is_explicit_error(self):
        with self.assertRaises(FileNotFoundError):
            VisualReferenceExtractor(SimpleNamespace(vision_yolo_weights="missing_detector_938726.pt"))

    def test_cached_motion_skips_online_flow_and_aligns_time(self):
        extractor = VisualReferenceExtractor(SimpleNamespace(vision_yolo_weights=""))
        def fail_online_flow(_):
            raise AssertionError("Online optical flow must not run when cache features are supplied")
        extractor._motion = fail_online_flow
        rgb = torch.zeros(1, 4, 3, 32, 32)
        grid = torch.rand(1, 7, 2, 2, 2)
        features = torch.arange(4).view(1, 4, 1).float().expand(-1, -1, 4)
        _, motion = extractor(normalized(rgb), grid, motion_features=features)
        torch.testing.assert_close(motion[0, :, 0], torch.tensor([0.5, 2.5]))
        with self.assertRaises(ValueError):
            extractor(normalized(rgb), grid, motion_features=features[:, :2])

    def test_detector_excludes_caption_bfloat16_autocast(self):
        extractor = VisualReferenceExtractor(SimpleNamespace(vision_yolo_weights=""))
        extractor.detector = torch.nn.Linear(2, 2).bfloat16()
        def inspect_precision(rgb, selected):
            self.assertFalse(torch.is_autocast_cpu_enabled())
            self.assertEqual(rgb.dtype, torch.float32)
            self.assertEqual(next(extractor.detector.parameters()).dtype, torch.float32)
            return [rgb.new_empty((0, 5))]
        extractor._detect_fp32 = inspect_precision
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            extractor._detect(torch.zeros(1, 1, 3, 32, 32).bfloat16(), torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
