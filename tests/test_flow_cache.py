import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.datasets.flow_cache import DenseFlowCache, crop_flow_statistics, uniform_indices


class DenseFlowCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source = self.root / "frames.tsv"
        self.source.write_bytes(b"clip_a\tmetadata\tframes\nclip_b\tmetadata\tframes\n")
        self.source.with_suffix(".lineidx").write_text("0\n23\n")
        self.cache = DenseFlowCache(self.root / "cache", self.source, num_frames=4, create=True)
        self.flow = np.zeros((4, 2, 64, 112), dtype=np.float32)
        self.flow[1:, 0] = 0.125
        self.row_hash = hashlib.sha256(b"source row").hexdigest()

    def tearDown(self):
        self.directory.cleanup()

    def write(self):
        self.cache.write(0, self.flow, "clip_a", (720, 1280), 6, uniform_indices(6, 4), self.row_hash)

    def test_atomic_roundtrip_and_missing_cache_error(self):
        self.write()
        restored = self.cache.load(0, source_size=(720, 1280), expected_key="clip_a", row_sha256=self.row_hash)
        np.testing.assert_array_equal(restored, self.flow)
        self.assertEqual(restored.dtype, np.float32)
        self.assertEqual(list(self.cache.directory.glob("*.tmp")), [])
        with self.assertRaises(RuntimeError):
            self.cache.load(1)

    def test_reject_stale_source_key_dimensions_and_row_digest(self):
        self.write()
        for kwargs in [dict(source_size=(224, 224)), dict(expected_key="clip_b"), dict(row_sha256="0" * 64)]:
            with self.assertRaises(ValueError):
                self.cache.load(0, **kwargs)
        self.source.write_bytes(self.source.read_bytes() + b"changed")
        with self.assertRaises(FileNotFoundError):
            DenseFlowCache(self.root / "cache", self.source, num_frames=4)

    def test_reject_sampling_and_config_mismatch(self):
        with self.assertRaises(ValueError):
            self.cache.write(0, self.flow, "clip_a", (720, 1280), 6, [0, 1, 2, 3], self.row_hash)
        self.write()
        with self.assertRaises(FileNotFoundError):
            DenseFlowCache(self.root / "cache", self.source, num_frames=3)
        with np.load(self.cache.path(0), allow_pickle=False) as record:
            metadata = json.loads(str(record["metadata"].item()))
        metadata["sample_indices"] = [0, 1, 2, 3]
        np.savez_compressed(self.cache.path(0), flow=self.flow.astype(np.float16), metadata=np.array(json.dumps(metadata)))
        with self.assertRaises(ValueError):
            self.cache.load(0)

    def test_crop_uses_matching_pixels_and_rescales_both_vector_axes(self):
        self.flow[1:, 0, :, :56] = 0.1
        self.flow[1:, 0, :, 56:] = 0.3
        self.flow[1:, 1] = 0.2
        left = crop_flow_statistics(self.flow, (0, 0, 0.5, 0.5))
        right = crop_flow_statistics(self.flow, (0.5, 0.5, 0.5, 0.5))
        torch.testing.assert_close(left[1:, 0], torch.full((3,), 0.2))
        torch.testing.assert_close(right[1:, 0], torch.full((3,), 0.6))
        torch.testing.assert_close(left[1:, 1], torch.full((3,), 0.4))
        torch.testing.assert_close(left[1:, 2], torch.full((3,), float(np.hypot(0.2, 0.4))))
        torch.testing.assert_close(left[0], torch.zeros(4))

    def test_horizontal_flip_reverses_dx_but_preserves_magnitude(self):
        regular = crop_flow_statistics(self.flow)
        flipped = crop_flow_statistics(self.flow, horizontal_flip=True)
        torch.testing.assert_close(regular[:, 0], -flipped[:, 0])
        torch.testing.assert_close(regular[:, 1:], flipped[:, 1:])

    def test_invalid_crop_fails_explicitly(self):
        for crop in [(0, 0, 1, 0), (0.5, 0, 0.75, 1), (0, -0.1, 1, 1), (0, 0, float("nan"), 1)]:
            with self.assertRaises(ValueError):
                crop_flow_statistics(self.flow, crop)


if __name__ == "__main__":
    unittest.main()
