import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from src.datasets.flow_cache import uniform_indices
from src.datasets.lk_flow_cache import (SparseLKFlowCache, compute_sparse_lk,
                                       crop_lk_statistics_numpy, resized_shape)


def track_record(displacements, starts=None, shape=(224, 448)):
    vectors = np.asarray(displacements, dtype=np.float32).reshape(-1, 2)
    if starts is None:
        starts = np.tile(np.array([[100, 100]], dtype=np.float32), (len(vectors), 1))
    starts = np.asarray(starts, dtype=np.float32)
    tracks = np.zeros((3, 200, 4), dtype=np.float32)
    valid = np.zeros((3, 200), dtype=np.bool_)
    for t in (1, 2):
        tracks[t, :len(vectors)] = np.concatenate((starts, starts + vectors), axis=1)
        valid[t, :len(vectors)] = True
    return {"tracks": tracks, "valid": valid, "resized_size": shape}


class LKStatisticsTests(unittest.TestCase):
    def test_pixel_units_and_crop_selection(self):
        record = track_record([[6, 8], [-15, 20]], starts=[[100, 100], [300, 100]])
        left = crop_lk_statistics_numpy(record, (0, 0, 1, 0.5))
        right = crop_lk_statistics_numpy(record, (0, 0.5, 1, 0.5))
        np.testing.assert_allclose(left[1], [6, 8, 10, 0.6], atol=1e-6)
        np.testing.assert_allclose(right[1], [-15, 20, 25, -0.6], atol=1e-6)
        np.testing.assert_array_equal(left[0], np.zeros(4))
        # A final image resize rescales displacements before angle/magnitude.
        half = crop_lk_statistics_numpy(record, (0, 0, 1, 0.5), output_size=(112, 112))
        np.testing.assert_allclose(half[1], [3, 4, 5, 0.6], atol=1e-6)

    def test_circular_mean_cosine_is_not_mean_cosine(self):
        # Angles -60,+60 have mean cosine 0.5 but mean direction 0 => cosine 1.
        record = track_record([[5, 5 * np.sqrt(3)], [5, -5 * np.sqrt(3)]], shape=(224, 224))
        values = crop_lk_statistics_numpy(record)
        np.testing.assert_allclose(values[1], [5, 0, 10, 1], atol=1e-5)
        record = track_record([[-5, 5 * np.sqrt(3)], [-5, -5 * np.sqrt(3)]], shape=(224, 224))
        self.assertAlmostEqual(float(crop_lk_statistics_numpy(record)[1, 3]), -1, places=6)

    def test_horizontal_flip_changes_dx_and_direction_not_magnitude(self):
        record = track_record([[6, 8]], shape=(224, 224))
        flipped = crop_lk_statistics_numpy(record, horizontal_flip=True)
        np.testing.assert_allclose(flipped[1], [-6, 8, 10, -0.6], atol=1e-6)

    def test_no_tracks_zero_and_stationary_tracks_match_legacy_formula(self):
        empty = track_record([], shape=(224, 224))
        np.testing.assert_array_equal(crop_lk_statistics_numpy(empty), np.zeros((3, 4)))
        stationary = track_record([[0, 0]], shape=(224, 224))
        np.testing.assert_array_equal(crop_lk_statistics_numpy(stationary)[1], [0, 0, 0, 1])
        # Start is outside this crop, so it contributes nothing.
        np.testing.assert_array_equal(crop_lk_statistics_numpy(stationary, (0, 0.75, 1, 0.25)), np.zeros((3, 4)))

    def test_forward_lk_has_correct_signed_pixel_displacement(self):
        rng = np.random.RandomState(8)
        original = cv2.GaussianBlur(rng.randint(0, 256, (224, 224, 3), dtype=np.uint8), (5, 5), 0)
        for dx, dy in [(4, 0), (-4, 0), (0, 3)]:
            shifted = cv2.warpAffine(original, np.float32([[1, 0, dx], [0, 1, dy]]),
                                     (224, 224), borderMode=cv2.BORDER_REFLECT)
            record = compute_sparse_lk(np.stack((original, shifted)))
            self.assertGreater(int(record["valid"][1].sum()), 10)
            values = crop_lk_statistics_numpy(record)
            np.testing.assert_allclose(values[1, :3], [dx, dy, np.hypot(dx, dy)], atol=0.25)
            self.assertAlmostEqual(float(values[1, 3]), dx / np.hypot(dx, dy), delta=0.03)

    def test_constant_rgb_has_no_tracks_and_resize_matches_video_geometry(self):
        record = compute_sparse_lk(np.zeros((3, 300, 400, 3), dtype=np.uint8))
        self.assertEqual(record["resized_size"], (224, 298))
        self.assertEqual(resized_shape((400, 300)), (298, 224))
        self.assertFalse(record["valid"].any())
        np.testing.assert_array_equal(crop_lk_statistics_numpy(record), np.zeros((3, 4)))

    def test_invalid_inputs_fail(self):
        record = track_record([[1, 0]])
        for crop in [(0, 0, 1, 0), (0.5, 0, 1, 1), (0, -0.1, 1, 1), (0, 0, float("nan"), 1)]:
            with self.assertRaises(ValueError):
                crop_lk_statistics_numpy(record, crop)
        with self.assertRaises(ValueError):
            compute_sparse_lk(np.zeros((2, 224, 224, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            crop_lk_statistics_numpy(record, output_size=(0, 224))


class LKCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "frames.tsv"
        self.source.write_bytes(b"clip_a\tmeta\tframes\nclip_b\tmeta\tframes\n")
        self.source.with_suffix(".lineidx").write_text("0\n19\n", encoding="utf-8")
        self.cache = SparseLKFlowCache(self.root / "cache", self.source, num_frames=3, create=True)
        self.record = track_record([[6, 8]])
        self.row_hash = hashlib.sha256(b"source row").hexdigest()

    def tearDown(self):
        self.temporary.cleanup()

    def write(self):
        self.cache.write(0, self.record, "clip_a", (720, 1440), 5, uniform_indices(5, 3), self.row_hash)

    def test_atomic_roundtrip_and_missing_record(self):
        self.write()
        restored = self.cache.load(0, source_size=(720, 1440), expected_key="clip_a", row_sha256=self.row_hash)
        np.testing.assert_array_equal(restored["tracks"], self.record["tracks"])
        np.testing.assert_array_equal(restored["valid"], self.record["valid"])
        self.assertEqual(restored["resized_size"], self.record["resized_size"])
        self.assertEqual(len(list(self.cache.directory.glob("*.npz"))), 1)
        with self.assertRaises(RuntimeError):
            self.cache.load(1)

    def test_reject_identity_geometry_key_row_and_sampling_mismatch(self):
        self.write()
        for kwargs in [dict(source_size=(224, 224)), dict(expected_key="clip_b"), dict(row_sha256="0" * 64)]:
            with self.assertRaises(ValueError):
                self.cache.load(0, **kwargs)
        with self.assertRaises(FileNotFoundError):
            SparseLKFlowCache(self.root / "cache", self.source, num_frames=4)
        with self.assertRaises(FileNotFoundError):
            SparseLKFlowCache(self.root / "cache", self.source, num_frames=3, resize_short_side=112)
        with self.assertRaises(ValueError):
            self.cache.write(1, self.record, "clip_b", (720, 1440), 5, [0, 1, 2], self.row_hash)
        with self.assertRaises(ValueError):
            self.cache.write(1, self.record, "clip_b", (720, 1280), 5, [0, 2, 4], self.row_hash)
        with np.load(self.cache.path(0), allow_pickle=False) as raw:
            meta = json.loads(str(raw["metadata"].item()))
        for field, value in [("identity", "bad"), ("sample_indices", [0, 1, 2]), ("resized_size", [224, 447])]:
            altered = dict(meta)
            altered[field] = value
            np.savez_compressed(self.cache.path(0), tracks=self.record["tracks"], valid=self.record["valid"],
                                metadata=np.array(json.dumps(altered)))
            with self.assertRaises(ValueError):
                self.cache.load(0)

    def test_reject_changed_source_and_invalid_track_padding(self):
        self.write()
        self.record["valid"][0, 0] = True
        with self.assertRaises(ValueError):
            self.write()
        self.source.write_bytes(self.source.read_bytes() + b"changed")
        with self.assertRaises(FileNotFoundError):
            SparseLKFlowCache(self.root / "cache", self.source, num_frames=3)

    def test_precompute_cli_is_resumable_and_does_not_import_torch(self):
        frames = []
        for value in range(3):
            ok, png = cv2.imencode(".png", np.full((224, 224, 3), value * 50, dtype=np.uint8))
            self.assertTrue(ok)
            frames.append(base64.b64encode(png.tobytes()))
        source = self.root / "tiny.tsv"
        source.write_bytes(b"clip_test\t{}\t" + b"\t".join(frames) + b"\n")
        source.with_suffix(".lineidx").write_text("0\n", encoding="utf-8")
        yaml = self.root / "testing.yaml"
        yaml.write_text("img: tiny.tsv\n", encoding="utf-8")
        script = Path(__file__).resolve().parents[1] / "scripts" / "precompute_lk_flow.py"
        command = [sys.executable, str(script), "--yaml", str(yaml), "--cache-dir", str(self.root / "tiny_cache"),
                   "--num-frames", "3", "--workers", "1", "--log-every", "1"]
        first = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertIn('"status": "complete"', first.stdout)
        second = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertIn('"skipped": 1', second.stdout)
        cache = SparseLKFlowCache(self.root / "tiny_cache", source, num_frames=3)
        record = cache.load(0, expected_key="clip_test", source_size=(224, 224))
        self.assertFalse(record["valid"].any())
        # Preprocessing imports are deliberately safe on CPU-only environments.
        probe = subprocess.run([sys.executable, "-c", "import runpy,sys; runpy.run_path(sys.argv[1]); "
                                "assert 'torch' not in sys.modules", str(script)],
                               capture_output=True, text=True, check=True)
        self.assertEqual(probe.returncode, 0)


if __name__ == "__main__":
    unittest.main()
