"""Regression checks for real short-video TSV sampling and offline LK alignment."""
import base64
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.datasets.flow_cache import uniform_indices
from src.datasets.vision_language_tsv import VisionLanguageTSVDataset
from src.datasets.lk_flow_cache import SparseLKFlowCache
from scripts import precompute_lk_flow


class MMAUShortFrames(unittest.TestCase):
    def make_row(self):
        frames = []
        encoded = []
        rng = np.random.RandomState(88)
        texture = rng.randint(20, 190, (32, 32, 3), dtype=np.uint8)
        for i in range(3):
            rgb = np.roll(texture, i, axis=1)
            rgb[0, 0] = [30 + i * 40, 20, 10]
            frames.append(rgb)
            ok, data = cv2.imencode('.png', rgb[:, :, ::-1])
            self.assertTrue(ok)
            encoded.append(base64.b64encode(data).decode())
        return ['training_short', '{}', *encoded], np.asarray(frames)

    def test_rgb_short_clip_keeps_all_frames_in_cache_order(self):
        row, frames = self.make_row()
        dataset = object.__new__(VisionLanguageTSVDataset)
        dataset.cfg = {'preextracted_frames': True}
        dataset.decoder_num_frames = 32
        dataset.flow_cache = object()
        dataset.visual_tsv = None
        dataset.get_row_from_tsv = lambda *_: row
        decoded, video = dataset.get_visual_data(0)
        self.assertTrue(video)
        expected = frames[uniform_indices(3, 32)].transpose(0, 3, 1, 2)
        np.testing.assert_array_equal(decoded, expected)
        self.assertEqual(set(uniform_indices(3, 32)), {0, 1, 2})

    def test_real_lk_preprocessor_accepts_short_tsv_and_records_same_indices(self):
        row, _ = self.make_row()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'training.img.tsv'
            source.write_text('\t'.join(row) + '\n')
            source.with_suffix('.lineidx').write_text('0\n')
            cache = SparseLKFlowCache(root / 'cache', source, 32, 32, create=True)
            try:
                precompute_lk_flow.worker_init(root / 'cache', str(source), 32, 32, False)
                index, status, _ = precompute_lk_flow.process_row(0)
                self.assertEqual((index, status), (0, 'written'))
                with np.load(cache.path(0), allow_pickle=False) as result:
                    metadata = json.loads(str(result['metadata'].item()))
                    self.assertEqual(metadata['sample_indices'], uniform_indices(3, 32))
                    self.assertEqual(metadata['source_frame_count'], 3)
                    self.assertTrue(np.isfinite(result['tracks']).all())
                    self.assertEqual(result['tracks'].shape, (32, 200, 4))
            finally:
                precompute_lk_flow._SOURCE.close()

    def test_empty_clips_are_rejected_and_single_frame_is_repeated(self):
        for values in [(0, 32), (10, 0), (-1, 32)]:
            with self.assertRaises(ValueError):
                uniform_indices(*values)
        self.assertEqual(uniform_indices(1, 32), [0] * 32)

    def test_existing_long_clip_sampling_is_unchanged(self):
        for count in [32, 33, 64, 97]:
            expected = [int(round(i * (count - 1) / 31)) for i in range(32)]
            self.assertEqual(uniform_indices(count, 32), expected)


if __name__ == '__main__':
    unittest.main()
