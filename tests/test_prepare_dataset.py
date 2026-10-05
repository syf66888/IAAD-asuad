"""Exercise prepared media and TSVs using the real loader and LK cache."""
import base64
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import torch
import yaml

from src.preprocessing import prepare_dataset as prepare
from src.datasets.flow_cache import uniform_indices
from src.datasets.vl_dataloader import build_dataset
from src.layers.bert import BertTokenizer
from src.configs.config import SharedConfigs
from src.tasks.train import get_custom_args, check_arguments


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.media = self.root / 'media with spaces'
        self.media.mkdir()

    def frame_folder(self, key, count=28):
        folder = self.media / key
        folder.mkdir()
        for i in range(count):
            image = np.full((16, 24, 3), 20 + i, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(folder / ('frame%04d.png' % (i + 1))), image))
        return folder

    def annotation_files(self, dataset, keys):
        source = self.root / 'annotations'
        annotations = [{'image_id': key, 'id': i, 'action': 'the car stops',
                        'justification': 'the light is red', 'caption': 'the car stops the light is red'}
                       for i, key in enumerate(keys)]
        for suffix in ('', '_des', '_exp'):
            folder = source / (dataset + suffix)
            folder.mkdir(parents=True)
            content = {'images': [{'id': key, 'file_name': key} for key in keys], 'annotations': annotations}
            (folder / 'training_32frames_caption_coco_format.json').write_text(json.dumps(content), encoding='utf-8')
        return source

    def test_28_frames_keep_uniform_32_frame_alignment(self):
        folder = self.frame_folder('clip')
        row = prepare.pack_frames('clip', sorted(folder.glob('*.png')), 32, 16)
        self.assertEqual(len(row), 34)
        self.assertEqual(json.loads(row[1])['width'], 24)
        for actual, expected in zip(row[2:], uniform_indices(28, 32)):
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(actual), dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(int(frame[0, 0, 0]), 20 + int(expected))

    def test_missing_media_is_not_replaced_with_black_frames(self):
        with self.assertRaises(FileNotFoundError):
            prepare.prepare_clip('missing', self.media, None, {}, 32, 16)

    def test_parallel_rows_keep_annotation_order(self):
        keys = ['training_clip_00002', 'training_clip_00001']
        source = self.annotation_files('BDDX', keys)
        for key in keys:
            self.frame_folder(key, 3)
        options = prepare.make_parser().parse_args(['--dataset', 'bddx', '--media-root', str(self.media),
            '--annotations-root', str(source), '--data-root', str(self.root / 'data'),
            '--splits', 'training', '--workers', '2', '--image-size', '16'])
        result = prepare.run(options)
        config = yaml.safe_load(Path(result['yaml_files'][0]).read_text(encoding='utf-8'))
        frame_path = self.root / 'data/BDDX' / config['img']
        rows = frame_path.read_bytes().splitlines()
        self.assertEqual([row.split(b'\t')[0].decode() for row in rows], keys)

    def test_prepared_data_and_lk_are_read_by_real_model_loader(self):
        key = 'training_clip_00000'
        source = self.annotation_files('MMAU', [key])
        self.frame_folder(key)
        data_root = self.root / 'data'
        flow = self.root / 'lk'
        options = prepare.make_parser().parse_args(['--dataset', 'mmau', '--media-root', str(self.media),
            '--annotations-root', str(source), '--data-root', str(data_root), '--flow-cache', str(flow),
            '--splits', 'training', '--workers', '1'])
        prepare.run(options)
        parser = SharedConfigs()
        parser.shared_video_captioning_config(cbs=True, scst=True)
        with mock.patch('sys.argv', ['train.py', '--config', str(prepare.ROOT / 'configs/mmau_train.json')]):
            args = get_custom_args(parser)
        args.data_dir = str(data_root)
        args.flow_cache_dir = str(flow)
        args.device = torch.device('cpu')
        args.num_gpus = 1
        args.distributed = False
        check_arguments(args)
        tokenizer = BertTokenizer.from_pretrained(str(prepare.ROOT / 'models/captioning/bert-base-uncased'))
        dataset = build_dataset(args, args.train_yaml, tokenizer, is_train=True)
        sample_key, batch, meta = dataset[0]
        self.assertEqual(sample_key, key)
        self.assertEqual(tuple(batch[3].shape), (32, 3, 224, 224))
        self.assertTrue(torch.isfinite(batch[3]).all())
        self.assertEqual(tuple(batch[-1].shape), (32, 4))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg is required for video preparation')
    def test_raw_video_timestamp_crop_with_spaces_in_path(self):
        video = self.media / 'source video.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-f', 'lavfi', '-i', 'color=c=red:s=32x24:d=1:r=16',
            '-f', 'lavfi', '-i', 'color=c=blue:s=32x24:d=1:r=16',
            '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[v]', '-map', '[v]', '-c:v', 'mpeg4', str(video)],
            check=True, capture_output=True)
        row = prepare.prepare_clip('clip_00000', self.media, None,
            {'clip_00000': {'video': video.name, 'start': 1, 'end': 2}}, 32, 24)
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(row[2]), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(float(frame[:, :, 0].mean()), float(frame[:, :, 2].mean()) + 100)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required for video preparation')
    def test_point_annotation_repeats_real_last_frame(self):
        video = self.media / 'point source.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                        '-i', 'color=c=blue:s=32x24:d=1:r=16', '-c:v', 'mpeg4', str(video)],
                       check=True, capture_output=True)
        row = prepare.prepare_clip('point', self.media, None,
            {'point': {'video': video.name, 'start': 1, 'end': 1}}, 32, 24)
        self.assertEqual(len(row), 34)
        self.assertTrue(all(frame == row[2] for frame in row[2:]))
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(row[2]), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(float(frame[:, :, 0].mean()), 150)


if __name__ == '__main__':
    unittest.main()
