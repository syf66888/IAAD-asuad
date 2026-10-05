"""Prepare the caption and frame TSV format consumed by the asuad loader.

This consolidates the original caption conversion, FFmpeg frame extraction,
and OpenCV image-to-TSV packing into one configurable entrypoint.
"""
import argparse
import base64
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.datasets.flow_cache import uniform_indices
from src.utils.tsv_file_ops import tsv_writer

ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
SPLITS = ('training', 'validation', 'testing')
_WORKER_CONTEXT = None


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def natural_order(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', path.name)]


def source_frames(root, key):
    directory = root / key
    if directory.is_dir():
        files = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        files = [p for p in root.glob(key + '_frame*') if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=natural_order)


def video_path(root, name):
    path = Path(name)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_file():
        return candidate
    candidates = [candidate.with_suffix(ext) for ext in VIDEO_EXTENSIONS]
    candidates = [p for p in candidates if p.is_file()]
    if len(candidates) > 1:
        raise ValueError('Multiple videos match: ' + str(candidate))
    return candidates[0] if candidates else None


def resolve_video(root, key, media_index):
    # A clip already named by its sample ID needs no second timestamp crop.
    clip = video_path(root, key)
    if clip is not None:
        return clip, None, None
    entry = media_index.get(key)
    if entry is None:
        raise FileNotFoundError('No clip video or frame folder for ' + key + ' under ' + str(root))
    if isinstance(entry, str):
        entry = {'video': entry}
    video = video_path(root, entry['video'])
    if video is None:
        raise FileNotFoundError('Source video is missing: ' + str(root / entry['video']))
    start = float(entry.get('start', 0))
    end = float(entry['end']) if entry.get('end') is not None else None
    if not math.isfinite(start) or start < 0 or (end is not None and (not math.isfinite(end) or end < start)):
        raise ValueError('Invalid clip timestamps for ' + key)
    return video, start, end


def extract_frames(video, directory, num_frames, start=None, end=None):
    if start is not None and end == start:
        # Point annotations supply a single frame, repeated to the input length.
        capture = cv2.VideoCapture(str(video))
        try:
            fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps <= 0 or count < 1:
                raise ValueError('Cannot read video timing: ' + str(video))
            index = min(int(start * fps), int(count) - 1)
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            target = directory / 'frame0001.jpg'
            if not ok or not cv2.imwrite(str(target), image):
                raise ValueError('Cannot read the annotated frame: ' + str(video))
            return [target] * num_frames
        finally:
            capture.release()
    for executable in ('ffmpeg', 'ffprobe'):
        if shutil.which(executable) is None:
            raise RuntimeError('Install FFmpeg before preparing video inputs.')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                            '-of', 'default=noprint_wrappers=1:nokey=1', str(video)],
                           check=True, capture_output=True, text=True)
    video_duration = float(probe.stdout.strip())
    begin = 0.0 if start is None else start
    stop = video_duration if end is None else min(end, video_duration)
    duration = stop - begin
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('The requested clip has no decodable duration: ' + str(video))
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(video)]
    if begin:
        command += ['-ss', str(begin)]
    command += ['-t', str(duration), '-vf', 'fps={:.12g}'.format(num_frames / duration),
                '-frames:v', str(num_frames), str(directory / 'frame%04d.jpg')]
    subprocess.run(command, check=True, capture_output=True)
    frames = sorted(directory.glob('frame*.jpg'))
    if not frames:
        raise ValueError('FFmpeg extracted no frames: ' + str(video))
    # Match the original extractor's last-frame padding for short outputs.
    return (frames + [frames[-1]] * max(0, num_frames - len(frames)))[:num_frames]


def pack_frames(key, paths, num_frames=32, image_size=256):
    if not paths:
        raise ValueError('No source frames for ' + key)
    selected = uniform_indices(len(paths), num_frames)
    binaries = []
    geometry = None
    for index in selected:
        path = paths[int(index)]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Cannot decode frame: ' + str(path))
        height, width = image.shape[:2]
        scale = image_size / min(height, width)
        size = (round(width * scale), round(height * scale))
        if geometry is not None and size != geometry:
            raise ValueError('Frame geometry changes within ' + key)
        geometry = size
        resized = cv2.resize(image, size)
        ok, encoded = cv2.imencode('.jpg', resized)
        if not ok:
            raise ValueError('Cannot encode frame: ' + str(path))
        binaries.append(base64.b64encode(encoded.tobytes()))
    metadata = json.dumps({'class': -1, 'width': geometry[0], 'height': geometry[1]})
    return [key, metadata] + binaries


def prepare_clip(key, media_root, frames_root, media_index, num_frames, image_size):
    cv2.setNumThreads(1)
    frame_paths = source_frames(frames_root or media_root, key)
    if frame_paths:
        return pack_frames(key, frame_paths, num_frames, image_size)
    video, start, end = resolve_video(media_root, key, media_index)
    with tempfile.TemporaryDirectory(prefix='asuad_frames_') as temp:
        frame_paths = extract_frames(video, Path(temp), num_frames, start, end)
        return pack_frames(key, frame_paths, num_frames, image_size)


def worker_init(media_root, frames_root, media_index, image_size):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = dict(media_root=media_root, frames_root=frames_root,
                           media_index=media_index, num_frames=32, image_size=image_size)
    cv2.setNumThreads(1)


def worker_clip(key):
    return prepare_clip(key, **_WORKER_CONTEXT)


def ordered_parallel_rows(pool, keys, buffer_size):
    """Bound queued frame rows while retaining annotation order."""
    iterator = iter(keys)
    pending = deque()
    for _ in range(buffer_size):
        key = next(iterator, None)
        if key is None:
            break
        pending.append(pool.submit(worker_clip, key))
    while pending:
        yield pending.popleft().result()
        key = next(iterator, None)
        if key is not None:
            pending.append(pool.submit(worker_clip, key))
def annotation_rows(content):
    keys = [str(image['id']) for image in content['images']]
    if len(keys) != len(set(keys)):
        raise ValueError('Annotation sample IDs must be unique.')
    captions = defaultdict(list)
    for annotation in content['annotations']:
        key = str(annotation['image_id'])
        action, justification = annotation['action'], annotation['justification']
        if not action.strip() or not justification.strip():
            raise ValueError('Both caption fields are required for ' + key)
        captions[key].append({'action': action, 'justification': justification,
                              'caption': annotation.get('caption', action + ' ' + justification)})
    if set(keys) != set(captions):
        raise ValueError('Image IDs and caption IDs disagree.')
    return keys, captions


def copy_annotations(source_root, target_root, dataset, split):
    name = split + '_32frames_caption_coco_format.json'
    for suffix in ('', '_des', '_exp'):
        source = source_root / (dataset + suffix) / name
        target = target_root / (dataset + suffix) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise FileExistsError('Different annotations already exist: ' + str(target))
            shutil.copy2(source, target)
        if suffix:
            (target.parent / (split + '_32frames.yaml')).write_text(
                yaml.safe_dump({'caption_coco_format': name}), encoding='utf-8')


def prepare_split(options, dataset, split, media_index):
    folder = options.data_root / dataset
    folder.mkdir(parents=True, exist_ok=True)
    annotation_name = split + '_32frames_caption_coco_format.json'
    content = read_json(options.annotations_root / dataset / annotation_name)
    keys, captions = annotation_rows(content)
    frame_name = '{}_32frames_img_size{}.img.tsv'.format(split, options.image_size)
    frame_tsv = folder / 'frame_tsv' / frame_name
    outputs = [frame_tsv, folder / (split + '.caption.tsv'), folder / (split + '.label.tsv'),
               folder / (split + '.caption.linelist.tsv')]
    if not options.overwrite:
        for path in outputs:
            if path.exists():
                raise FileExistsError('Prepared output already exists; use --overwrite to rebuild: ' + str(path))
    copy_annotations(options.annotations_root, options.data_root, dataset, split)
    worker = partial(prepare_clip, media_root=options.media_root, frames_root=options.frames_root,
                     media_index=media_index, num_frames=32, image_size=options.image_size)
    if options.workers > 1:
        with ProcessPoolExecutor(max_workers=options.workers, initializer=worker_init,
                                 initargs=(options.media_root, options.frames_root, media_index, options.image_size)) as pool:
            rows = ordered_parallel_rows(pool, keys, options.workers * 2)
            tsv_writer(tqdm(rows, total=len(keys), desc=dataset + ' ' + split), str(frame_tsv))
    else:
        rows = (worker(key) for key in keys)
        tsv_writer(tqdm(rows, total=len(keys), desc=dataset + ' ' + split), str(frame_tsv))
    for label in ('caption', 'label'):
        tsv_writer(([key, json.dumps(captions[key], ensure_ascii=False)] for key in keys),
                   str(folder / (split + '.' + label + '.tsv')))
    tsv_writer(([row, caption] for row, key in enumerate(keys) for caption in range(len(captions[key]))),
               str(folder / (split + '.caption.linelist.tsv')))
    config = {'preextracted_frames': True, 'img': 'frame_tsv/' + frame_name,
              'label': split + '.label.tsv', 'caption': split + '.caption.tsv',
              'caption_linelist': split + '.caption.linelist.tsv', 'caption_coco_format': annotation_name}
    yaml_path = folder / (split + '_32frames.yaml')
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return yaml_path


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['bddx', 'mmau'], required=True)
    parser.add_argument('--media-root', type=Path, required=True, help='Clip videos or frame folders named by annotation sample ID.')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--annotations-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--frames-root', type=Path, help='Optional separate root of existing frame folders.')
    parser.add_argument('--media-index', type=Path, help='Optional JSON mapping sample IDs to video, start, and end.')
    parser.add_argument('--flow-cache', type=Path, help='Generate the LK cache after creating the frame TSVs.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=list(SPLITS))
    parser.add_argument('--overwrite', action='store_true')
    return parser


def run(options):
    if options.workers < 1 or options.image_size < 1:
        raise ValueError('workers and image-size must be positive.')
    options.data_root = options.data_root.expanduser().resolve()
    options.annotations_root = options.annotations_root.expanduser().resolve()
    options.media_root = options.media_root.expanduser().resolve(strict=True)
    if options.frames_root:
        options.frames_root = options.frames_root.expanduser().resolve(strict=True)
    dataset = options.dataset.upper()
    index_path = options.media_index or options.annotations_root / dataset / 'media_index.json'
    if options.media_index and not index_path.is_file():
        raise FileNotFoundError(index_path)
    media_index = read_json(index_path) if index_path.is_file() else {}
    yaml_paths = [prepare_split(options, dataset, split, media_index) for split in options.splits]
    if options.flow_cache:
        from scripts.precompute_lk_flow import resolve_source, run_source
        flow_options = argparse.Namespace(cache_dir=str(options.flow_cache.expanduser().resolve()),
            num_frames=32, resize_short_side=224, workers=options.workers, force=options.overwrite, limit=0, log_every=100)
        for yaml_path in yaml_paths:
            run_source(flow_options, resolve_source(yaml_path))
    return {'dataset': dataset, 'data_root': str(options.data_root),
            'yaml_files': [str(path) for path in yaml_paths], 'flow_cache': str(options.flow_cache) if options.flow_cache else None}


def main():
    parser = make_parser()
    options = parser.parse_args()
    try:
        result = run(options)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
