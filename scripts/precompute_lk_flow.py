#!/usr/bin/env python3
"""CPU-only, resumable sparse LK cache generation for explicit BDDX split YAMLs.

Example:
 python scripts/precompute_lk_flow.py --yaml datasets/BDDX/training_32frames.yaml \
   datasets/BDDX/testing_32frames.yaml --cache-dir datasets/BDDX/lk_flow_cache --workers 8

All source TSV rows are processed, so one TSV shared by multiple split YAMLs
is computed only once. Existing files are skipped only after metadata, full
source-row SHA256, shape, sampling, dtype and finite-value validation.
"""
import argparse
import base64
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.flow_cache import uniform_indices
from src.datasets.lk_flow_cache import SparseLKFlowCache, compute_sparse_lk

_CACHE = None
_SOURCE = None
_OFFSETS = None
_FORCE = False


def resolve_source(yaml_path):
    yaml_path = Path(yaml_path).resolve(strict=True)
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if config.get("composite", False):
        raise ValueError("Composite TSV preprocessing is not supported; pass a concrete split YAML")
    name = config.get("img")
    if not isinstance(name, str):
        raise ValueError("Split YAML must define img: <frame TSV>")
    source = Path(name)
    if not source.is_file():
        source = yaml_path.parent / source
    if source.suffix != ".tsv" or not source.is_file():
        raise FileNotFoundError("Split YAML img must resolve to a frame TSV: " + str(source))
    return source.resolve()


def worker_init(cache_dir, source_path, num_frames, resize_short_side, force):
    global _CACHE, _SOURCE, _OFFSETS, _FORCE
    # Multiprocess workers do no torch/CUDA work. Avoid nested OpenCV pools.
    cv2.setNumThreads(1)
    _CACHE = SparseLKFlowCache(cache_dir, source_path, num_frames, resize_short_side)
    _SOURCE = open(source_path, "rb")
    with Path(source_path).with_suffix(".lineidx").open() as handle:
        _OFFSETS = [int(line.strip()) for line in handle if line.strip()]
    _FORCE = force


def process_row(row_index):
    _SOURCE.seek(_OFFSETS[row_index])
    raw = _SOURCE.readline()
    row_hash = hashlib.sha256(raw).hexdigest()
    columns = raw.rstrip(b"\r\n").split(b"\t")
    if len(columns) < 3:
        raise ValueError("Row {} has no pre-extracted frame".format(row_index))
    key = columns[0].decode("utf-8")
    if not _FORCE and _CACHE.path(row_index).exists():
        try:
            _CACHE.load(row_index, expected_key=key, row_sha256=row_hash)
            return row_index, "skipped", _CACHE.path(row_index).stat().st_size
        except (RuntimeError, ValueError, KeyError):
            pass  # A corrupt/stale partial result is recomputed atomically.
    encoded_frames = columns[2:]
    selected = uniform_indices(len(encoded_frames), _CACHE.num_frames)
    frames, source_size = [], None
    for index in selected:
        encoded = np.frombuffer(base64.b64decode(encoded_frames[index], validate=True), dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Cannot decode clip {}, row {}, frame {}".format(key, row_index, index))
        shape = tuple(frame.shape[:2])
        if source_size is None:
            source_size = shape
        elif shape != source_size:
            raise ValueError("Frame dimensions change within clip {}".format(key))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tracks = compute_sparse_lk(np.stack(frames), _CACHE.resize_short_side)
    _CACHE.write(row_index, tracks, key, source_size, len(encoded_frames), selected, row_hash)
    return row_index, "written", _CACHE.path(row_index).stat().st_size


def run_source(args, source):
    cache = SparseLKFlowCache(args.cache_dir, source, args.num_frames, args.resize_short_side, create=True)
    total = int(cache.manifest["source_rows"])
    limit = min(total, args.limit) if args.limit else total
    count = skipped = written = stored_bytes = 0
    started = time.monotonic()
    cache.update_manifest(0, "running", requested_rows=limit)
    print(json.dumps({"event": "start", "source": str(source), "rows": limit, "workers": args.workers,
                      "manifest": str(cache.manifest_path), "identity": cache.identity}), flush=True)
    context = mp.get_context("spawn")
    pool = None
    try:
        if args.workers == 1:
            worker_init(args.cache_dir, str(source), args.num_frames, args.resize_short_side, args.force)
            results = map(process_row, range(limit))
        else:
            pool = context.Pool(args.workers, initializer=worker_init,
                                initargs=(args.cache_dir, str(source), args.num_frames, args.resize_short_side, args.force))
            results = pool.imap_unordered(process_row, range(limit), chunksize=4)
        for row, status, size in results:
            count += 1
            skipped += status == "skipped"
            written += status == "written"
            stored_bytes += size
            if count % args.log_every == 0 or count == limit:
                elapsed = time.monotonic() - started
                cache.update_manifest(count, "running", requested_rows=limit, bytes=stored_bytes,
                                      skipped_rows=skipped, written_rows=written, elapsed_seconds=round(elapsed, 3))
                print(json.dumps({"event": "progress", "source": source.name, "completed": count, "total": limit,
                                  "written": written, "skipped": skipped, "elapsed_seconds": round(elapsed, 2),
                                  "eta_seconds": round(elapsed / count * (limit - count), 2)}), flush=True)
        if pool is not None:
            pool.close()
            pool.join()
        state = "complete" if count == total else "partial"
        cache.update_manifest(count, state, requested_rows=limit, bytes=stored_bytes, skipped_rows=skipped,
                              written_rows=written, elapsed_seconds=round(time.monotonic() - started, 3))
        print(json.dumps({"event": "done", "source": str(source), "status": state, "rows": count,
                          "bytes": stored_bytes, "manifest": str(cache.manifest_path)}), flush=True)
    except BaseException as exc:
        if pool is not None:
            pool.terminate()
            pool.join()
        cache.update_manifest(count, "failed", requested_rows=limit, last_error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yaml", nargs="+", required=True, help="Explicit split YAML paths")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--resize-short-side", type=int, default=224,
                        help="Must match the unchanged RGB Resize short side / img_res")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Small validation subset per source; 0 means all")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Recompute valid records too")
    args = parser.parse_args()
    if min(args.workers, args.num_frames, args.log_every, args.resize_short_side) < 1 or args.limit < 0:
        parser.error("workers/num-frames/log-every/resize-short-side must be positive; limit must be nonnegative")
    sources = list(dict.fromkeys(resolve_source(path) for path in args.yaml))
    for source in sources:
        run_source(args, source)


if __name__ == "__main__":
    main()
