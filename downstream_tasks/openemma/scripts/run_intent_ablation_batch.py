#!/usr/bin/env python3
"""Run intent-level action-only / explanation-only OpenEMMA ablations.

This script reuses an existing MP4->pose->2fps OpenEMMA batch directory.
It does not rerun DROID or frame sampling. It only runs two additional
OpenEMMA variants:

  1. action annotation only, injected at intent
  2. justification/explanation annotation only, injected at intent

Baseline and action+justification results are read from the source batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


EXPERIMENTS_ROOT = Path("/mnt/ubm_code_nas/gac_liulian/gac_syf/experiments")
OPENEMMA_ROOT = EXPERIMENTS_ROOT / "OpenEMMA"
QWEN_PATH = EXPERIMENTS_ROOT / "qwen2-vl"

DEFAULT_SOURCE_ROOT = Path("/mnt/cpfs_ppu_large/gac_syf/openemma10_action_justification_2fps_ade3_100")
DEFAULT_OUTPUT_ROOT = Path("/mnt/cpfs_ppu_large/gac_syf/openemma10_intent_ablation_action_exp_2fps_ade3_100")


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_jsonl_last(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== {utc_now()} =====\n")
        log.write(format_command(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=str(OPENEMMA_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(OPENEMMA_ROOT) + (os.pathsep + existing if existing else "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def fps_tag(target_fps: float) -> str:
    return f"{target_fps:g}fps".replace(".", "p")


def result_is_valid(output_dir: Path) -> bool:
    result = read_jsonl_last(output_dir / "ade_results.jsonl")
    return result is not None and result.get("avgade") is not None


def metric(result: dict[str, Any] | None, key: str) -> float | None:
    if not result:
        return None
    value = result.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def write_extra_text(path: Path, label: str, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{label}: {safe_text(text)}\n", encoding="utf-8")
    return path


def run_openemma_variant(
    *,
    images_dir: Path,
    pose_file: Path,
    output_dir: Path,
    scene_name: str,
    extra_text_file: Path,
    frame_interval_s: float,
    min_required_future_steps: int,
    plot: bool,
    env: dict[str, str],
    log_path: Path,
) -> dict[str, Any]:
    if result_is_valid(output_dir):
        result = read_jsonl_last(output_dir / "ade_results.jsonl")
        assert result is not None
        return result

    command = [
        sys.executable,
        "main.py",
        "--dataset-type",
        "image_sequence",
        "--dataroot",
        str(images_dir),
        "--pose-file",
        str(pose_file),
        "--image-sequence-name",
        scene_name,
        "--output-dir",
        str(output_dir),
        "--model-path",
        "qwen",
        "--local-model-path",
        str(QWEN_PATH),
        "--image-input-mode",
        "last_frame",
        "--frame-interval-s",
        f"{frame_interval_s:.12f}",
        "--min-required-future-steps",
        str(min_required_future_steps),
        "--extra-scene-text-file",
        str(extra_text_file),
        "--extra-scene-text-placement",
        "intent",
        "--plot",
        "true" if plot else "false",
    ]
    run_command(command, log_path, env)
    result = read_jsonl_last(output_dir / "ade_results.jsonl")
    if result is None:
        raise FileNotFoundError(f"OpenEMMA ADE result missing in {output_dir}")
    return result


def build_row(
    source_row: dict[str, Any],
    output_video_dir: Path,
    baseline: dict[str, Any] | None,
    both: dict[str, Any] | None,
    action_only: dict[str, Any] | None,
    exp_only: dict[str, Any] | None,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "vidName": safe_text(source_row.get("vidName")),
        "video_id": source_row.get("video_id"),
        "id": source_row.get("id"),
        "status": status,
        "error": error or "",
        "output_dir": str(output_video_dir),
        "source_output_dir": safe_text(source_row.get("output_dir")),
        "action": safe_text(source_row.get("action")),
        "justification": safe_text(source_row.get("justification")),
    }

    variants = {
        "baseline": baseline,
        "both": both,
        "action_only": action_only,
        "exp_only": exp_only,
    }
    for key in ("ade_1s", "ade_3s", "ade_5s", "avgade"):
        baseline_value = metric(baseline, key)
        for variant_name, result in variants.items():
            row[f"{variant_name}_{key}"] = metric(result, key)
        for variant_name in ("both", "action_only", "exp_only"):
            variant_value = metric(variants[variant_name], key)
            row[f"{variant_name}_{key}_improvement_vs_baseline"] = (
                baseline_value - variant_value
                if baseline_value is not None and variant_value is not None
                else None
            )
        both_value = metric(both, key)
        for variant_name in ("action_only", "exp_only"):
            variant_value = metric(variants[variant_name], key)
            row[f"{variant_name}_{key}_delta_vs_both"] = (
                variant_value - both_value
                if both_value is not None and variant_value is not None
                else None
            )

    return row


def write_summary(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    rows_sorted = sorted(rows, key=lambda row: row.get("vidName", ""))
    completed = [row for row in rows_sorted if row.get("status") == "completed"]

    aggregate: dict[str, Any] = {
        "updated_at": utc_now(),
        "source_root": str(args.source_root),
        "output_root": str(output_root),
        "config": {
            "target_fps": args.target_fps,
            "frame_interval_s": 1.0 / args.target_fps,
            "min_required_future_steps": args.min_required_future_steps,
            "placement": "intent",
            "plot": args.plot,
            "limit": args.limit,
            "resume": args.resume,
        },
        "counts": {
            "rows": len(rows_sorted),
            "completed": len(completed),
            "failed": sum(row.get("status") == "failed" for row in rows_sorted),
            "source_incomplete": sum(row.get("status") == "source_incomplete" for row in rows_sorted),
        },
        "rows": rows_sorted,
    }

    for metric_name in ("ade_1s", "ade_3s", "ade_5s", "avgade"):
        for variant_name in ("both", "action_only", "exp_only"):
            key = f"{variant_name}_{metric_name}_improvement_vs_baseline"
            values = [row.get(key) for row in completed if isinstance(row.get(key), (float, int))]
            aggregate[f"mean_{key}"] = float(sum(values) / len(values)) if values else None
            aggregate[f"improved_count_{key}"] = sum(float(value) > 0 for value in values)
            aggregate[f"worsened_count_{key}"] = sum(float(value) < 0 for value in values)

    json_dump(output_root / "summary.json", aggregate)

    if rows_sorted:
        fieldnames: list[str] = []
        for row in rows_sorted:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        tmp_csv = output_root / "summary.csv.tmp"
        with tmp_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)
        tmp_csv.replace(output_root / "summary.csv")


def load_source_rows(source_root: Path) -> list[dict[str, Any]]:
    summary_path = source_root / "summary.json"
    payload = read_json(summary_path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Could not find rows in {summary_path}")
    return rows


def process_one(source_row: dict[str, Any], args: argparse.Namespace, env: dict[str, str]) -> dict[str, Any]:
    vid_name = safe_text(source_row.get("vidName"))
    video_id = f"{int(vid_name):06d}"
    tag = fps_tag(args.target_fps)
    source_video_dir = args.source_root / "videos" / video_id
    output_video_dir = args.output_root / "videos" / video_id
    logs_dir = output_video_dir / "logs"
    status_path = output_video_dir / "status.json"

    baseline_dir = source_video_dir / f"openemma_baseline_no_injection_{tag}"
    both_dir = source_video_dir / f"openemma_injected_action_justification_intent_{tag}"
    sampled_dir = source_video_dir / f"droid_xz_height_scaled_stopaware_{tag}"
    images_dir = sampled_dir / "images"
    pose_candidates = sorted(sampled_dir.glob("*_poses.json"))

    baseline = read_jsonl_last(baseline_dir / "ade_results.jsonl")
    both = read_jsonl_last(both_dir / "ade_results.jsonl")
    action_only = read_jsonl_last(output_video_dir / f"openemma_injected_action_only_intent_{tag}" / "ade_results.jsonl")
    exp_only = read_jsonl_last(output_video_dir / f"openemma_injected_exp_only_intent_{tag}" / "ade_results.jsonl")

    if source_row.get("status") != "completed" or not baseline or not both or not images_dir.exists() or not pose_candidates:
        return build_row(
            source_row,
            output_video_dir,
            baseline,
            both,
            action_only,
            exp_only,
            "source_incomplete",
            "Source batch is not completed or required sampled image/pose/baseline/both result is missing.",
        )

    if args.resume and action_only and exp_only:
        return build_row(source_row, output_video_dir, baseline, both, action_only, exp_only, "completed", None)

    output_video_dir.mkdir(parents=True, exist_ok=True)
    json_dump(
        status_path,
        {
            "status": "running",
            "started_at": utc_now(),
            "vidName": vid_name,
            "source_video_dir": str(source_video_dir),
            "target_fps": args.target_fps,
            "min_required_future_steps": args.min_required_future_steps,
        },
    )

    try:
        pose_file = pose_candidates[0]
        scene_name = f"accident_{video_id}_droid_xz_height_scaled_stopaware_{tag}_openemma"
        frame_interval_s = 1.0 / args.target_fps

        action_text_file = write_extra_text(
            output_video_dir / "extra_scene_text_action_only.txt",
            "Action annotation",
            safe_text(source_row.get("action")),
        )
        exp_text_file = write_extra_text(
            output_video_dir / "extra_scene_text_exp_only.txt",
            "Justification annotation",
            safe_text(source_row.get("justification")),
        )

        action_only = run_openemma_variant(
            images_dir=images_dir,
            pose_file=pose_file,
            output_dir=output_video_dir / f"openemma_injected_action_only_intent_{tag}",
            scene_name=scene_name,
            extra_text_file=action_text_file,
            frame_interval_s=frame_interval_s,
            min_required_future_steps=args.min_required_future_steps,
            plot=args.plot,
            env=env,
            log_path=logs_dir / "01_openemma_action_only_intent.log",
        )
        exp_only = run_openemma_variant(
            images_dir=images_dir,
            pose_file=pose_file,
            output_dir=output_video_dir / f"openemma_injected_exp_only_intent_{tag}",
            scene_name=scene_name,
            extra_text_file=exp_text_file,
            frame_interval_s=frame_interval_s,
            min_required_future_steps=args.min_required_future_steps,
            plot=args.plot,
            env=env,
            log_path=logs_dir / "02_openemma_exp_only_intent.log",
        )

        json_dump(
            status_path,
            {
                "status": "completed",
                "completed_at": utc_now(),
                "vidName": vid_name,
                "source_video_dir": str(source_video_dir),
                "baseline": baseline,
                "both": both,
                "action_only": action_only,
                "exp_only": exp_only,
            },
        )
        return build_row(source_row, output_video_dir, baseline, both, action_only, exp_only, "completed", None)
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "error_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        json_dump(
            status_path,
            {
                "status": "failed",
                "failed_at": utc_now(),
                "vidName": vid_name,
                "error": error_text,
                "traceback_log": str(logs_dir / "error_traceback.log"),
                "baseline": baseline,
                "both": both,
                "action_only": action_only,
                "exp_only": exp_only,
            },
        )
        return build_row(source_row, output_video_dir, baseline, both, action_only, exp_only, "failed", error_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--min-required-future-steps", type=int, default=6)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sleep-between-videos-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    os.chdir(OPENEMMA_ROOT)
    rows = load_source_rows(args.source_root)
    if args.limit is not None:
        rows = rows[: args.limit]

    json_dump(
        args.output_root / "manifest.json",
        {
            "created_or_updated_at": utc_now(),
            "script": str(Path(__file__).resolve()),
            "source_root": str(args.source_root),
            "output_root": str(args.output_root),
            "selected_count": len(rows),
            "target_fps": args.target_fps,
            "frame_interval_s": 1.0 / args.target_fps,
            "min_required_future_steps": args.min_required_future_steps,
            "variants": ["baseline", "action_only_intent", "exp_only_intent", "action_justification_intent"],
        },
    )

    env = build_env()
    results: list[dict[str, Any]] = []
    started = time.time()
    for index, source_row in enumerate(rows, start=1):
        vid_name = safe_text(source_row.get("vidName"))
        print(f"[{utc_now()}] ({index}/{len(rows)}) processing vidName={vid_name}", flush=True)
        row = process_one(source_row, args, env)
        results.append(row)
        write_summary(args.output_root, results, args)
        if args.sleep_between_videos_s > 0 and index < len(rows):
            time.sleep(args.sleep_between_videos_s)

    write_summary(args.output_root, results, args)
    print(
        json.dumps(
            {
                "finished_at": utc_now(),
                "elapsed_s": round(time.time() - started, 3),
                "summary_json": str(args.output_root / "summary.json"),
                "summary_csv": str(args.output_root / "summary.csv"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
