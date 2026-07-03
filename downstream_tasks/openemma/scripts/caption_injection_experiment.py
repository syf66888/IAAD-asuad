#!/usr/bin/env python3
"""Run OpenEMMA intent-injection experiments from generated captions.

Subcommands:
  prepare  Build a small OpenEMMA video manifest and InternVL-style records.
  generate Generate action/explanation text with qwen or miniinternvl.
  run      Inject generated text into OpenEMMA and run planning.
  summarize Compare baseline, GT injection, and generated-caption injection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable


EXPERIMENTS_ROOT = Path("/mnt/ubm_code_nas/gac_liulian/gac_syf/experiments")
OPENEMMA_ROOT = EXPERIMENTS_ROOT / "OpenEMMA"
DEFAULT_SOURCE_ROOT = Path("/mnt/cpfs_ppu_large/gac_syf/openemma10_action_justification_2fps_ade3_100")
DEFAULT_QWEN_PATH = EXPERIMENTS_ROOT / "qwen2-vl"
DEFAULT_MINI_CHECKPOINT = Path(
    "/mnt/cpfs_ppu_large/gac_syf/mini_internvl2_bddx/outputs/"
    "Mini-InternVL2-4B-DA-BDD-mmau-action-exp-8frames-maxpatch4-llm-lora-r16-epoch4-merged-20260609"
)

BAD_OR_EXCLUDED_VIDEOS = {
    "10316",
    "02015",
    "06430",
    "06417",
    "09957",
    "08621",
    "05000",
    "01788",
    "10009",
    "01712",
    "05294",
    "04295",
    "02219",
    "02001",
    "05243",
    "04449",
    "02048",
    "02505",
}

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN
HORIZONS = {"1s": 2, "2s": 4, "3s": 6}

ACTION_PROMPT = (
    "The images are sampled from an accident-avoidance driving video in temporal order. "
    "Describe only the ego vehicle action or risky driving situation in one concise sentence."
)
EXPLAIN_PROMPT = (
    "The images are sampled from an accident-avoidance driving video in temporal order. "
    "Provide the safe driving advice or avoidance guidance for this situation in one concise paragraph."
)
QWEN3_MMAU_ACTION_EXPLAIN_INSTRUCTION = (
    "You are given a short ego-view driving video clip. Respond with exactly two lines. "
    "Action: summarize the key driving event and maneuver labels concisely. "
    "Explain: state the safety hazard, cause, or driving advice grounded in the video. "
    "Preserve concrete cues such as lane direction, intersections, braking distance, speed, "
    "pedestrians, collisions, loss of control, and surrounding vehicles when visible."
)


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def image_files(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"):
        paths.extend(image_dir.glob(pattern))
    return sorted(set(paths), key=natural_key)


def uniform_sample(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    if count <= 1:
        return [paths[len(paths) // 2]]
    indices = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
    return [paths[idx] for idx in indices]


def video_id_from_vid_name(vid_name: Any) -> str:
    return f"{int(str(vid_name)):06d}"


def vid_name_5(video_id: str) -> str:
    return f"{int(video_id):05d}"


def fps_tag(target_fps: float) -> str:
    return f"{target_fps:g}fps".replace(".", "p")


def load_source_rows(source_root: Path) -> list[dict[str, Any]]:
    payload = read_json(source_root / "summary.json")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"summary.json rows missing in {source_root}")
    return rows


def result_valid(path: Path) -> bool:
    row = read_jsonl_last(path)
    return bool(row and row.get("avgade") is not None)


def select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_root = Path(args.source_root)
    requested = {video_id_from_vid_name(item) for item in args.video_ids} if args.video_ids else None
    selected = []
    for row in load_source_rows(source_root):
        if row.get("status") != "completed":
            continue
        video_id = video_id_from_vid_name(row.get("vidName"))
        if requested and video_id not in requested:
            continue
        if args.exclude_default_bad and vid_name_5(video_id) in BAD_OR_EXCLUDED_VIDEOS:
            continue
        tag = fps_tag(args.target_fps)
        video_dir = source_root / "videos" / video_id
        sampled_dir = video_dir / f"droid_xz_height_scaled_stopaware_{tag}"
        baseline_result = video_dir / f"openemma_baseline_no_injection_{tag}" / "ade_results.jsonl"
        gt_result = video_dir / f"openemma_injected_action_justification_intent_{tag}" / "ade_results.jsonl"
        pose_files = sorted(sampled_dir.glob("*_poses.json"))
        if sampled_dir.joinpath("images").exists() and pose_files and result_valid(baseline_result) and result_valid(gt_result):
            selected.append(row)
        if args.limit is not None and len(selected) >= args.limit:
            break
    return selected


def internvl_prompt(num_frames: int, task: str) -> str:
    prefix = "\n".join(f"Frame-{idx + 1}: <image>" for idx in range(num_frames))
    prompt = ACTION_PROMPT if task == "action" else EXPLAIN_PROMPT
    return f"{prefix}\n{prompt}"


def qwen_prompt(task: str) -> str:
    return ACTION_PROMPT if task == "action" else EXPLAIN_PROMPT


def prediction_action(pred: dict[str, Any] | None) -> str:
    if not pred:
        return ""
    return safe_text(
        pred.get("action")
        or pred.get("prediction_action")
        or pred.get("pred_action")
        or pred.get("generated_action")
    )


def prediction_explanation(pred: dict[str, Any] | None) -> str:
    if not pred:
        return ""
    return safe_text(
        pred.get("exp")
        or pred.get("explanation")
        or pred.get("prediction_explain")
        or pred.get("prediction_justification")
        or pred.get("generated_explanation")
    )


def prepare(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    data_dir = output_root / "caption_inputs"
    tag = fps_tag(args.target_fps)
    rows = select_rows(args)
    if not rows:
        raise RuntimeError("No eligible videos selected.")

    manifest_rows = []
    action_records = []
    explain_records = []
    for row in rows:
        video_id = video_id_from_vid_name(row.get("vidName"))
        video_dir = source_root / "videos" / video_id
        sampled_dir = video_dir / f"droid_xz_height_scaled_stopaware_{tag}"
        images_dir = sampled_dir / "images"
        pose_file = sorted(sampled_dir.glob("*_poses.json"))[0]
        frames = uniform_sample(image_files(images_dir), args.num_frames)
        sample_id = f"openemma_{video_id}"
        action_gt = safe_text(row.get("action"))
        explain_gt = safe_text(row.get("justification"))
        image_list = [str(path.resolve()) for path in frames]
        base_record = {
            "id": f"{sample_id}_action",
            "image": image_list,
            "width_list": [],
            "height_list": [],
            "source": {
                "dataset": "openemma_cleaned",
                "video_id": video_id,
                "vidName": vid_name_5(video_id),
                "source_video_dir": str(video_dir),
                "sampled_dir": str(sampled_dir),
                "pose_file": str(pose_file),
                "selected_frames": image_list,
            },
        }
        action_record = dict(base_record)
        action_record["id"] = f"{sample_id}_action"
        action_record["conversations"] = [
            {"from": "human", "value": internvl_prompt(len(image_list), "action")},
            {"from": "gpt", "value": action_gt},
        ]
        explain_record = dict(base_record)
        explain_record["id"] = f"{sample_id}_explain"
        explain_record["conversations"] = [
            {"from": "human", "value": internvl_prompt(len(image_list), "exp")},
            {"from": "gpt", "value": explain_gt},
        ]
        action_records.append(action_record)
        explain_records.append(explain_record)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "video_id": video_id,
                "vidName": vid_name_5(video_id),
                "source_video_dir": str(video_dir),
                "sampled_dir": str(sampled_dir),
                "images_dir": str(images_dir),
                "pose_file": str(pose_file),
                "baseline_dir": str(video_dir / f"openemma_baseline_no_injection_{tag}"),
                "gt_injected_dir": str(video_dir / f"openemma_injected_action_justification_intent_{tag}"),
                "gt_action": action_gt,
                "gt_explanation": explain_gt,
                "caption_image_paths": image_list,
            }
        )

    write_jsonl(data_dir / "annotations" / "action" / f"{args.split}.jsonl", action_records)
    write_jsonl(data_dir / "annotations" / "explain" / f"{args.split}.jsonl", explain_records)
    write_json(
        output_root / "manifest.json",
        {
            "created_at": utc_now(),
            "source_root": str(source_root),
            "data_dir": str(data_dir),
            "split": args.split,
            "target_fps": args.target_fps,
            "num_caption_frames": args.num_frames,
            "selected_count": len(manifest_rows),
            "rows": manifest_rows,
        },
    )
    print(json.dumps({"manifest": str(output_root / "manifest.json"), "selected_count": len(manifest_rows)}, indent=2))


def prepare_qwen3_manifest(args: argparse.Namespace) -> None:
    manifest = load_manifest(Path(args.manifest))
    rows_out = []
    for index, row in enumerate(manifest["rows"]):
        source_video_dir = Path(row["source_video_dir"])
        frame_dir = source_video_dir / "frames_30fps_1280"
        all_frames = image_files(frame_dir)
        if not all_frames:
            all_frames = [Path(path) for path in row.get("caption_image_paths", [])]
        if not all_frames:
            raise RuntimeError(f"No frames found for {row['sample_id']}")
        selected_frames = uniform_sample(all_frames, args.num_frames)
        duration = max((len(all_frames) - 1) / args.source_fps, 1e-3) if len(all_frames) > 1 else 1.0
        action = safe_text(row.get("gt_action"))
        explanation = safe_text(row.get("gt_explanation"))
        rows_out.append(
            {
                "sample_id": row["sample_id"],
                "dataset": "openemma_cleaned",
                "split": args.split,
                "row_index": index,
                "global_index": index,
                "video_name": row["vidName"],
                "video_frame_paths": [str(path.resolve()) for path in selected_frames],
                "caption": f"Action: {action}\nExplain: {explanation}",
                "action": action,
                "justification": explanation,
                "start_time": 0.0,
                "end_time": duration,
                "duration": duration,
                "num_input_frames": len(selected_frames),
                "source_video_dir": str(source_video_dir),
            }
        )
    output_jsonl = Path(args.output_jsonl).resolve()
    write_jsonl(output_jsonl, rows_out)
    write_json(
        output_jsonl.with_suffix(".meta.json"),
        {
            "created_at": utc_now(),
            "source_manifest": str(Path(args.manifest).resolve()),
            "num_rows": len(rows_out),
            "num_frames": args.num_frames,
            "source_fps": args.source_fps,
            "instruction": QWEN3_MMAU_ACTION_EXPLAIN_INSTRUCTION,
        },
    )
    print(json.dumps({"qwen3_manifest": str(output_jsonl), "rows": len(rows_out)}, indent=2))


def clean_prediction_text(text: Any) -> str:
    return safe_text(str(text or "").strip(" \t\r\n\"'"))


def sample_id_from_record_id(record_id: str) -> str:
    for suffix in ("_action", "_explain"):
        if record_id.endswith(suffix):
            return record_id[: -len(suffix)]
    return record_id


def read_caption_records(data_dir: Path, split: str, max_samples: int | None) -> list[dict[str, Any]]:
    path = data_dir / "annotations" / "action" / f"{split}.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def generate_qwen(args: argparse.Namespace, records: list[dict[str, Any]], out_path: Path) -> None:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor_kwargs = {}
    if args.max_pixels:
        processor_kwargs["max_pixels"] = args.max_pixels
    if args.min_pixels:
        processor_kwargs["min_pixels"] = args.min_pixels
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)

    def ask(paths: list[str], prompt: str) -> str:
        content = [{"type": "image", "image": path} for path in paths]
        content.append({"type": "text", "text": prompt})
        message = [{"role": "user", "content": content}]
        text_prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(next(model.parameters()).device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=bool(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
            )
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as writer:
        for index, record in enumerate(records, start=1):
            sample_id = sample_id_from_record_id(str(record["id"]))
            paths = [str(path) for path in record["image"]]
            print(f"[{utc_now()}] qwen {index}/{len(records)} {sample_id}", flush=True)
            action = clean_prediction_text(ask(paths, qwen_prompt("action")))
            explanation = clean_prediction_text(ask(paths, qwen_prompt("exp")))
            writer.write(json.dumps({"image_id": sample_id, "action": action, "exp": explanation}, ensure_ascii=False) + "\n")
            writer.flush()


def generate_miniinternvl(args: argparse.Namespace, records: list[dict[str, Any]], out_path: Path) -> None:
    import torch
    from PIL import Image

    internvl_chat_dir = EXPERIMENTS_ROOT / "InternVL" / "internvl_chat"
    if str(internvl_chat_dir) not in sys.path:
        sys.path.insert(0, str(internvl_chat_dir))
    from internvl.model import load_model_and_tokenizer
    from internvl.train.dataset import build_transform, dynamic_preprocess

    model_args = SimpleNamespace(
        checkpoint=str(args.checkpoint),
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        auto=False,
    )
    model, tokenizer = load_model_and_tokenizer(model_args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    transform = build_transform(is_train=False, input_size=image_size)
    generation_config = {
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.temperature > 0),
        "temperature": args.temperature,
    }

    def load_pixels(paths: list[str]) -> tuple[torch.Tensor, list[int]]:
        pixel_values = []
        num_patches_list = []
        max_num = max(1, args.max_dynamic_patches // max(1, len(paths)))
        for path in paths:
            image = Image.open(path).convert("RGB")
            tiles = dynamic_preprocess(
                image,
                min_num=1,
                max_num=max_num,
                image_size=image_size,
                use_thumbnail=False,
            )
            tensor = torch.stack([transform(tile) for tile in tiles])
            pixel_values.append(tensor)
            num_patches_list.append(tensor.size(0))
        return torch.cat(pixel_values, dim=0), num_patches_list

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as writer:
        for index, record in enumerate(records, start=1):
            sample_id = sample_id_from_record_id(str(record["id"]))
            paths = [str(path) for path in record["image"]]
            print(f"[{utc_now()}] miniinternvl {index}/{len(records)} {sample_id}", flush=True)
            pixel_values, num_patches_list = load_pixels(paths)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            action = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=internvl_prompt(len(paths), "action"),
                generation_config=generation_config.copy(),
                num_patches_list=num_patches_list,
            )
            explanation = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=internvl_prompt(len(paths), "exp"),
                generation_config=generation_config.copy(),
                num_patches_list=num_patches_list,
            )
            writer.write(
                json.dumps(
                    {"image_id": sample_id, "action": clean_prediction_text(action), "exp": clean_prediction_text(explanation)},
                    ensure_ascii=False,
                )
                + "\n"
            )
            writer.flush()


def generate(args: argparse.Namespace) -> None:
    out_path = Path(args.output_jsonl).resolve()
    if out_path.exists() and not args.force:
        print(json.dumps({"predictions": str(out_path), "status": "exists"}, indent=2))
        return
    records = read_caption_records(Path(args.data_dir).resolve(), args.split, args.max_samples)
    if args.backend == "qwen":
        generate_qwen(args, records, out_path)
    elif args.backend == "miniinternvl":
        generate_miniinternvl(args, records, out_path)
    else:
        raise ValueError(args.backend)
    print(json.dumps({"predictions": str(out_path), "num_samples": len(records)}, indent=2))


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(OPENEMMA_ROOT) + (os.pathsep + existing if existing else "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== {utc_now()} =====\n")
        log.write(format_command(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=str(OPENEMMA_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    mapping = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("image_id") or row.get("sample_id") or "")
        if sample_id:
            mapping[sample_id] = row
    return mapping


def load_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if "rows" not in payload:
        raise ValueError(f"Manifest rows missing: {path}")
    return payload


def run_injection(args: argparse.Namespace) -> None:
    manifest = load_manifest(Path(args.manifest))
    predictions = prediction_map(Path(args.predictions_jsonl))
    output_root = Path(args.output_root).resolve()
    variant_root = output_root / "variants" / args.variant_name
    env = build_env()
    tag = fps_tag(args.target_fps)
    rows_out = []
    rows = manifest["rows"]
    if args.limit is not None:
        rows = rows[: args.limit]

    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        pred = predictions.get(sample_id)
        video_id = row["video_id"]
        video_out = variant_root / "videos" / video_id
        output_dir = video_out / f"openemma_injected_{args.variant_name}_intent_{tag}"
        result = read_jsonl_last(output_dir / "ade_results.jsonl")
        if result and result.get("avgade") is not None and args.resume:
            status = "completed"
            error = ""
        else:
            status = "completed"
            error = ""
            if not pred:
                status = "missing_prediction"
                error = f"No prediction for {sample_id}"
            else:
                extra_text = (
                    f"Action annotation: {prediction_action(pred)}\n"
                    f"Justification annotation: {prediction_explanation(pred)}\n"
                )
                extra_path = video_out / f"extra_scene_text_{args.variant_name}.txt"
                extra_path.parent.mkdir(parents=True, exist_ok=True)
                extra_path.write_text(extra_text, encoding="utf-8")
                scene_name = f"accident_{video_id}_droid_xz_height_scaled_stopaware_{tag}_openemma"
                command = [
                    sys.executable,
                    "main.py",
                    "--dataset-type",
                    "image_sequence",
                    "--dataroot",
                    row["images_dir"],
                    "--pose-file",
                    row["pose_file"],
                    "--image-sequence-name",
                    scene_name,
                    "--output-dir",
                    str(output_dir),
                    "--model-path",
                    "qwen",
                    "--local-model-path",
                    str(args.openemma_qwen_path),
                    "--image-input-mode",
                    "last_frame",
                    "--frame-interval-s",
                    f"{1.0 / args.target_fps:.12f}",
                    "--min-required-future-steps",
                    str(args.min_required_future_steps),
                    "--extra-scene-text-file",
                    str(extra_path),
                    "--extra-scene-text-placement",
                    "intent",
                    "--plot",
                    "true" if args.plot else "false",
                ]
                try:
                    print(f"[{utc_now()}] run {args.variant_name} {index}/{len(rows)} video={video_id}", flush=True)
                    run_command(command, video_out / "logs" / f"openemma_{args.variant_name}.log", env)
                    result = read_jsonl_last(output_dir / "ade_results.jsonl")
                    if not result:
                        status = "failed"
                        error = "Missing ade_results.jsonl after run"
                except Exception as exc:
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    (video_out / "logs").mkdir(parents=True, exist_ok=True)
                    (video_out / "logs" / "error_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        rows_out.append(
            {
                "sample_id": sample_id,
                "video_id": video_id,
                "vidName": row["vidName"],
                "variant": args.variant_name,
                "status": status,
                "error": error,
                "output_dir": str(output_dir),
                "prediction_action": prediction_action(pred),
                "prediction_explanation": prediction_explanation(pred),
            }
        )
        write_json(variant_root / "run_summary.json", {"updated_at": utc_now(), "rows": rows_out})
        if args.sleep_between_videos_s > 0 and index < len(rows):
            time.sleep(args.sleep_between_videos_s)
    print(json.dumps({"summary": str(variant_root / "run_summary.json"), "rows": len(rows_out)}, indent=2))


def load_pose_positions(pose_file: Path) -> list[list[float]]:
    payload = read_json(pose_file)
    records = payload.get("poses", payload) if isinstance(payload, dict) else payload
    positions = []
    for record in records:
        matrix = record.get("matrix") if isinstance(record, dict) else record
        positions.append([float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])])
    return positions


def finite_mean(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(sum(values) / len(values)) if values else None


def median(values: list[float]) -> float | None:
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def trimmed_mean(values: list[float], proportion: float = 0.1) -> float | None:
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not values:
        return None
    trim = int(len(values) * proportion)
    if trim > 0 and len(values) > 2 * trim:
        values = values[trim:-trim]
    return float(sum(values) / len(values))


def metric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_window_index(path: Path) -> int | None:
    match = re.search(r"_(\d+)_pred_traj\.npy$", path.name)
    return int(match.group(1)) if match else None


def compute_variant_metrics(output_dir: Path, pose_file: Path) -> tuple[dict[str, float | None], dict[str, Any]]:
    import numpy as np

    positions = np.array(load_pose_positions(pose_file), dtype=float)
    window_metrics: dict[str, list[float]] = {f"ade_{horizon}": [] for horizon in HORIZONS}
    window_metrics.update({f"l2_{horizon}": [] for horizon in HORIZONS})
    failure_counts = {f"failure_l2_{horizon}_count": 0 for horizon in HORIZONS}
    window_count = 0
    for pred_path in sorted(output_dir.glob("*_pred_traj.npy"), key=natural_key):
        index = parse_window_index(pred_path)
        if index is None:
            continue
        pred = np.load(pred_path)
        future = positions[index + OBS_LEN:min(index + TTL_LEN, len(positions))]
        if len(future) == 0:
            continue
        errors = np.linalg.norm(future[: len(pred), :2] - pred[: len(future), :2], axis=1)
        window_count += 1
        for horizon, steps in HORIZONS.items():
            if len(errors) >= steps:
                ade = float(np.mean(errors[:steps]))
                l2 = float(errors[steps - 1])
                window_metrics[f"ade_{horizon}"].append(ade)
                window_metrics[f"l2_{horizon}"].append(l2)
                if l2 > 10.0:
                    failure_counts[f"failure_l2_{horizon}_count"] += 1
    metrics = {key: finite_mean(values) for key, values in window_metrics.items()}
    metrics["num_pred_windows"] = float(window_count)
    metrics["avg_ade_1s2s3s"] = finite_mean([metrics.get("ade_1s"), metrics.get("ade_2s"), metrics.get("ade_3s")])
    extra = {"window_metrics": window_metrics, **failure_counts}
    return metrics, extra


def flatten_result_metrics(result: dict[str, Any] | None) -> dict[str, float | None]:
    if not result:
        return {}
    return {
        "main_ade_1s": metric_value(result, "ade_1s"),
        "main_ade_3s": metric_value(result, "ade_3s"),
        "main_ade_5s": metric_value(result, "ade_5s"),
        "main_avgade": metric_value(result, "avgade"),
    }


@dataclass
class VariantSpec:
    name: str
    label: str
    kind: str
    root: Path | None = None


def summarize(args: argparse.Namespace) -> None:
    manifest = load_manifest(Path(args.manifest))
    output_root = Path(args.output_root).resolve()
    tag = fps_tag(args.target_fps)
    variant_specs = [
        VariantSpec("baseline", "baseline", "source"),
        VariantSpec("gt", "gt_action_exp", "source"),
        VariantSpec("miniinternvl", "miniinternvl_action_exp", "generated", output_root / "variants" / "miniinternvl"),
        VariantSpec("qwen", "qwen_action_exp", "generated", output_root / "variants" / "qwen"),
    ]
    predictions_by_variant = {}
    for name, path in (("miniinternvl", args.mini_predictions_jsonl), ("qwen", args.qwen_predictions_jsonl)):
        predictions_by_variant[name] = prediction_map(Path(path)) if path else {}

    per_video_rows = []
    failure_totals: dict[str, dict[str, float]] = {}
    for row in manifest["rows"]:
        video_id = row["video_id"]
        sample_id = row["sample_id"]
        pose_file = Path(row["pose_file"])
        out_row: dict[str, Any] = {
            "sample_id": sample_id,
            "video_id": video_id,
            "vidName": row["vidName"],
            "gt_action": row.get("gt_action", ""),
            "gt_explanation": row.get("gt_explanation", ""),
        }
        for pred_variant in ("miniinternvl", "qwen"):
            pred = predictions_by_variant.get(pred_variant, {}).get(sample_id, {})
            out_row[f"{pred_variant}_action"] = prediction_action(pred)
            out_row[f"{pred_variant}_explanation"] = prediction_explanation(pred)

        for spec in variant_specs:
            if spec.name == "baseline":
                output_dir = Path(row["baseline_dir"])
            elif spec.name == "gt":
                output_dir = Path(row["gt_injected_dir"])
            else:
                assert spec.root is not None
                output_dir = spec.root / "videos" / video_id / f"openemma_injected_{spec.name}_intent_{tag}"
            result = read_jsonl_last(output_dir / "ade_results.jsonl")
            metrics, extra = compute_variant_metrics(output_dir, pose_file)
            out_row[f"{spec.name}_status"] = "completed" if result else "missing"
            out_row[f"{spec.name}_output_dir"] = str(output_dir)
            for key, value in flatten_result_metrics(result).items():
                out_row[f"{spec.name}_{key}"] = value
            for key, value in metrics.items():
                out_row[f"{spec.name}_{key}"] = value
            failure_totals.setdefault(spec.name, {"windows": 0.0})
            failure_totals[spec.name]["windows"] += metrics.get("num_pred_windows") or 0.0
            for horizon in HORIZONS:
                failure_totals[spec.name].setdefault(f"failure_l2_{horizon}_count", 0.0)
                failure_totals[spec.name][f"failure_l2_{horizon}_count"] += extra.get(f"failure_l2_{horizon}_count", 0)
        for spec in variant_specs:
            if spec.name == "baseline":
                continue
            for metric_name in ["avg_ade_1s2s3s", "ade_1s", "ade_2s", "ade_3s", "l2_1s", "l2_2s", "l2_3s"]:
                baseline = out_row.get(f"baseline_{metric_name}")
                value = out_row.get(f"{spec.name}_{metric_name}")
                if isinstance(baseline, (float, int)) and baseline != 0 and isinstance(value, (float, int)):
                    out_row[f"{spec.name}_{metric_name}_improvement_pct_vs_baseline"] = (baseline - value) / baseline * 100.0
                else:
                    out_row[f"{spec.name}_{metric_name}_improvement_pct_vs_baseline"] = None
        per_video_rows.append(out_row)

    summary_dir = output_root / "tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    per_video_csv = summary_dir / "per_video_caption_injection_metrics.csv"
    fieldnames: list[str] = []
    for row in per_video_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with per_video_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_video_rows)

    metrics = ["avg_ade_1s2s3s", "ade_1s", "ade_2s", "ade_3s", "l2_1s", "l2_2s", "l2_3s"]
    aggregate_rows = []
    for stat_name, stat_fn in (("mean", finite_mean), ("median", median), ("trimmed_mean_10pct", trimmed_mean)):
        for metric_name in metrics:
            agg: dict[str, Any] = {"statistic": stat_name, "metric": metric_name}
            baseline_value = stat_fn([row.get(f"baseline_{metric_name}") for row in per_video_rows])
            agg["baseline"] = baseline_value
            for spec in variant_specs[1:]:
                value = stat_fn([row.get(f"{spec.name}_{metric_name}") for row in per_video_rows])
                agg[spec.name] = value
                agg[f"{spec.name}_improvement_pct_vs_baseline"] = (
                    (baseline_value - value) / baseline_value * 100.0
                    if baseline_value not in (None, 0) and value is not None
                    else None
                )
            gt_improve = agg.get("gt_improvement_pct_vs_baseline")
            for name in ("miniinternvl", "qwen"):
                improve = agg.get(f"{name}_improvement_pct_vs_baseline")
                agg[f"{name}_improvement_gap_vs_gt_pp"] = improve - gt_improve if improve is not None and gt_improve is not None else None
            aggregate_rows.append(agg)

    aggregate_csv = summary_dir / "aggregate_caption_injection_metrics.csv"
    with aggregate_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(aggregate_rows[0].keys()) if aggregate_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    failure_rows = []
    baseline_failures = failure_totals.get("baseline", {})
    for horizon in HORIZONS:
        baseline_rate = (
            baseline_failures.get(f"failure_l2_{horizon}_count", 0.0) / baseline_failures.get("windows", 1.0) * 100.0
            if baseline_failures.get("windows")
            else None
        )
        for spec in variant_specs:
            totals = failure_totals.get(spec.name, {})
            windows = totals.get("windows", 0.0)
            count = totals.get(f"failure_l2_{horizon}_count", 0.0)
            rate = count / windows * 100.0 if windows else None
            failure_rows.append(
                {
                    "horizon": horizon,
                    "variant": spec.name,
                    "failure_threshold_m": 10.0,
                    "failure_count": count,
                    "window_count": windows,
                    "failure_rate_pct": rate,
                    "failure_rate_delta_pp_vs_baseline": rate - baseline_rate if rate is not None and baseline_rate is not None else None,
                }
            )
    failure_csv = summary_dir / "failure_rates_l2_gt10m.csv"
    with failure_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(failure_rows[0].keys()) if failure_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_rows)

    write_json(
        summary_dir / "summary_paths.json",
        {
            "updated_at": utc_now(),
            "per_video_csv": str(per_video_csv),
            "aggregate_csv": str(aggregate_csv),
            "failure_csv": str(failure_csv),
            "num_videos": len(per_video_rows),
        },
    )
    print(
        json.dumps(
            {
                "per_video_csv": str(per_video_csv),
                "aggregate_csv": str(aggregate_csv),
                "failure_csv": str(failure_csv),
                "num_videos": len(per_video_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--target-fps", type=float, default=2.0)
    prepare_parser.add_argument("--split", default="testing")
    prepare_parser.add_argument("--num-frames", type=int, default=8)
    prepare_parser.add_argument("--limit", type=int, default=8)
    prepare_parser.add_argument("--video-ids", nargs="*", default=None)
    prepare_parser.add_argument("--exclude-default-bad", action=argparse.BooleanOptionalAction, default=True)
    prepare_parser.set_defaults(func=prepare)

    qwen3_manifest_parser = subparsers.add_parser("prepare-qwen3-manifest")
    qwen3_manifest_parser.add_argument("--manifest", type=Path, required=True)
    qwen3_manifest_parser.add_argument("--output-jsonl", type=Path, required=True)
    qwen3_manifest_parser.add_argument("--num-frames", type=int, default=32)
    qwen3_manifest_parser.add_argument("--source-fps", type=float, default=30.0)
    qwen3_manifest_parser.add_argument("--split", default="testing")
    qwen3_manifest_parser.set_defaults(func=prepare_qwen3_manifest)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--backend", choices=["qwen", "miniinternvl"], required=True)
    generate_parser.add_argument("--data-dir", type=Path, required=True)
    generate_parser.add_argument("--split", default="testing")
    generate_parser.add_argument("--output-jsonl", type=Path, required=True)
    generate_parser.add_argument("--max-samples", type=int, default=None)
    generate_parser.add_argument("--force", action="store_true")
    generate_parser.add_argument("--max-new-tokens", type=int, default=128)
    generate_parser.add_argument("--temperature", type=float, default=0.0)
    generate_parser.add_argument("--model-path", type=Path, default=DEFAULT_QWEN_PATH)
    generate_parser.add_argument("--min-pixels", type=int, default=None)
    generate_parser.add_argument("--max-pixels", type=int, default=512 * 512)
    generate_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MINI_CHECKPOINT)
    generate_parser.add_argument("--max-dynamic-patches", type=int, default=4)
    generate_parser.add_argument("--num-beams", type=int, default=1)
    generate_parser.add_argument("--load-in-8bit", action="store_true")
    generate_parser.add_argument("--load-in-4bit", action="store_true")
    generate_parser.set_defaults(func=generate)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--predictions-jsonl", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--variant-name", choices=["miniinternvl", "qwen"], required=True)
    run_parser.add_argument("--target-fps", type=float, default=2.0)
    run_parser.add_argument("--openemma-qwen-path", type=Path, default=DEFAULT_QWEN_PATH)
    run_parser.add_argument("--min-required-future-steps", type=int, default=6)
    run_parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--sleep-between-videos-s", type=float, default=0.5)
    run_parser.set_defaults(func=run_injection)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--manifest", type=Path, required=True)
    summarize_parser.add_argument("--output-root", type=Path, required=True)
    summarize_parser.add_argument("--target-fps", type=float, default=2.0)
    summarize_parser.add_argument("--mini-predictions-jsonl", type=Path, default=None)
    summarize_parser.add_argument("--qwen-predictions-jsonl", type=Path, default=None)
    summarize_parser.set_defaults(func=summarize)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
