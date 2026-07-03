#!/usr/bin/env bash
set -Eeuo pipefail

EXP_ROOT="/mnt/cpfs_ppu_large/gac_syf/openemma10_qwen3_caption_n82_compare"
OUT_ROOT="${EXP_ROOT}/qwen3_format_finetune_uploaded_plus_n82_alltrain_val_excl_n82_ep5"
MANIFEST_DIR="${OUT_ROOT}/manifests"
TRAIN_OUT="${OUT_ROOT}/outputs/qwen3vl4b_mmau_best_openemma_uploaded_plus_n82_alltrain_val_excl_n82_full_lr2e6_ep5"
PRED_DIR="${OUT_ROOT}/predictions/best_test_n82_beam1"
LOG_DIR="${OUT_ROOT}/logs"
STATUS_FILE="${OUT_ROOT}/pipeline_status.json"

QWEN3_PY="/mnt/ubm_code_nas/gac_liulian/gac_syf/qwen3_vl_mmau/.venv_qwen3vl_ppu/bin/python"
QWEN3_CODE="/mnt/ubm_code_nas/gac_liulian/gac_syf/qwen3_vl_bddx"
OPENEMMA_ROOT="/mnt/ubm_code_nas/gac_liulian/gac_syf/experiments/OpenEMMA"
OPENEMMA_PY="${OPENEMMA_ROOT}/.venv-ppu/bin/python"
CAPTION_INIT_CKPT="/mnt/cpfs_ppu_large/gac_syf/qwen3_vl_mmau/outputs/qwen3vl4b_mmau_full_32f_action_explain/best"
OPENEMMA_QWEN="/mnt/ubm_code_nas/gac_liulian/gac_syf/experiments/qwen2-vl"

INSTRUCTION="You are given a short ego-view driving video clip. Respond with exactly two lines in this format: Action: <one sentence describing the ego-car action and accident event>. Explain: <one sentence beginning with safety advice or rationale for the ego car>. Match the wording style of the training captions; do not output label-bank fragments."

mkdir -p "${LOG_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

write_status() {
  local stage="$1"
  local detail="$2"
  "${OPENEMMA_PY}" - "$STATUS_FILE" "$stage" "$detail" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "stage": sys.argv[2],
    "detail": sys.argv[3],
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

trap 'rc=$?; line=${BASH_LINENO[0]:-$LINENO}; echo "[$(timestamp)] pipeline failed rc=${rc} line=${line}"; write_status "failed" "exit_code=${rc} line=${line}"; exit "${rc}"' ERR

echo "[$(timestamp)] pipeline start"
echo "OUT_ROOT=${OUT_ROOT}"
write_status "started" "pipeline launched"

echo "[$(timestamp)] checking python environments"
"${QWEN3_PY}" - <<'PY'
from transformers import Qwen3VLForConditionalGeneration
import torch
print("Qwen3VL import OK")
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY

echo "[$(timestamp)] checking manifests"
"${OPENEMMA_PY}" - <<PY
import json
from pathlib import Path

manifest_dir = Path("${MANIFEST_DIR}")
expected = {"train.jsonl": 165, "val.jsonl": 10, "test.jsonl": 82}
for name, count in expected.items():
    path = manifest_dir / name
    if not path.exists():
        raise SystemExit(f"missing manifest: {path}")
    actual = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    print(name, actual)
    if actual != count:
        raise SystemExit(f"{name}: expected {count}, got {actual}")
meta = json.loads((manifest_dir / "manifest_metadata.json").read_text(encoding="utf-8"))
print(json.dumps(meta, ensure_ascii=False, indent=2))
PY

write_status "training" "Qwen caption fine-tuning 5 epochs"
echo "[$(timestamp)] train Qwen caption model"
PYTHONUNBUFFERED=1 \
PYTHONPATH="${QWEN3_CODE}:${PYTHONPATH:-}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${QWEN3_PY}" "${QWEN3_CODE}/train.py" \
  --model-name-or-path "${CAPTION_INIT_CKPT}" \
  --train-manifest "${MANIFEST_DIR}/train.jsonl" \
  --val-manifest "${MANIFEST_DIR}/val.jsonl" \
  --output-dir "${TRAIN_OUT}" \
  --train-mode full \
  --target-format action_explain \
  --instruction "${INSTRUCTION}" \
  --input-mode video \
  --num-train-epochs 5 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-6 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --max-grad-norm 0.5 \
  --action-loss-weight 1.4 \
  --reason-loss-weight 1.0 \
  --max-video-frames 32 \
  --train-num-workers 2 \
  --eval-num-workers 1 \
  --mixed-precision bf16 \
  --model-dtype bfloat16 \
  --attn-implementation sdpa \
  --gradient-checkpointing \
  --seed 42 \
  --logging-steps 2 \
  --selection-metric split_score \
  --val-gen-max-samples 10 \
  --val-gen-max-new-tokens 192 \
  --val-gen-num-beams 1 \
  --val-gen-attn-implementation eager

write_status "caption_eval" "generating captions on full N=82 with best checkpoint"
echo "[$(timestamp)] evaluate/generate N=82 captions"
PYTHONUNBUFFERED=1 \
PYTHONPATH="${QWEN3_CODE}:${PYTHONPATH:-}" \
"${QWEN3_PY}" "${QWEN3_CODE}/eval.py" \
  --checkpoint-dir "${TRAIN_OUT}/best" \
  --manifest "${MANIFEST_DIR}/test.jsonl" \
  --output-dir "${PRED_DIR}" \
  --target-format action_explain \
  --instruction "${INSTRUCTION}" \
  --per-device-batch-size 1 \
  --num-workers 1 \
  --max-video-frames 32 \
  --mixed-precision bf16 \
  --model-dtype bfloat16 \
  --attn-implementation eager \
  --max-new-tokens 192 \
  --num-beams 1

write_status "openemma" "running OpenEMMA caption injection on full N=82"
echo "[$(timestamp)] run OpenEMMA qwen caption injection"
"${OPENEMMA_PY}" "${OPENEMMA_ROOT}/scripts/caption_injection_experiment.py" run \
  --manifest "${EXP_ROOT}/manifest.json" \
  --predictions-jsonl "${PRED_DIR}/predictions.jsonl" \
  --output-root "${OUT_ROOT}" \
  --variant-name qwen \
  --openemma-qwen-path "${OPENEMMA_QWEN}" \
  --resume

write_status "summarize" "summarizing metrics"
echo "[$(timestamp)] summarize"
"${OPENEMMA_PY}" "${OPENEMMA_ROOT}/scripts/caption_injection_experiment.py" summarize \
  --manifest "${EXP_ROOT}/manifest.json" \
  --output-root "${OUT_ROOT}" \
  --qwen-predictions-jsonl "${PRED_DIR}/predictions.jsonl"

write_status "table" "writing final table"
echo "[$(timestamp)] write final table"
"${OPENEMMA_PY}" - <<'PY'
import csv
import json
from pathlib import Path

root = Path("/mnt/cpfs_ppu_large/gac_syf/openemma10_qwen3_caption_n82_compare/qwen3_format_finetune_uploaded_plus_n82_alltrain_val_excl_n82_ep5")
agg_path = root / "tables/aggregate_caption_injection_metrics.csv"
failure_path = root / "tables/failure_rates_l2_gt10m.csv"
out_md = root / "tables/main_table_baseline_gt_qwen_alltrain_ep5_n82.md"
out_csv = root / "tables/main_table_baseline_gt_qwen_alltrain_ep5_n82.csv"
caption_metrics_path = root / "predictions/best_test_n82_beam1/metrics.json"
train_summary_path = root / "outputs/qwen3vl4b_mmau_best_openemma_uploaded_plus_n82_alltrain_val_excl_n82_full_lr2e6_ep5/training_summary.json"

metric_labels = {
    "ade_1s": "ADE@1s",
    "ade_2s": "ADE@2s",
    "ade_3s": "ADE@3s",
    "l2_1s": "L2@1s",
    "l2_2s": "L2@2s",
    "l2_3s": "L2@3s",
}
stat_labels = {"mean": "Mean", "median": "Median", "trimmed_mean_10pct": "10% trimmed mean"}

rows = {}
with agg_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        rows[(row["statistic"], row["metric"])] = row

def fnum(value):
    if value is None or value == "":
        return ""
    return f"{float(value):.2f}"

def fpct(value):
    if value is None or value == "":
        return ""
    value = float(value)
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"

csv_fields = ["metric"]
for stat in ["mean", "median", "trimmed_mean_10pct"]:
    for col in ["baseline", "gt", "gt_imp_pct", "qwen_alltrain_ep5", "qwen_alltrain_ep5_imp_pct"]:
        csv_fields.append(f"{stat}_{col}")

main_rows = []
for metric, label in metric_labels.items():
    out = {"metric": label}
    for stat in ["mean", "median", "trimmed_mean_10pct"]:
        row = rows[(stat, metric)]
        out[f"{stat}_baseline"] = float(row["baseline"])
        out[f"{stat}_gt"] = float(row["gt"])
        out[f"{stat}_gt_imp_pct"] = float(row["gt_improvement_pct_vs_baseline"])
        out[f"{stat}_qwen_alltrain_ep5"] = float(row["qwen"])
        out[f"{stat}_qwen_alltrain_ep5_imp_pct"] = float(row["qwen_improvement_pct_vs_baseline"])
    main_rows.append(out)

with out_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(main_rows)

failure_by_horizon = {}
with failure_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        failure_by_horizon.setdefault(row["horizon"], {})[row["variant"]] = row

def fail_fmt(row):
    if not row or not row.get("window_count") or float(row["window_count"]) == 0:
        return ""
    return f"{int(float(row['failure_count']))}/{int(float(row['window_count']))} ({float(row['failure_rate_pct']):.2f}%)"

def fail_delta(row):
    if not row or row.get("failure_rate_delta_pp_vs_baseline", "") == "":
        return ""
    value = float(row["failure_rate_delta_pp_vs_baseline"])
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f} pp"

train = json.loads(train_summary_path.read_text(encoding="utf-8"))
caption = json.loads(caption_metrics_path.read_text(encoding="utf-8"))

lines = [
    "# N=82 OpenEMMA Caption Injection Results",
    "",
    "Setting: all 82 N82 samples are added to caption training; validation/eval for checkpoint selection keeps the original uploaded non-N82 val split (10 samples), with N82 excluded. OpenEMMA evaluation below is on full N=82, so this is train-seen evaluation for N82.",
    "",
    f"Caption fine-tune best: epoch {train.get('best_epoch')}, selection_metric={train.get('selection_metric')}, best_selection_value={train.get('best_selection_value'):.6f}, val_loss={train.get('best_val_loss'):.6f}.",
    f"N=82 caption metrics: CIDEr={caption.get('CIDEr'):.6f}, split_score={caption.get('split_score'):.6f}, action_CIDEr={caption.get('action_CIDEr'):.6f}, explain_CIDEr={caption.get('explain_CIDEr'):.6f}.",
    "",
]

for stat in ["mean", "median", "trimmed_mean_10pct"]:
    lines.append(f"## {stat_labels[stat]}")
    lines.append("")
    lines.append("| Metric | Baseline ↓ | GT ↓ | GT Imp. ↑ | Qwen-alltrain-ep5 ↓ | Qwen Imp. ↑ |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in main_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["metric"],
                    fnum(row[f"{stat}_baseline"]),
                    fnum(row[f"{stat}_gt"]),
                    fpct(row[f"{stat}_gt_imp_pct"]),
                    fnum(row[f"{stat}_qwen_alltrain_ep5"]),
                    fpct(row[f"{stat}_qwen_alltrain_ep5_imp_pct"]),
                ]
            )
            + " |"
        )
    lines.append("")

lines.extend(
    [
        "## Failure Rate (L2 > 10m)",
        "",
        "| Horizon | Baseline | GT | Qwen-alltrain-ep5 | Qwen Δ vs Baseline |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
)
for horizon in ["1s", "2s", "3s"]:
    row = failure_by_horizon[horizon]
    lines.append(
        "| "
        + " | ".join(
            [
                horizon,
                fail_fmt(row["baseline"]),
                fail_fmt(row["gt"]),
                fail_fmt(row["qwen"]),
                fail_delta(row["qwen"]),
            ]
        )
        + " |"
    )

lines.extend(
    [
        "",
        "Source CSV files:",
        f"- aggregate: `{agg_path}`",
        f"- failure: `{failure_path}`",
        f"- per-video: `{root / 'tables/per_video_caption_injection_metrics.csv'}`",
    ]
)
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_md)
print(out_csv)
PY

write_status "completed" "pipeline completed successfully"
echo "[$(timestamp)] pipeline completed"
