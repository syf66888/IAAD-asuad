# IAAD / asuad

**asuad** is a video-based model for driving behavior understanding and accident understanding.

## Introduction

asuad uses visual input only and does not require vehicle control signals such as steering, throttle, or brake inputs.

asuad combines a Video Swin Transformer, a BERT caption decoder, YOLO object references, Lucas–Kanade motion features, and gated feature fusion. It generates two text fields from each video:

- **BDDX:** driving action description (`des`) and its explanation (`exp`).
- **MMAU:** accident description (`des`) and accident prevention advice (`exp`).

## Downloads

**Best model weights and pretrained assets:** [Download](https://pan.baidu.com/s/19Qcz2E9RY6EvcUKDRD0rRQ?pwd=sqqi) (access code: `sqqi`)

**IAAD dataset:** [Download](https://pan.baidu.com/s/1JPB9f7h4fzmXATYWamBn8w?pwd=5h2r) (access code: `5h2r`)

Extract the model files into a separate `weights/` directory:

```text
weights/
├── bddx/
│   ├── model.bin
│   ├── testing_predictions.json
│   ├── metrics.json
│   ├── turn_accuracy.json
│   └── turn_predictions.json
├── mmau/
│   ├── model.bin
│   ├── testing_predictions.json
│   └── metrics.json
└── pretrained/
    ├── swin_base_patch244_window877_kinetics600_22k.pth
    └── yolov5su.pt
```

The BDDX and MMAU files are the corresponding **best models**. Each model directory includes its test predictions and evaluation scores. Tokenizer and model configuration files are included in this repository.

## Environment Setup

The environment configuration is **Linux, NVIDIA RTX 4090 (24 GB), Python 3.8.20, PyTorch 1.13.1, TorchVision 0.14.1, and CUDA 11.6**.

```bash
conda create --name asuad python=3.8.20
conda activate asuad
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
sudo apt-get update
sudo apt-get install default-jre ffmpeg
export CUDA_VISIBLE_DEVICES=0
```

Java is used for caption evaluation, and FFmpeg is used for video preparation.

## Data Preparation

Annotation JSONs are included in [`datasets/`](datasets/). Download the corresponding media and choose `bddx` or `mmau`:

```bash
export DATASET=bddx
export DATA_ROOT="$PWD/datasets"
export MEDIA_ROOT="/path/to/IAAD/$DATASET"
export FLOW_CACHE="/path/to/lk/$DATASET"
export WEIGHTS_ROOT="/path/to/weights"

python scripts/prepare_dataset.py \
  --dataset "$DATASET" --media-root "$MEDIA_ROOT" \
  --data-root "$DATA_ROOT" --flow-cache "$FLOW_CACHE"
```

[`prepare_dataset.py`](scripts/prepare_dataset.py) extracts video frames, creates frame and caption TSVs, and prepares the offline LK features used by the model. It accepts clip videos or frame folders named by the annotation sample ID. For source videos, a `media_index.json` maps each sample to its video and timestamps; the supplied BDDX mapping is read automatically. Use `--media-index /path/to/media_index.json` for another media layout.

## Training

Train on either dataset:

```bash
python train.py --dataset "$DATASET" \
  --data-root "$DATA_ROOT" --flow-cache "$FLOW_CACHE" \
  --weights-root "$WEIGHTS_ROOT" --output-dir "outputs/$DATASET/ce"
```

Use `--config /path/to/config.json` to load custom settings.

After training, **SCST is an optional step for either dataset** that can be used to try to further improve caption generation:

```bash
python scripts/build_reward_cache.py --dataset "$DATASET" \
  --data-root "$DATA_ROOT" --cache "outputs/$DATASET/reward_cache.json.gz"

python train_scst.py --dataset "$DATASET" \
  --data-root "$DATA_ROOT" --flow-cache "$FLOW_CACHE" \
  --weights-root "$WEIGHTS_ROOT" --ce-output-dir "outputs/$DATASET/ce" \
  --reward-cache "outputs/$DATASET/reward_cache.json.gz" \
  --output-dir "outputs/$DATASET/scst"
```

## Inference and Evaluation

Run inference with the best model for the selected dataset:

```bash
python inference.py --dataset "$DATASET" \
  --data-root "$DATA_ROOT" --flow-cache "$FLOW_CACHE" \
  --weights-root "$WEIGHTS_ROOT" --output-dir "outputs/$DATASET/evaluation"
```

The command generates both captions and evaluates BLEU, CIDEr, and ROUGE-L. Predictions and evaluation scores are saved under the specified output directory. To use your own trained model, append `--checkpoint /path/to/model.bin`.

### BDDX Turning Accuracy

BDDX evaluation also reports **turning accuracy**: the proportion of annotated explicit left/right turns whose generated description (`des`) correctly identifies the action and direction. A matching turn or steering description is counted as correct; an omitted turn, opposite direction, or lane change is counted as incorrect. Reference descriptions of lane changes, lane positions, planned turns, and U-turns are excluded from this subset. The metric uses action/description text only.

The evaluation writes `*.turn_accuracy.json` with the overall and per-direction scores, and `*.turn_predictions.json` with the reference, generated description, labels, and outcome for every sample. The same evaluation runs during BDDX training and optional SCST.

To evaluate saved descriptions directly, without loading model weights or videos:

```bash
python scripts/evaluate_bddx_turns.py \
  --predictions results/bddx/testing_predictions.json \
  --output-dir results/bddx
```

The script accepts `testing_predictions.json`, description COCO JSONs, and prediction TSVs. It uses the supplied BDDX test action annotations by default; pass `--references /path/to/annotations.json` to evaluate another annotation file.

## Best Models

Caption scores below are multiplied by 100; B4 denotes BLEU-4.

| Dataset | des B4 | des CIDEr | exp B4 | exp CIDEr |
|---|---:|---:|---:|---:|
| BDDX | 36.10 | 259.34 | 10.89 | 105.24 |
| MMAU | 30.44 | 207.90 | 39.53 | 278.62 |

BDDX turning accuracy for the best model:

| Left turn | Right turn | Overall |
|---:|---:|---:|
| 88.61% (70/79) | 84.85% (84/99) | **86.52% (154/178)** |

The scores and per-sample decisions are available in [`turn_accuracy.json`](results/bddx/turn_accuracy.json) and [`turn_predictions.json`](results/bddx/turn_predictions.json).

Complete test predictions are available in [`results/bddx/`](results/bddx/) and [`results/mmau/`](results/mmau/), with the same output files included beside each downloaded model. `testing_predictions.json` contains `image_id`, `description`, and `explanation` for each video.

## Code Layout

```text
train.py                    # CE training
train_scst.py               # Optional SCST training
inference.py                # Caption generation and evaluation
scripts/prepare_dataset.py  # Video, caption, and LK feature preparation
scripts/evaluate_bddx_turns.py # Turning accuracy from saved BDDX descriptions
configs/                    # Training and inference settings
datasets/                   # Dataset annotations
results/                    # Best model predictions and scores
src/                        # Model, data loader, and evaluation implementation
```

## TODO

- [ ] Release the Tiny model.
- [ ] Release the IAAD dataset annotations.
- [ ] Upload the preprocessed datasets.

## Acknowledgments

This implementation builds on the following open-source projects:

- [ADAPT](https://github.com/jxbbb/ADAPT)
- [SwinBERT](https://github.com/microsoft/SwinBERT)
- [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)
- [BDD-X Dataset](https://github.com/JinkyuKimUCB/BDD-X-dataset)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [COCO Caption Evaluation](https://github.com/tylin/coco-caption)
- [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

See [`LICENSE`](LICENSE) and the retained third-party notices for licensing details.
