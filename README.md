# PepperFall

Fall detection for the Pepper robot that fuses skeleton motion (YOLO11x-pose + ST-GCN) with a frozen vision-language model (Qwen2.5-VL-7B).
CMU 11-785 (Introduction to Deep Learning) course project, midterm stage.

Each video is cut into 1.6 s clips (16 frames at 10 fps, stride 8).
The skeleton branch encodes the main person's 17 COCO joints with ST-GCN; the VLM branch reads four frames of the same clip and returns p(fall), the answer-token hidden state, and pooled video patch tokens.
The two are combined by concatenation or cross-attention, and trained either as a binary fall detector or as a 16-class OmniFall action classifier.
All models are evaluated on datasets they never see during training (GMDCSA24, CAUCAFall), with AUROC as the main metric.

## Repository layout

```
configs/            default hyperparameters (stgcn.yaml, fusion.yaml)
data/preprocess/    pose extraction, deduplication, clip building, VLM feature extraction
data/processed/splits/   FallVision train/val/test video lists (session-grouped split)
models/             ST-GCN and fusion models (concat, cross-attention)
train/              training scripts
eval/               cross-dataset evaluation, shortcut tests, summary tables
logs/*.sh           the run scripts used for the reported experiments
scripts/            FallVision download / unpack
report/             midterm report body (LaTeX) and figure scripts
```

Data, keypoints, extracted features, checkpoints, and results are not tracked.

## Setup

Python 3.10, CUDA 12.4.

```bash
conda create -n fall python=3.10 -y
conda activate fall
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-server.txt
```

Download `yolo11x-pose.pt` from Ultralytics into `weights/`. Qwen2.5-VL-7B-Instruct is downloaded from Hugging Face on first use.

## Data

| Dataset | Role | Location |
|---|---|---|
| FallVision | train / val / test (video labels) | `data/raw/FallVision` (`scripts/download_fallvision.sh`, then `scripts/unpack_fallvision.sh`) |
| OF-Syn (OmniFall) | train / val / test (segment + 16-action labels) | `data/raw/OF-Syn` (videos, `meta/labels`, `meta/splits`) |
| GMDCSA24 | test only | `data/raw/GMDCSA24/videos` |
| CAUCAFall | test only | `data/raw/CAUCAFall/videos` |

OmniFall segment labels for GMDCSA24 and CAUCAFall go in `data/raw/omnifall_labels/{GMDCSA24,caucafall}.csv`.

## Pipeline

All commands run from the repository root. Most scripts accept `--shard/--num-shards/--device` to split work across GPUs, followed by a `--merge-only` call.

**1. FallVision preprocessing**
```bash
python data/preprocess/find_duplicates.py                 # duplicates + recording-session groups
python data/preprocess/extract_pose.py --device 0          # YOLO11x-pose + ByteTrack keypoints
python data/preprocess/build_clips.py --T 16               # clips, Run A / Run B normalization, split
python data/preprocess/vlm_hidden.py --T 16                # VLM hidden state + p(fall)
python data/preprocess/vlm_tokens.py --T 16                # VLM patch tokens (tokmean)
```

**2. OF-Syn and external datasets**
```bash
python data/preprocess/transcode_av1.py --src data/raw/OF-Syn/videos --dst data/raw/OF-Syn/videos_h264
python data/preprocess/extract_pose.py --root data/raw/GMDCSA24/videos --out data/keypoints_ext/GMDCSA24
python data/preprocess/extract_pose.py --root data/raw/OF-Syn/videos_h264 --out data/keypoints_ext/OF-Syn
python data/preprocess/build_external.py --name GMDCSA24  # also CAUCAFall, OF-Syn
python data/preprocess/vlm_tokens.py --ext GMDCSA24 --split test
python data/preprocess/vlm_tokens.py --ext OF-Syn --mean-only
```

**3. Baselines**
```bash
python train/train_stgcn.py --run A --T 16 --seed 0        # ST-GCN on FallVision video labels
python eval/vlm_zero_shot.py --split test                  # zero-shot Qwen2.5-VL
```

**4. Main experiments** (`train/train_event.py`)
```bash
# --mode: stgcn | hidden | tokmean | skel_tokmean (concat) | skel_xattn | skel_xattn_sm (cross-attention)
# --label: event (default) | video | multiclass
python train/train_event.py --train OF-Syn FallVision --mode stgcn --run A --seed 0
python train/train_event.py --train OF-Syn --label video --mode stgcn --run A --seed 0
python train/train_event.py --train OF-Syn --label multiclass --mode skel_tokmean --run A --seed 0
python train/train_event.py --train OF-Syn --filter camera_elevation=high,top   # viewpoint shift
```
The exact runs in the report (three seeds each) are in `logs/run_*.sh`.

**5. Evaluation and tables**
```bash
python eval/shortcut_check.py        # static-posture shortcut tests on FallVision
python eval/eval_external.py --datasets GMDCSA24 CAUCAFall
python eval/summarize_event.py       # all models, same protocol -> results/event_summary.md
python report/make_figs.py           # report figure
```

## Laptop inference (Mac, real time)

`infer/` runs a trained checkpoint on a laptop (Apple Silicon `mps`) with the same preprocessing as training.

```bash
pip install -r requirements-mac.txt
python infer/run.py --ckpt CKPT_DIR --source video.mp4            # offline check
python infer/run.py --ckpt CKPT_DIR --source camera --show        # laptop webcam
python infer/run.py --ckpt CKPT_DIR --source pepper:PEPPER_IP --show
```
`CKPT_DIR` is a run folder with `best.pt` and `metrics.json`. The VLM is chosen from the folder name (`*_q25-3b` → Qwen2.5-VL-3B, `*_q3-2b` → Qwen3-VL-2B, no suffix → Qwen2.5-VL-7B) or set with `--vlm-model`.
For Pepper, copy `infer/pepper_bridge.py` to the robot and run it there (Python 2.7 + NAOqi); it streams the head camera to the laptop and speaks when the laptop raises an alarm.

## Labels

- **video**: every clip of a fall video is positive (FallVision's labeling).
- **event**: a clip is positive if it overlaps an OmniFall fall segment by at least 0.3 s, negative if it does not overlap, and ignored otherwise.
- **multiclass**: the OmniFall action covering most of the clip (16 classes); the alarm score is p(fall), or p(fall) + p(fallen).

## Team

Youjin Kim, Mijin Koo, Yoobin Park
