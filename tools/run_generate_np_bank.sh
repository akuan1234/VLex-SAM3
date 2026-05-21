#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT_PREFIX="${DATA_ROOT_PREFIX:-$REPO_ROOT/data}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints/InternVL2_5}"
BACKEND="${BACKEND:-internvl}"
DEVICE="${DEVICE:-cuda}"
GENERATOR_SCRIPT="${GENERATOR_SCRIPT:-$REPO_ROOT/tools/generate_visual_np_prompt_bank.py}"

SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-6}"
PHRASES_PER_SAMPLE="${PHRASES_PER_SAMPLE:-6}"
MIN_PROMPTS="${MIN_PROMPTS:-3}"
MAX_PROMPTS="${MAX_PROMPTS:-5}"

usage() {
  cat <<EOF
Usage:
  bash tools/run_generate_np_bank.sh <dataset> [extra generator args...]

Datasets:
  loveda | potsdam | vaihingen | vdd | uavid | udd5 | openearthmap | isaid

Environment overrides:
  DATA_ROOT_PREFIX=/path/to/data
  MODEL_PATH=/path/to/InternVL2_5
  PYTHON_BIN=python
  PHRASES_PER_SAMPLE=6

Note:
  The NP generation and selection implementation is not included in this review
  release. This command template will become runnable when
  tools/generate_visual_np_prompt_bank.py is released after paper acceptance.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

DATASET_KEY="${1,,}"
shift
EXTRA_ARGS=("$@")

COMMON_ARGS=(
  "$GENERATOR_SCRIPT"
  --backend "$BACKEND"
  --model-path "$MODEL_PATH"
  --samples-per-class "$SAMPLES_PER_CLASS"
  --phrases-per-sample "$PHRASES_PER_SAMPLE"
  --min-prompts "$MIN_PROMPTS"
  --max-prompts "$MAX_PROMPTS"
  --device "$DEVICE"
)

case "$DATASET_KEY" in
  loveda)
    DATASET_ARGS=(
      --dataset-profile loveda
      --seed-prompts "$REPO_ROOT/configs/cls_loveda.txt"
      --config "$REPO_ROOT/configs/cfg_loveda.py"
      --data-root-override "$DATA_ROOT_PREFIX/LoveDA"
      --dataset-split train
      --label-values 1,2,3,4,5,6,7
      --ignore-label 0
      --image-suffix .png
      --ann-suffix .png
      --output-json "$REPO_ROOT/configs/generated/cls_loveda_internvl_np.json"
      --output-txt "$REPO_ROOT/configs/generated/cls_loveda_internvl_np.txt"
    )
    ;;
  potsdam)
    DATASET_ARGS=(
      --dataset-profile potsdam
      --seed-prompts "$REPO_ROOT/configs/cls_potsdam.txt"
      --config "$REPO_ROOT/configs/cfg_potsdam.py"
      --data-root-override "$DATA_ROOT_PREFIX/Potsdam"
      --dataset-split train
      --label-values 0,1,2,3,4,5
      --ignore-label 255
      --image-suffix .png
      --ann-suffix .png
      --output-json "$REPO_ROOT/configs/generated/cls_potsdam_internvl_np.json"
      --output-txt "$REPO_ROOT/configs/generated/cls_potsdam_internvl_np.txt"
    )
    ;;
  vaihingen)
    DATASET_ARGS=(
      --dataset-profile vaihingen
      --seed-prompts "$REPO_ROOT/configs/cls_vaihingen.txt"
      --config "$REPO_ROOT/configs/cfg_vaihingen.py"
      --data-root-override "$DATA_ROOT_PREFIX/Vaihingen"
      --dataset-split train
      --label-values 0,1,2,3,4,5
      --ignore-label 255
      --image-suffix .png
      --ann-suffix .png
      --output-json "$REPO_ROOT/configs/generated/cls_vaihingen_internvl_np.json"
      --output-txt "$REPO_ROOT/configs/generated/cls_vaihingen_internvl_np.txt"
    )
    ;;
  vdd)
    DATASET_ARGS=(
      --dataset-profile vdd
      --seed-prompts "$REPO_ROOT/configs/cls_vdd.txt"
      --image-dir "$DATA_ROOT_PREFIX/VDD/train/src"
      --ann-dir "$DATA_ROOT_PREFIX/VDD/train/gt"
      --label-values 0,1,2,3,4,5,6
      --ignore-label 255
      --image-suffix .JPG
      --ann-suffix .png
      --output-json "$REPO_ROOT/configs/generated/cls_vdd_internvl_np.json"
      --output-txt "$REPO_ROOT/configs/generated/cls_vdd_internvl_np.txt"
    )
    ;;
  uavid)
    DATASET_ARGS=(
      --dataset-profile uavid
      --seed-prompts "$REPO_ROOT/configs/cls_uavid.txt"
      --image-dir "$DATA_ROOT_PREFIX/UAVid/img_dir/train"
      --ann-dir "$DATA_ROOT_PREFIX/UAVid/ann_dir/train"
      --label-values 0,1,2,3,4,5,6
      --ignore-label 255
      --image-suffix .png
      --ann-suffix .png
      --internvl-max-tiles 8
      --output-json "$REPO_ROOT/configs/generated/cls_uavid_internvl_np.json"
      --output-txt "$REPO_ROOT/configs/generated/cls_uavid_internvl_np.txt"
    )
    ;;
  udd5)
    DATASET_ARGS=(
      --dataset-profile udd5
      --seed-prompts "$REPO_ROOT/configs/cls_udd5.txt"
      --image-dir "$DATA_ROOT_PREFIX/UDD5/train/src"
      --ann-dir "$DATA_ROOT_PREFIX/UDD5/train/gt_label"
      --label-values 0,1,2,3,4
      --ignore-label 255
      --image-suffix .JPG
      --ann-suffix .png
      --output-json "$REPO_ROOT/configs/generated/cls_udd5_internvl_np.json"
      --output-txt "$REPO_ROOT/configs/generated/cls_udd5_internvl_np.txt"
    )
    ;;
  openearthmap)
    DATASET_ARGS=(
      --dataset-profile openearthmap
      --seed-prompts "$REPO_ROOT/configs/cls_openearthmap.txt"
      --image-dir "$DATA_ROOT_PREFIX/OpenEarthMap/train/images"
      --ann-dir "$DATA_ROOT_PREFIX/OpenEarthMap/train/labels"
      --label-values 0,1,2,3,4,5,6,7,8
      --ignore-label 255
      --image-suffix .tif
      --ann-suffix .tif
      --output-json "$REPO_ROOT/configs/generated/cls_openearthmap_internvl_np.json"
      --output-txt "$REPO_ROOT/configs/generated/cls_openearthmap_internvl_np.txt"
    )
    ;;
  isaid)
    DATASET_ARGS=(
      --dataset-profile isaid
      --seed-prompts "$REPO_ROOT/configs/cls_iSAID.txt"
      --config "$REPO_ROOT/configs/cfg_iSAID.py"
      --data-root-override "$DATA_ROOT_PREFIX/iSAID"
      --dataset-split train
      --label-values 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
      --ignore-label 255
      --image-suffix .png
      --ann-suffix .png
      --ann-stem-suffix _instance_color_RGB
      --max-scan-images 10000
      --shuffle-scan
      --stop-when-filled
      --min-scan-images 3000
      --candidate-pool-per-class 24
      --output-json "$REPO_ROOT/configs/generated/cls_iSAID_internvl_np.json"
      --output-txt "$REPO_ROOT/configs/generated/cls_iSAID_internvl_np.txt"
    )
    ;;
  *)
    echo "Unknown dataset: $DATASET_KEY" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ ! -f "$GENERATOR_SCRIPT" ]]; then
  echo "NP generation code is not included in this review release." >&2
  echo "Expected generator: $GENERATOR_SCRIPT" >&2
  echo "It will be open-sourced after the paper is accepted." >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/configs/generated"
echo "[VLex-SAM3] dataset=$DATASET_KEY backend=$BACKEND phrases_per_crop=$PHRASES_PER_SAMPLE"
"$PYTHON_BIN" "${COMMON_ARGS[@]}" "${DATASET_ARGS[@]}" "${EXTRA_ARGS[@]}"
