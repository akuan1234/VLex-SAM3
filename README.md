# VLex-SAM3

Official code release for **VLex-SAM3: Visual Lexicalization for Training-Free Open-Vocabulary Remote Sensing Segmentation**.

VLex-SAM3 keeps SAM3 frozen and improves open-vocabulary remote-sensing segmentation from the text side. It builds a class-wise lexical expert bank from support image crops and uses frozen multi-source SAM3 responses during inference.

## Highlights

- Training-free OVSS: no model fine-tuning and no additional visual backbone.
- Data-grounded visual lexicalization: mask-highlighted support crops are described by a local MLLM as short noun phrases.
- Lexical expert banks: class-semantic profile selection keeps compact, visually grounded prompts for each class.
- Lexical OR inference: responses from multiple noun phrases in the same class are aggregated by class-wise max.
- FMRE: frozen SAM3 semantic, instance, and presence responses are consolidated with uncertainty-guided feature calibration.

<img width="1450" height="933" alt="image" src="https://github.com/user-attachments/assets/40401932-70dc-42ac-b363-6912fab5739b" />


## Repository Layout

```text
configs/                    Dataset configs and released lexical expert banks
configs/prompt_banks/        Final VLex-SAM3 prompt banks used in the paper
sam3/                        Frozen SAM3 image-model code
tools/run_generate_np_bank.sh Command template for lexical-bank generation
vlex_sam3_segmentor.py       MMSegmentation segmentor wrapper
custom_datasets.py           Dataset registrations for the eight benchmarks
eval.py                      MMSegmentation evaluation entry
demo.py                      Single-image prediction helper
```

SAM3 checkpoints, MLLM weights, and datasets are not included. The dataset can be found at [SegEarth-OV](https://github.com/likyoo/SegEarth-OV).

## Installation

Create an environment and install the dependencies:

```bash
conda create -n vlex-sam3 python=3.10 -y
conda activate vlex-sam3
pip install -r requirements.txt
```

For CUDA environments, install the PyTorch wheel matching your driver before running `pip install -r requirements.txt` if needed. If `mmcv` fails to build from pip, install the matching prebuilt wheel with OpenMMLab MIM.

Place the frozen SAM3 image checkpoint at `weights/sam3/sam3.pt`, or override it from the command line with `model.sam3_checkpoint=/path/to/sam3.pt`.

## Data Layout

The configs expect the following default roots:

```text
data/OpenEarthMap
data/LoveDA
data/iSAID
data/Potsdam
data/Vaihingen
data/UAVid
data/UDD5
data/VDD
```

You can override any root without editing files:

```bash
python eval.py configs/cfg_loveda.py \
  --cfg-options test_dataloader.dataset.data_root=/path/to/LoveDA
```

Expected evaluation subfolders follow the configs:

| Dataset | Images | Masks |
| --- | --- | --- |
| OpenEarthMap | `img_dir/val` | `ann_dir/val` |
| LoveDA | `img_dir/val` | `ann_dir/val` |
| iSAID | `img_dir/val` | `ann_dir/val` |
| Potsdam | `img_dir/val` | `ann_dir/val` |
| Vaihingen | `img_dir/val` | `ann_dir/val` |
| UAVid | `img_dir/test` | `ann_dir/test` |
| UDD5 | `val/src` | `val/gt` |
| VDD | `test/src` | `test/gt` |

## Evaluation

Run one benchmark:

```bash
python eval.py configs/cfg_loveda.py
```

Run distributed evaluation:

```bash
bash dist_test.sh configs/cfg_loveda.py
```

The released configs already point to the final prompt banks in `configs/prompt_banks/` and use the final `prob_thd` values.

## Reported Results

| Dataset | mIoU |
| --- | ---: |
| OpenEarthMap | 44.29 |
| LoveDA | 48.52 |
| iSAID | 37.78 |
| Potsdam | 58.47 |
| Vaihingen | 63.87 |
| UAVid | 60.61 |
| UDD5 | 74.41 |
| VDD | 71.21 |
| Average | 57.40 |

<img width="643" height="842" alt="image" src="https://github.com/user-attachments/assets/e6ec963f-5e92-489a-a0d9-9067750952f7" />

## Lexical Expert Banks

The final lexical expert banks used in the paper are released in `configs/prompt_banks/`.
The NP generation and selection code is not included in this review release and will be open-sourced after the paper is accepted.
`tools/run_generate_np_bank.sh` preserves the exact command template used to regenerate dataset-specific lexical banks once the generation code is released.
