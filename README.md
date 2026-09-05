# VLex-SAM3

Official code release for **VLex-SAM3: Visual Lexicalization for Training-Free Open-Vocabulary Remote Sensing Segmentation**.

VLex-SAM3 keeps SAM3 frozen and improves open-vocabulary remote-sensing segmentation through visual lexicalization and Frozen Multi-Source Response Enhancement (FMRE). It builds class-wise lexical expert banks from annotated support image crops. At inference, FMRE combines taxonomy- and evidence-aware routing, class-wise expert aggregation, and uncertainty-guided calibration of the frozen SAM3 responses.

## Highlights

- Training-free OVSS: no model fine-tuning and no additional visual backbone.
- Data-grounded visual lexicalization: mask-highlighted support crops are described by a local MLLM as short noun phrases.
- Lexical expert banks: class-semantic profile selection keeps compact, visually grounded prompts for each class.
- Lexical OR inference: responses from multiple noun phrases in the same class are aggregated by class-wise max.
- Taxonomy- and evidence-aware routing: a fixed class-semantic policy and semantic support regulate the positive instance residual.
- FMRE: frozen SAM3 semantic, instance, and presence responses are consolidated, aggregated by class, and refined with uncertainty-guided feature calibration.

<img width="1450" height="933" alt="image" src="https://github.com/user-attachments/assets/40401932-70dc-42ac-b363-6912fab5739b" />


## Repository Layout

```text
configs/                    Dataset configs and released lexical expert banks
configs/prompt_banks/        Final VLex-SAM3 prompt banks used in the paper
configs/taxonomy_profiles.json Fixed taxonomy-aware routing policy
docs/reproducibility.md      Final settings and benchmark results
sam3/                        Frozen SAM3 image-model code
vlex_sam3_segmentor.py       MMSegmentation segmentor wrapper
custom_datasets.py           Dataset registrations for the eight benchmarks
eval.py                      MMSegmentation evaluation entry
weights/                     Checkpoint placement instructions
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

Run the commands below from the repository root. Inference uses the released lexical expert banks and routing configuration; it does not require the MLLM or lexical-bank generation code.

Run one benchmark:

```bash
python eval.py configs/cfg_loveda.py
```

Run distributed evaluation:

```bash
bash dist_test.sh configs/cfg_loveda.py
```

The released configs point to the final prompt banks in `configs/prompt_banks/` and specify both `prob_thd` and `confidence_threshold`. The shared configuration enables `routing_mode='evidence_guarded'` with `configs/taxonomy_profiles.json`. See [Reproducibility Notes](docs/reproducibility.md) for dataset settings and label conventions.

## Reported Results

| Dataset | mIoU |
| --- | ---: |
| OpenEarthMap | 46.8 |
| LoveDA | 48.4 |
| iSAID | 39.9 |
| Potsdam | 57.8 |
| Vaihingen | 63.8 |
| UAVid | 60.7 |
| UDD5 | 74.6 |
| VDD | 70.9 |
| Average | 57.9 |

Results are mIoU (%) and match Table I of the revised manuscript. Dataset scores are reported to one decimal place. Average is the arithmetic mean of the eight displayed dataset scores, rounded to one decimal place.

<img width="706" height="884" alt="image" src="https://github.com/user-attachments/assets/0fbbb324-f57e-4c03-add4-829343bbfccd" />


## Lexical Expert Banks

The final lexical expert banks used in the paper are released in `configs/prompt_banks/`.
The offline noun-phrase generation and selection code, together with the support-crop construction and replay materials, will be released after the paper is accepted. The current release supports evaluation with the supplied banks and `configs/taxonomy_profiles.json`.
