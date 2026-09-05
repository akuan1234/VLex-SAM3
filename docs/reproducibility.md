# Reproducibility Notes

The revised VLex-SAM3 inference uses the released lexical expert banks in
`configs/prompt_banks/` and the routing policy in
`configs/taxonomy_profiles.json`. The following settings correspond to the
revised manuscript. Config paths are relative to `configs/`; bank paths are
relative to `configs/prompt_banks/`.

| Dataset | Config | Prompt bank | `prob_thd` | `confidence_threshold` | mIoU (%) |
| --- | --- | --- | ---: | ---: | ---: |
| OpenEarthMap | `cfg_openearthmap.py` | `openearthmap_vlex_sam3.txt` | 0.05 | 0.05 | 46.8 |
| LoveDA | `cfg_loveda.py` | `loveda_vlex_sam3.txt` | 0.50 | 0.50 | 48.4 |
| iSAID | `cfg_iSAID.py` | `isaid_vlex_sam3.txt` | 0.70 | 0.40 | 39.9 |
| Potsdam | `cfg_potsdam.py` | `potsdam_vlex_sam3.txt` | 0.10 | 0.25 | 57.8 |
| Vaihingen | `cfg_vaihingen.py` | `vaihingen_vlex_sam3.txt` | 0.00 | 0.50 | 63.8 |
| UAVid | `cfg_uavid.py` | `uavid_vlex_sam3.txt` | 0.30 | 0.20 | 60.7 |
| UDD5 | `cfg_udd5.py` | `udd5_vlex_sam3.txt` | 0.35 | 0.60 | 74.6 |
| VDD | `cfg_vdd.py` | `vdd_vlex_sam3.txt` | 0.30 | 0.50 | 70.9 |

Average mIoU across the eight benchmarks is 57.9, following the reporting
convention in the [README](../README.md#reported-results).

The shared `base_config.py` enables `routing_mode='evidence_guarded'`,
`semantic_support_kernel=7`, and `strict_taxonomy=True`. Each dataset config
specifies `evaluation_class_names` in the order of its released bank. Keep
the bank, class order, and routing configuration together when reproducing
the reported results.

Run evaluation from the repository root with the frozen SAM3 checkpoint
placed as described in [Weights](../weights/README.md):

```bash
python eval.py configs/cfg_loveda.py
```

The released Vaihingen configuration expects integer label masks with raw
`0` ignored and raw `1..6` representing the six semantic classes. It applies
`reduce_zero_label=True` during loading to obtain class IDs `0..5`; do not
apply that conversion a second time. Other datasets use the label settings
specified in their own configs.

The default lexical expert budget is five prompts per class. Larger banks may
introduce lower-quality experts, and the class-wise max aggregation can amplify
their false positives.

Inference with the supplied banks does not require an MLLM. The offline
lexical-bank generation and selection code and support-crop materials will
be released after the paper is accepted.
