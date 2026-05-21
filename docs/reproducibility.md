# Reproducibility Notes

VLex-SAM3 is training-free. The released configs use the final lexical expert
banks in `configs/prompt_banks/` and the final probability thresholds reported
in the paper.

| Dataset | Config | Prompt bank | `prob_thd` | mIoU |
| --- | --- | --- | ---: | ---: |
| OpenEarthMap | `cfg_openearthmap.py` | `openearthmap_vlex_sam3.txt` | 0.05 | 44.29 |
| LoveDA | `cfg_loveda.py` | `loveda_vlex_sam3.txt` | 0.50 | 48.52 |
| iSAID | `cfg_iSAID.py` | `isaid_vlex_sam3.txt` | 0.60 | 37.78 |
| Potsdam | `cfg_potsdam.py` | `potsdam_vlex_sam3.txt` | 0.10 | 58.47 |
| Vaihingen | `cfg_vaihingen.py` | `vaihingen_vlex_sam3.txt` | 0.00 | 63.87 |
| UAVid | `cfg_uavid.py` | `uavid_vlex_sam3.txt` | 0.30 | 60.61 |
| UDD5 | `cfg_udd5.py` | `udd5_vlex_sam3.txt` | 0.30 | 74.41 |
| VDD | `cfg_vdd.py` | `vdd_vlex_sam3.txt` | 0.30 | 71.21 |

Average mIoU across the eight benchmarks is 57.4.

The default lexical expert budget is five prompts per class. Larger banks may
introduce lower-quality experts, and the class-wise max aggregation can amplify
their false positives.
