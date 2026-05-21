# Weights

SAM3 checkpoints are not included in this repository.

Place the frozen image model checkpoint at:

```text
weights/sam3/sam3.pt
```

or pass a custom path with:

```bash
python eval.py configs/cfg_loveda.py --cfg-options model.sam3_checkpoint=/path/to/sam3.pt
```
