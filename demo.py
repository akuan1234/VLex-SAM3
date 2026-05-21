import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from mmseg.apis import inference_model, init_model

import custom_datasets  # noqa: F401
import vlex_sam3_segmentor  # noqa: F401


DEFAULT_PALETTE = [
    [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25],
    [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
    [240, 50, 230], [210, 245, 60], [250, 190, 190], [0, 128, 128],
    [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
]


def parse_args():
    parser = argparse.ArgumentParser(description='Run VLex-SAM3 on one image.')
    parser.add_argument('image', help='Input RGB image.')
    parser.add_argument('--config', default='configs/cfg_loveda.py', help='Dataset config.')
    parser.add_argument('--out', default='demo_prediction.png', help='Output color mask.')
    parser.add_argument('--device', default='cuda:0', help='Inference device.')
    parser.add_argument('--sam3-checkpoint', default=None, help='Override SAM3 checkpoint path.')
    parser.add_argument('--prompt-bank', default=None, help='Override prompt-bank path.')
    return parser.parse_args()


def colorize(mask, palette):
    palette = palette or DEFAULT_PALETTE
    palette_arr = np.asarray(palette, dtype=np.uint8)
    if palette_arr.shape[0] <= int(mask.max()):
        repeats = int(mask.max()) // palette_arr.shape[0] + 1
        palette_arr = np.tile(palette_arr, (repeats, 1))
    return Image.fromarray(palette_arr[mask], mode='RGB')


def main():
    args = parse_args()
    cfg_options = {}
    if args.sam3_checkpoint:
        cfg_options['model.sam3_checkpoint'] = args.sam3_checkpoint
    if args.prompt_bank:
        cfg_options['model.classname_path'] = args.prompt_bank

    model = init_model(args.config, checkpoint=None, device=args.device, cfg_options=cfg_options)
    result = inference_model(model, args.image)
    pred = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.int64)

    palette = None
    if getattr(model, 'dataset_meta', None):
        palette = model.dataset_meta.get('palette')
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colorize(pred, palette).save(out_path)
    print(f'Saved prediction to {out_path}')


if __name__ == '__main__':
    main()
