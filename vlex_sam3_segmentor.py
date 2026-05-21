import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from prompt_bank import load_class_prompts
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


@MODELS.register_module()
class VLexSAM3Segmentation(BaseSegmentor):
    """Training-free VLex-SAM3 segmentor with frozen SAM3 responses."""

    FGC_FEATURE_LEVEL = 1
    FGC_MAX_SIZE = 288
    FGC_NUM_ITER = 3
    FGC_ALPHA = 0.50
    FGC_TEMPERATURE = 0.07
    FGC_UNCERTAINTY_MARGIN = 0.12
    FGC_UNCERTAINTY_TOP = 0.80
    FGC_AFTER_CLASS = True
    FGC_USE_ADAPTIVE_BACKGROUND = False
    FGC_BG_MARGIN = 0.08

    def __init__(self, classname_path,
                 device=torch.device('cuda'),
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 use_transformer_decoder=True,
                 use_feature_guided_calibration=True,
                 sam3_checkpoint='weights/sam3/sam3.pt',
                 sam3_bpe_path='sam3/assets/bpe_simple_vocab_16e6.txt.gz',
                 **kwargs):
        super().__init__()

        self.device = device
        self._ddp_dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        # Initialize SAM3 model
        model = build_sam3_image_model(
            bpe_path=sam3_bpe_path,
            checkpoint_path=sam3_checkpoint,
            device=str(device),
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        # FMRE keeps the frozen SAM3 interface fully training-free: instance
        # responses, semantic responses, and presence scores are consolidated
        # without introducing any trainable fusion head.
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder
        self.use_feature_guided_calibration = use_feature_guided_calibration
        self.fgc_feature_level = self.FGC_FEATURE_LEVEL
        self.fgc_max_size = self.FGC_MAX_SIZE
        self.fgc_num_iter = self.FGC_NUM_ITER
        self.fgc_alpha = self.FGC_ALPHA
        self.fgc_temperature = self.FGC_TEMPERATURE
        self.fgc_uncertainty_margin = self.FGC_UNCERTAINTY_MARGIN
        self.fgc_uncertainty_top = self.FGC_UNCERTAINTY_TOP
        self.fgc_after_class = self.FGC_AFTER_CLASS
        self.fgc_use_adaptive_background = self.FGC_USE_ADAPTIVE_BACKGROUND
        self.fgc_bg_margin = self.FGC_BG_MARGIN
        self._last_fgc_backbone_out = None

    def _query_logits_to_class_logits(self, query_logits):
        if self.num_cls == self.num_queries:
            return query_logits
        cls_index = nn.functional.one_hot(self.query_idx, num_classes=self.num_cls)
        cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
        return (query_logits.unsqueeze(0) * cls_index).max(1)[0]

    @staticmethod
    def _shift_with_replicate(x, dy, dx):
        h_pad = abs(dy)
        w_pad = abs(dx)
        padded = F.pad(x, [w_pad, w_pad, h_pad, h_pad], mode='replicate')
        y_start = h_pad + dy
        x_start = w_pad + dx
        return padded[..., y_start:y_start + x.shape[-2], x_start:x_start + x.shape[-1]]

    def _fmre_instance_response(self, inference_state, output_size):
        h, w = output_size
        response = torch.zeros((h, w), device=self.device)
        if not self.use_transformer_decoder:
            return response
        if inference_state['masks_logits'].shape[0] <= 0:
            return response

        inst_len = inference_state['masks_logits'].shape[0]
        for inst_id in range(inst_len):
            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
            instance_score = inference_state['object_score'][inst_id]

            if instance_logits.shape != (h, w):
                instance_logits = F.interpolate(
                    instance_logits.view(1, 1, *instance_logits.shape),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

            response = torch.max(response, instance_logits * instance_score)
        return response

    def _fmre_semantic_response(self, inference_state, output_size):
        if not self.use_sem_seg:
            return None

        h, w = output_size
        semantic_logits = inference_state['semantic_mask_logits']
        if semantic_logits.shape != (h, w):
            semantic_logits = F.interpolate(
                semantic_logits,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        return semantic_logits

    def _fmre_consolidate_prompt_response(self, inference_state, output_size):
        response = self._fmre_instance_response(inference_state, output_size)

        semantic_response = self._fmre_semantic_response(inference_state, output_size)
        if semantic_response is not None:
            response = torch.max(response, semantic_response)

        if self.use_presence_score:
            response = response * inference_state["presence_score"]
        return response

    def _fmre_should_calibrate_after_class(self):
        return self.use_feature_guided_calibration and self.fgc_after_class

    def _fmre_uncertainty_guided_calibration(self, seg_logits, backbone_out, logits_are_class=False):
        return self._feature_guided_propagation(
            seg_logits,
            backbone_out,
            logits_are_class=logits_are_class
        )

    def _feature_guided_propagation(self, seg_logits, backbone_out, logits_are_class=False):
        if not self.use_feature_guided_calibration:
            return seg_logits
        if backbone_out is None or "backbone_fpn" not in backbone_out:
            return seg_logits

        features = backbone_out["backbone_fpn"][int(self.fgc_feature_level)]
        if features is None:
            return seg_logits

        h, w = seg_logits.shape[-2:]
        feat = features.detach().float()
        if feat.dim() == 4:
            feat = feat[0]

        max_size = int(self.fgc_max_size)
        if max_size > 0 and max(feat.shape[-2:]) > max_size:
            scale = float(max_size) / max(feat.shape[-2:])
            feat_h = max(1, int(round(feat.shape[-2] * scale)))
            feat_w = max(1, int(round(feat.shape[-1] * scale)))
            feat = F.interpolate(
                feat.unsqueeze(0),
                size=(feat_h, feat_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        feat_h, feat_w = feat.shape[-2:]
        low_logits = F.interpolate(
            seg_logits.unsqueeze(0).float(),
            size=(feat_h, feat_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        low_class_logits = low_logits if logits_are_class else self._query_logits_to_class_logits(low_logits)
        top_values = torch.topk(low_class_logits, k=min(2, self.num_cls), dim=0).values
        top1 = top_values[0]
        top2 = top_values[1] if top_values.shape[0] > 1 else torch.zeros_like(top1)
        uncertainty = ((top1 - top2) < self.fgc_uncertainty_margin) | (top1 < self.fgc_uncertainty_top)
        uncertainty = uncertainty.float().clamp(0, 1)
        if float(uncertainty.max().item()) <= 0:
            return seg_logits

        feat = F.normalize(feat, dim=0)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        temperature = max(float(self.fgc_temperature), 1e-4)

        current = low_logits
        for _ in range(max(int(self.fgc_num_iter), 0)):
            weighted_sum = current.clone()
            weight_sum = torch.ones((1, feat_h, feat_w), device=current.device, dtype=current.dtype)
            for dy, dx in offsets:
                shifted_feat = self._shift_with_replicate(feat.unsqueeze(0), dy, dx).squeeze(0)
                affinity = (feat * shifted_feat).sum(0, keepdim=True)
                affinity = torch.exp((affinity - 1.0) / temperature).clamp(max=1.0)
                shifted_logits = self._shift_with_replicate(current.unsqueeze(0), dy, dx).squeeze(0)
                weighted_sum = weighted_sum + affinity * shifted_logits
                weight_sum = weight_sum + affinity
            propagated = weighted_sum / weight_sum.clamp_min(1e-6)
            current = current * (1.0 - uncertainty) + propagated * uncertainty

        refined = F.interpolate(
            current.unsqueeze(0),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        uncertainty_full = F.interpolate(
            uncertainty.view(1, 1, feat_h, feat_w),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0).clamp(0, 1)
        blend = (float(self.fgc_alpha) * uncertainty_full).clamp(0, 1)
        seg_logits = seg_logits * (1.0 - blend) + refined * blend
        return seg_logits.clamp(0, 1)

    def _inference_single_view(self, image):
        """Inference on a single PIL image or crop patch."""
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)

            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)
                seg_logits[query_idx] = self._fmre_consolidate_prompt_response(
                    inference_state,
                    output_size=(h, w)
                )

        self._last_fgc_backbone_out = inference_state.get("backbone_out")
        if not self.fgc_after_class:
            seg_logits = self._fmre_uncertainty_guided_calibration(
                seg_logits,
                self._last_fgc_backbone_out
            )
        return seg_logits

    def slide_inference(self, image, stride, crop_size):
        """Inference by sliding-window with overlap using PIL cropping."""
        w_img, h_img = image.size

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size

        use_slide_class_fgc = self._fmre_should_calibrate_after_class()
        pred_channels = self.num_cls if use_slide_class_fgc else self.num_queries

        # Initialize accumulators
        preds = torch.zeros((pred_channels, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)

                # Adjust start points to ensure crop size is valid at boundaries
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                # Crop via PIL
                crop_img = image.crop((x1, y1, x2, y2))

                # Inference on crop
                crop_seg_logit = self._inference_single_view(crop_img)
                if use_slide_class_fgc:
                    crop_seg_logit = self._query_logits_to_class_logits(crop_seg_logit)
                    crop_seg_logit = self._fmre_uncertainty_guided_calibration(
                        crop_seg_logit,
                        self._last_fgc_backbone_out,
                        logits_are_class=True
                    )

                # Accumulate results
                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."

        preds = preds / count_mat
        if use_slide_class_fgc:
            self._last_fgc_backbone_out = None
        return preds

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            # Fallback for meta info construction
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]

        for i, meta in enumerate(batch_img_metas):
            # Load original image to preserve details for SAM3
            image_path = meta.get('img_path')
            image = Image.open(image_path).convert('RGB')
            ori_shape = meta['ori_shape']

            # Determine inference mode
            if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                seg_logits = self.slide_inference(image, self.slide_stride, self.slide_crop)
            else:
                seg_logits = self._inference_single_view(image)

            # Resize to original shape if necessary (e.g. padding effects)
            if seg_logits.shape[-2:] != ori_shape:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0),
                    size=ori_shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            # Post-processing
            if self.num_cls != self.num_queries and seg_logits.shape[0] == self.num_queries:
                seg_logits = self._query_logits_to_class_logits(seg_logits)
                seg_pred = seg_logits.argmax(0, keepdim=True)
            elif seg_logits.shape[0] != self.num_cls:
                raise RuntimeError(
                    f"Unexpected logit channels: {seg_logits.shape[0]}, "
                    f"expected {self.num_queries} queries or {self.num_cls} classes."
                )

            if (self._fmre_should_calibrate_after_class()
                    and self._last_fgc_backbone_out is not None):
                seg_logits = self._fmre_uncertainty_guided_calibration(
                    seg_logits,
                    self._last_fgc_backbone_out,
                    logits_are_class=True
                )

            seg_pred = torch.argmax(seg_logits, dim=0)

            # Apply probability threshold
            max_vals = seg_logits.max(0)[0]
            if self.use_feature_guided_calibration and self.fgc_use_adaptive_background and self.num_cls > 1:
                top_values = torch.topk(seg_logits, k=2, dim=0).values
                margin = top_values[0] - top_values[1]
                low_conf_ambiguous = (max_vals < self.prob_thd) & (margin < self.fgc_bg_margin)
                seg_pred[low_conf_ambiguous] = self.bg_idx
            else:
                seg_pred[max_vals < self.prob_thd] = self.bg_idx

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': seg_logits}),
                'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
            })

        return data_samples

    def _forward(data_samples):
        """
    """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    return load_class_prompts(path)
