from typing import Optional

import numpy as np

import torch.cuda
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

from indexing.clip_embedding import ClipEmbedding


def mask_to_box(mask, pad=4):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    return max(0, x0 - pad), max(0, y0 - pad), x1 + pad + 1, y1 + pad + 1


class SAMMasks:
    def __init__(self,
                 model_type: str = "vit_h",
                 checkpoint: str = "sam_vit_h_4b8939.pth",
                 device: Optional[str] = None
                 ):
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device

    def load_sam(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(self.device)
        predictor = SamPredictor(sam)

        return predictor

    def generate_masks(self,
                       predictor,
                       image,
                       points,
                       point_labels,
                       bbox,
                       params,
                       mode="auto"):
        rgb = image.convert("RGB")
        np_img = np.array(rgb)

        if mode == "auto":
            defaults = dict(
                points_per_side=32,
                pred_iou_threshold=0.86,
                stability_score_threshold=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=256
            )

            if params:
                defaults.update(params)

            amg = SamAutomaticMaskGenerator(model=predictor.model, **defaults)

            masks = amg.generate(np_img)

            masks = sorted(masks, key=lambda m: m.get("area", 0), reverse=True)

            return masks

        elif mode == "prompt":
            predictor.set_image(np_img)

            _points = None
            _labels = None
            _box = None

            if points is not None:
                _points = np.array(points, dtype=np.int32)

                if point_labels is None:
                    _labels = np.ones((len(points),), dtype=np.int32)
                else:
                    _labels = np.array(point_labels, dtype=np.int32)

            if bbox is not None:
                _box = np.array(bbox, dtype=np.int32)

            masks, scores, logits = predictor.predict(
                point_coords=_points,
                point_labels=_labels,
                box=_box,
                multimask_output=True
            )

            out = []
            for m, s in zip(masks, scores):
                out.append({
                    "segmentation": m.astype(np.uint8),
                    "score": float(s),
                    "area": int(m.sum())
                })

            out = sorted(out, key=lambda x: (x["score"], x["area"]), reverse=True)

            return out

        else:
            raise ValueError("mode deve essere 'auto' o 'prompt'")

    def isolate_segment_rgba(self,
                             image,
                             mask,
                             min_area=300,
                             pad=6):
        area = int(mask.sum())
        if area < min_area:
            return None

        x0, y0, x1, y1 = mask_to_box(mask=mask,
                                     pad=pad)
        if x1 <= x0 or y1 <= y0:
            return None

        rgb = image.convert("RGB")
        crop_rgb = rgb.crop((x0, y0, x1, y1))
        crop_mask = Image.fromarray((mask[y0:y1, x0:x1] * 255).asytpe(np.uint8), mode="L")

        crop_rgba = Image.new("RGBA", crop_rgb.size, (0, 0, 0, 0))
        crop_rgba.paste(crop_rgb, (0, 0), mask=crop_mask)

        return crop_rgba

    def create_clip_embedding(self, mask):
        clip = ClipEmbedding(mask)
        mask_embedding = clip.create_embeddings()

        return mask_embedding




