from typing import Optional
import numpy as np
import torch.cuda
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from indexing.clip_embedding import ClipEmbedding


def mask_to_box(mask, pad=4):
    """
    Calcola il bounding box (x0,y0,x1,y1) a partire da una maschera binaria.
    pad: margine extra attorno al box.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    return max(0, x0 - pad), max(0, y0 - pad), x1 + pad + 1, y1 + pad + 1


class SAMMasks:
    """
    Classe che gestisce la segmentazione con SAM:
    - Caricamento modello (predictor)
    - Generazione maschere (auto o con prompt)
    - Isolamento segmento in RGBA
    - Creazione embedding CLIP per la maschera
    """
    def __init__(self,
                 model_type: str = "vit_b",
                 checkpoint: str = "../models/sam_vit_b_01ec64.pth",
                 device: Optional[str] = None):
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = torch.device("cpu")  # uso forzato su CPU

    def load_sam(self):
        """ Carica il modello SAM e restituisce il predictor. """
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(self.device)
        predictor = SamPredictor(sam)

        print("Using SAM")
        return predictor

    def generate_masks(self,
                       predictor,
                       image,
                       points,
                       point_labels,
                       bbox,
                       params,
                       mode="auto"):
        """
        Genera maschere di segmentazione:
        - mode="auto": usa SamAutomaticMaskGenerator
        - mode="prompt": segmenta con punti/bounding box forniti
        """
        rgb = image.convert("RGB")
        np_img = np.array(rgb)

        if mode == "auto":
            defaults = dict(
                points_per_side=8,
                pred_iou_thresh=0.75,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=4,
                min_mask_region_area=512
            )
            if params:
                defaults.update(params)

            amg = SamAutomaticMaskGenerator(model=predictor.model, **defaults)
            masks = amg.generate(np_img)
            return sorted(masks, key=lambda m: m.get("area", 0), reverse=True)

        elif mode == "prompt":
            predictor.set_image(np_img)
            _points, _labels, _box = None, None, None

            if points is not None:
                _points = np.array(points, dtype=np.int32)
                _labels = np.ones((len(points),), dtype=np.int32) if point_labels is None else np.array(point_labels, dtype=np.int32)
            if bbox is not None:
                _box = np.array(bbox, dtype=np.int32)

            masks, scores, logits = predictor.predict(
                point_coords=_points,
                point_labels=_labels,
                box=_box,
                multimask_output=True
            )

            out = [{"segmentation": m.astype(np.uint8),
                    "score": float(s),
                    "area": int(m.sum())} for m, s in zip(masks, scores)]

            return sorted(out, key=lambda x: (x["score"], x["area"]), reverse=True)

        else:
            raise ValueError("mode deve essere 'auto' o 'prompt'")

    def isolate_segment_rgba(self, image, mask, min_area=300, pad=6):
        """
        Estrae e restituisce un segmento in RGBA a partire da una maschera.
        Restituisce None se l’area è troppo piccola.
        """
        area = int(mask.sum())
        if area < min_area:
            return None

        x0, y0, x1, y1 = mask_to_box(mask=mask, pad=pad)
        if x1 <= x0 or y1 <= y0:
            return None

        crop_rgb = image.convert("RGB")
        mask_crop = mask[y0:y1, x0:x1] * 255
        crop_mask = Image.fromarray(mask_crop.astype(np.uint8), mode="L")
        crop_mask = crop_mask.resize(crop_rgb.size, resample=Image.NEAREST)

        crop_rgba = Image.new("RGBA", crop_rgb.size, (0, 0, 0, 0))
        crop_rgba.paste(crop_rgb, (0, 0), mask=crop_mask)

        return crop_rgba

    def create_clip_embedding(self, mask):
        """ Crea embedding CLIP a partire da una maschera. """
        clip = ClipEmbedding([{"image": mask}])
        mask_embedding_array = clip.create_embeddings(batch_size=1)
        return mask_embedding_array[0]





