from typing import Optional
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from indexing.clip_embedding import ClipEmbedding


class FaceMasks:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def detect_faces(self, image: Image.Image):
        """
        Rileva tutti i volti nell'immagine usando MTCNN.
        Restituisce bounding box e facce ritagliate.
        """
        boxes, probs = self.mtcnn.detect(image)

        faces = []
        if boxes is not None:
            for box in boxes:
                x0, y0, x1, y1 = [int(b) for b in box]
                face_crop = image.crop((x0, y0, x1, y1))
                faces.append({
                    "bbox": (x0, y0, x1, y1),
                    "image": face_crop
                })

        return faces

    def create_clip_embedding(self, face):
        """
        Genera embedding CLIP per un volto (PIL Image).
        """
        clip = ClipEmbedding([{"image": face}])
        embeddings = clip.create_embeddings(batch_size=1)
        return embeddings[0]

    def process_image(self, image: Image.Image):
        """
        Rileva volti e genera embedding per ciascuno.
        """
        faces = self.detect_faces(image)
        results = []

        for face in faces:
            emb = self.create_clip_embedding(face["image"])
            results.append({
                "bbox": face["bbox"],
                "clip_embedding": emb
            })

        return results
