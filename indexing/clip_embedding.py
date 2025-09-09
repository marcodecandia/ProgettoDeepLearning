from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from tqdm import tqdm


class ClipEmbedding:
    def __init__(self, data):
        self.data = data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def create_embeddings(self, batch_size=8):

        print(f"")

        if isinstance(self.data, Image.Image):
            images = [self.data]
            single_image = True
        else:
            images = [item["image"] for item in self.data]
            single_image = False

        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Creazione embedding"):
            batch = images[i:i + batch_size]

            inputs = self.processor(images=batch,
                                    return_tensors="pt").to(self.device)

            with torch.no_grad():
                embeddings = self.model.get_image_features(**inputs)
                embeddings /= embeddings.norm(p=2, dim=-1, keepdim=True)

            embeddings = embeddings.cpu().numpy()

            if single_image:
                all_embeddings.extend(embeddings)
            else:
                for item, emb in zip(self.data[i:i+batch_size], embeddings):
                    item["clip_embedding"] = emb

                    all_embeddings.append(emb)

        return np.array(all_embeddings)
