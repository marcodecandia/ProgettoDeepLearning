from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from tqdm import tqdm


class ClipEmbedding:
    def __init__(self, data, mode="image"):
        self.data = data
        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def create_embeddings(self, batch_size=8, trained_model=True):

        if trained_model:
            self.model.load_state_dict(torch.load("C:/Users/utente/PycharmProjects/ProgettoDeepLearning/models/trained_clip_model.pth", map_location="cpu"))

        if self.mode == "text":
            if isinstance(self.data, str):
                texts = [self.data]
                single_text = True
            else:
                texts = self.data
                single_text = False

            all_embeddings = []

            for i in tqdm(range(0, len(texts), batch_size), desc="Creating text embedding"):
                batch = texts[i:i + batch_size]

                inputs = self.processor(text=batch,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True).to(self.device)

                with torch.no_grad():
                    embeddings = self.model.get_text_features(**inputs)
                    embeddings /= embeddings.norm(p=2, dim=-1, keepdim=True)

                embeddings = embeddings.cpu().numpy()

                all_embeddings.extend(embeddings)

            return np.array(all_embeddings) if not single_text else all_embeddings[0]

        elif self.mode == "image":
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

            return np.array(all_embeddings) if not single_image else all_embeddings[0]

        else:
            raise ValueError(f"Invalid mode: {self.mode}. Use 'image' or 'text'")