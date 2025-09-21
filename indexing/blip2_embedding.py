import numpy as np
import torch.cuda
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class Blip2Embedding:
    def __init__(self, data, mode="image"):
        self.data = data
        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(self.device)

    def create_embeddings(self, batch_size=8):
        if self.mode == "image":
            if isinstance(self.data, Image.Image):
                images = [self.data]
                single_image = True
            else:
                images = [item["image"] for item in self.data]
                single_image = False

            all_embeddings = []

            for i in tqdm(range(0, len(images), batch_size), desc="Creating Blip-2 embeddings"):
                batch = images[i:i+batch_size]

                inputs = self.processor(
                    images=batch,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    embeddings = self.model.get_image_features(**inputs)
                    embeddings /= embeddings.norm(p=2, dim=-1, keepdim=True)

                embeddings = embeddings.cpu().numpy()

                if single_image:
                    all_embeddings.extend(embeddings)
                else:
                    for item, emb in zip(self.data[i:i+batch_size], embeddings):
                        item["blip_embedding"] = emb
                        all_embeddings.append(emb)

            return np.array(all_embeddings) if not single_image else all_embeddings[0]

        elif self.mode == "text":
            if isinstance(self.data, str):
                texts = [self.data]
                single_text = True
            else:
                texts = self.data
                single_text = False

            all_embeddings = []

            for i in tqdm(range(0, len(texts), batch_size), desc="Creating Blip-2 text embeddings"):
                batch = texts[i:i+batch_size]

                inputs = self.processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                with torch.no_grad():
                    embeddings = self.model.get_text_features(**inputs)
                    embeddings /= embeddings.norm(p=2, dim=-1, keepdim=True)

                embeddings = embeddings.cpu().numpy()

                all_embeddings.extend(embeddings)

            return np.array(all_embeddings) if not single_text else all_embeddings

        else:
            raise ValueError(f"Invalid mode: {self.mode}. Use 'image' or 'text'")