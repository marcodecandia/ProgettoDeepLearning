import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class DinoEmbedding:
    def __init__(self, data):
        self.data = data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model("vit_base_patch14_dinov2", pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def create_embeddings(self, batch_size=8):
        if isinstance(self.data, Image.Image):
            images = [self.data]
            single_image = True
        else:
            images = [item["image"] for item in self.data]
            single_image = False

        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Creating DINO embeddings"):
            batch_images = images[i:i + batch_size]
            batch_tensors = torch.stack([self.transform(img) for img in batch_images if img is not None]).to(self.device)

            with torch.no_grad():
                embeddings = self.model.forward_features(batch_tensors)
                embeddings /= embeddings.norm(p=2, dim=-1, keepdim=True)

            embeddings = embeddings.cpu().numpy()

            if single_image:
                all_embeddings.extend(embeddings)
            else:
                for item, emb in zip(self.data[i:i + batch_size], embeddings):
                    item["dino_embedding"] = emb
                    all_embeddings.append(emb)

        return np.array(all_embeddings) if not single_image else all_embeddings[0]