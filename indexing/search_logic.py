import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import timm
from torchvision import transforms

class SearchLogic:
    def __init__(self, mask_embedding=None, embedding_db=None, faiss_index=None, metadata=None, embedding_type="clip"):
        self.mask_embedding = mask_embedding
        self.embedding_db = embedding_db
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.embedding_type = embedding_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.embedding_type in ["clip", "clip_finetuned"]:
            self.embedding_key = "clip_embedding"
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        elif self.embedding_type == "dino":
            self.embedding_key = "dino_embedding"
            # Modello DINO 768-dim
            self.model = timm.create_model('vit_base_patch14_dinov2', pretrained=True).to(self.device)
            self.model.eval()
            self.preprocess = transforms.Compose([
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Tipo embedding non supportato: {self.embedding_type}")

    # -----------------------------
    # Estrai embedding immagine
    # -----------------------------
    def encode_image(self, image_input):
        """
        image_input: PIL.Image oppure percorso immagine (str)
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        if self.embedding_type in ["clip", "clip_finetuned"]:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)

        elif self.embedding_type == "dino":
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats = self.model.forward_features(img_tensor)
                # Prendi il CLS token 768-dim
                if isinstance(feats, dict):
                    embedding = feats["x_norm_clstoken"]
                else:
                    embedding = feats[:, 0, :]
                embedding = torch.nn.functional.normalize(embedding, dim=-1)

        embedding = embedding.cpu().numpy()
        return embedding.flatten()

    # -----------------------------
    # Similarità su lista Python
    # -----------------------------
    def similarity(self):
        if self.embedding_db is None:
            raise ValueError("embedding_db non fornito")

        mask_emb = np.array(self.mask_embedding).reshape(1, -1)
        similarities = []

        for idx, entry in enumerate(self.embedding_db):
            if self.embedding_key not in entry:
                continue
            db_emb = np.array(entry[self.embedding_key]).reshape(1, -1)
            # Normalizzazione L2
            db_emb /= np.linalg.norm(db_emb, axis=1, keepdims=True)
            cos_sim = cosine_similarity(mask_emb, db_emb)[0][0]
            similarities.append((idx, entry.get("label", "Unknown"), cos_sim))

        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities

    # -----------------------------
    # Similarità FAISS
    # -----------------------------
    def similarity_faiss(self, top_k=10, metric="cosine"):
        if self.faiss_index is None or self.metadata is None:
            raise ValueError("faiss_index o metadata non forniti")

        mask_emb = np.array(self.mask_embedding, dtype=np.float32).reshape(1, -1)
        if metric == "cosine":
            mask_emb /= np.linalg.norm(mask_emb, axis=1, keepdims=True)

        distances, indices = self.faiss_index.search(mask_emb, top_k)
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i == -1:
                continue
            label = self.metadata[i]["label"]
            score = 1 - dist if metric == "cosine" else -dist
            results.append((i, label, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def similarity_faiss_dino(self, top_k=10):
        """
        Similarità FAISS per embeddings DINO (768-dim).
        """
        if self.faiss_index is None or self.metadata is None:
            raise ValueError("faiss_index o metadata non forniti")

        # Query embedding
        mask_emb = np.array(self.mask_embedding, dtype=np.float32).reshape(1, -1)
        mask_emb /= np.linalg.norm(mask_emb, axis=1, keepdims=True)  # normalizza sempre

        # Ricerca top-k in FAISS
        distances, indices = self.faiss_index.search(mask_emb.astype(np.float32), top_k)

        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i == -1:
                continue
            label = self.metadata[i]["label"]
            score = 1 - dist  # per cosine similarity normalizzata
            results.append((i, label, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    # -----------------------------
    # Top-k predizioni
    # -----------------------------
    def top_predictions(self, similarities, top_k=10):
        return sorted(similarities, key=lambda x: x[2], reverse=True)[:top_k]

