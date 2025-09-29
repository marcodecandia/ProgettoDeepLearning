from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SearchLogic:
    def __init__(self, mask_embedding, embedding_db=None, faiss_index=None, metadata=None, embedding_type="clip"):
        self.mask_embedding = mask_embedding
        self.embedding_db = embedding_db
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.embedding_type = embedding_type

        self.embedding_key = "clip_embedding" if self.embedding_type == "clip" else "dino_embedding"

    def similarity(self):
        if self.embedding_db is None:
            raise ValueError("embedding_db non fornito")

        similarities = []

        mask_emb = np.array(self.mask_embedding).reshape(1, -1)

        emb_type = "clip_embedding" if self.embedding_type == "clip" else "dino_embedding"
        for idx, entry in enumerate(self.embedding_db):
            if self.embedding_key not in entry:
                continue

            db_emb = np.array(entry[emb_type]).reshape(1, -1)

            cos_sim = cosine_similarity(mask_emb, db_emb)[0][0]
            similarities.append((idx, entry.get("label", "Unknown"), cos_sim))

        return similarities


    def similarity_faiss(self, top_k=10, metric="cosine"):
        if self.faiss_index is None or self.metadata is None:
            raise ValueError("faiss_index o metadata non forniti")

        mask_emb = np.array(self.mask_embedding).astype(np.float32).reshape(1, -1)

        if metric == "cosine":
            mask_emb = mask_emb / np.linalg.norm(mask_emb, axis=1, keepdims=True)

        distances, indices = self.faiss_index.search(mask_emb, top_k)

        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i == -1:
                continue

            label = self.metadata[i]["label"]
            #path = self.metadata[i]["path"]

            score = 1 - dist if metric == "cosine" else - dist

            results.append((i, label, score))

        return results


    def character_similarity(self, similarities, mode="mean"):
        character_similarities = defaultdict(list)

        for _, label, score in similarities:
            character_similarities[label].append(score)

        if mode == "mean":
            character_agg = {label: np.mean(scores) for label, scores in character_similarities.items()}
        elif mode == "max":
            character_agg = {label: np.sum(scores) for label, scores in character_similarities.items()}
        else:
            raise ValueError("mode deve essere 'mean' o 'max'")

        labels, values = zip(*character_agg.items())
        probs = softmax(np.array(values))
        character_agg = dict(zip(labels, probs))

        return sorted(character_agg.items(), key=lambda x: x[1], reverse=True)




    def top_predictions(self, similarities, top_k=10):
        sorted_sim = sorted(similarities, key=lambda x: x[2], reverse=True)

        return sorted_sim[:top_k]
