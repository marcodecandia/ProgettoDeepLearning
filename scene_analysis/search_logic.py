import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SearchLogic:
    def __init__(self, mask_embedding, embedding_db):
        self.mask_embedding = mask_embedding
        self.embedding_db = embedding_db

    def similarity(self):
        similarities = []

        mask_emb = np.array(self.mask_embedding).reshape(1, -1)
        print(mask_emb.shape)

        for idx, entry in enumerate(self.embedding_db):
            db_emb = np.array(entry["clip_embedding"]).reshape(1, -1)
            print(db_emb.shape)

            cos_sim = cosine_similarity(mask_emb, db_emb)[0][0]
            similarities.append((idx, entry["label"], cos_sim))

        return similarities

    def top_predictions(self, similarities, top_k=10):
        sorted_sim = sorted(similarities, key=lambda x: x[2], reverse=True)

        return sorted_sim[:top_k]