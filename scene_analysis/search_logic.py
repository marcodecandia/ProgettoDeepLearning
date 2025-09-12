from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SearchLogic:
    def __init__(self, mask_embedding, embedding_db):
        self.mask_embedding = mask_embedding
        self.embedding_db = embedding_db

    def similarity(self):
        similarities = []

        mask_emb = np.array(self.mask_embedding).reshape(1, -1)

        for idx, entry in enumerate(self.embedding_db):
            db_emb = np.array(entry["clip_embedding"]).reshape(1, -1)

            cos_sim = cosine_similarity(mask_emb, db_emb)[0][0]
            similarities.append((idx, entry["label"], cos_sim))

        return similarities

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
