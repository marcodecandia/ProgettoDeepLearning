import faiss

from image_loader import ImageLoader
from clip_embedding import ClipEmbedding
import numpy as np
import pickle

data_root = '../data/train'

image_loader = ImageLoader(root=data_root)
data_list = image_loader.loader()

embedding_creator = ClipEmbedding(data_list)
embedding_array = embedding_creator.create_embeddings(batch_size=8)
embedding_array = embedding_array.astype(np.float32)

embedding_dim = embedding_array.shape[1]

index = faiss.IndexFlatL2(embedding_dim)

metadata = []

for i, item in enumerate(data_list):
    embedding = embedding_array[i].reshape(1, -1)
    index.add(embedding)

    metadata.append({
        "path": item["path"],
        "label": item["label"]
    })

faiss.write_index(index, "../embeddings/embedding_database_cliptrained_faiss.faiss")

with open("../embeddings/clip_embeddings_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)


