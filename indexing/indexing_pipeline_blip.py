import pickle

import faiss
import numpy as np

from indexing.blip2_embedding import Blip2Embedding
from indexing.image_loader import ImageLoader

data_root = "../data/train"
image_loader = ImageLoader(root=data_root)
data_list = image_loader.loader()

embedding_creator = Blip2Embedding(data_list, mode="image")
embedding_array = embedding_creator.create_embeddings(batch_size=8)
embedding_array = embedding_array.astype(np.float32)

metadata = [{"path": item["path"], "label": item["label"]} for item in data_list]

embedding_database_blip = []
for i, item in enumerate(data_list):
    entry = {
        "index": i,
        "path": item["path"],
        "label": item["label"],
        "blip_embedding": embedding_array[i]
    }
    embedding_database_blip.append(entry)

with open("../embeddings/embedding_database_blip.pkl", "wb") as f:
    pickle.dump(embedding_database_blip, f)


embedding_dim = embedding_array.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)

for emb in embedding_array:
    faiss_index.add(emb.reshape(1, -1))

faiss.write_index(faiss_index, "../embeddings/embedding_database_blip_faiss.faiss")

with open("../embeddings/blip_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

