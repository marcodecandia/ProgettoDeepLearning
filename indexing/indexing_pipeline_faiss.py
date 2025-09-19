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
index.add(embedding_array)

metadata = [{"path": item["path"], "label": item["label"]} for item in data_list]

faiss.write_index(index, "../embeddings/embedding_database_trained_faiss.faiss")

with open("../embeddings/clip_embeddings_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)


