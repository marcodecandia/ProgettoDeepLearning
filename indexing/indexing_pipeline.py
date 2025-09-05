from image_loader import ImageLoader
from clip_embedding import ClipEmbedding
import numpy as np
import pickle

data_root = '../data/test'

image_loader = ImageLoader(root=data_root)
data_list = image_loader.loader()
embedding_creator = ClipEmbedding(data_list)
embedding_array = embedding_creator.create_embeddings(batch_size=8)

np.save("clip_embeddings.npy", embedding_array)

metadata = [{"path": item["path"], "label": item["label"]} for item in data_list]

with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


