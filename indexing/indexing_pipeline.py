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

embedding_database = []

for i, item in enumerate(data_list):
    entry = {
        "index": i,
        "path": item["path"],
        "label": item["label"],
        "clip_embedding": embedding_array[i]
    }
    embedding_database.append(entry)


with open("embedding_database.pkl", "wb") as f:
    pickle.dump(embedding_database, f)

with open("embedding_database.pkl", "rb") as f:
    embedding_database = pickle.load(f)

print(embedding_database[0]["index"])
print(embedding_database[0]["label"])
print(embedding_database[0]["clip_embedding"].shape)


