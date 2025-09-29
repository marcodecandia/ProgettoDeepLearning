import pickle
import faiss
import numpy as np
import os
from PIL import Image
from indexing.image_loader import ImageLoader
from indexing.dino_embedding import DinoEmbedding

# Disabilita symlinks su Windows per HuggingFace
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

data_root = "../data/train"
image_loader = ImageLoader(root=data_root)
data_list = image_loader.loader()

print("A: Loaded images")

# Creazione embeddings DINOv2
embedding_creator = DinoEmbedding(data_list)  # scegli modello leggero
print("B: Initialized DINOEmbedding")

embedding_array = embedding_creator.create_embeddings(batch_size=8)
print("C: Created embeddings")

embedding_array = embedding_array.astype(np.float32)
print("D: Converted embeddings to float32")

# Metadati
metadata = [{"path": item["path"], "label": item["label"]} for item in data_list]

# Database con embeddings
embedding_database_dino = []
for i, item in enumerate(data_list):
    entry = {
        "index": i,
        "path": item["path"],
        "label": item["label"],
        "dino_embedding": embedding_array[i]
    }
    embedding_database_dino.append(entry)

with open("../embeddings/embedding_database_dino.pkl", "wb") as f:
    pickle.dump(embedding_database_dino, f)

with open("../embeddings/dino_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# Creazione indice FAISS
embedding_dim = embedding_array.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)

for emb in embedding_array:
    faiss_index.add(emb.reshape(1, -1))

faiss.write_index(faiss_index, "../embeddings/embedding_database_dino_faiss.faiss")

print("E: DINOv2 embeddings, FAISS index, and metadata saved successfully")
