import os
import pickle
import faiss
import numpy as np

from image_loader import ImageLoader
from clip_embedding import ClipEmbedding
from dino_embedding import DinoEmbedding

# ---------------------------
# Configurazioni principali
# ---------------------------
data_root = "../data/train"          # Cartella principale del dataset
save_dir = "../embeddings"           # Dove salvare gli embeddings e gli indici FAISS
os.makedirs(save_dir, exist_ok=True) # Crea la cartella se non esiste

# ---------------------------
# Caricamento immagini
# ---------------------------
image_loader = ImageLoader(root=data_root)
data_list = image_loader.loader()  # Restituisce lista di dizionari: path, label, image

# Anteprima delle prime 5 immagini
for item in data_list[:5]:
    print(item["path"], item["label"])

# ---------------------------
# Definizione modelli disponibili
# ---------------------------
MODELS = {
    "clip": lambda: ClipEmbedding(data_list, trained_model=False),       # CLIP base
    "clip_finetuned": lambda: ClipEmbedding(data_list, trained_model=True), # CLIP fine-tuned
    "dino": lambda: DinoEmbedding(data_list),                              # DINOv2
}

# ---------------------------
# Loop sui modelli per creare embeddings e database
# ---------------------------
for model_name, model_class in MODELS.items():
    print(f"\n=== Creazione embeddings con {model_name} ===")
    embedding_creator = model_class()  # Istanzia il modello

    # Creazione embeddings (batch_size=8 per evitare OOM)
    embedding_array = embedding_creator.create_embeddings(batch_size=8)

    # Converte in float32 e appiattisce
    embedding_array = np.array([np.array(e, dtype=np.float32).flatten() for e in embedding_array])
    embedding_array = np.stack(embedding_array)  # shape (N, dim)
    embedding_dim = embedding_array.shape[1]
    print(f"Dimensione embeddings per {model_name}: {embedding_dim}")

    # ===========================
    # 1. Creazione database FAISS
    # ===========================
    faiss_index = faiss.IndexFlatL2(embedding_dim)  # Indice L2 semplice
    metadata = []

    # Aggiungi tutti gli embeddings all'indice
    for i, item in enumerate(data_list):
        faiss_index.add(embedding_array[i].reshape(1, -1))  # Forma richiesta (1, dim)
        metadata.append({
            "path": item["path"],
            "label": item["label"]
        })

    # Salva l'indice FAISS su disco
    index_path = os.path.join(save_dir, f"{model_name}_faiss.index")
    faiss.write_index(faiss_index, index_path)

    # Salva i metadata associati
    metadata_path = os.path.join(save_dir, f"{model_name}_faiss_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Salvato indice FAISS e metadata per {model_name} in {save_dir}")

    # ===========================
    # 2. Creazione database Python list
    # ===========================
    embedding_db = []
    for i, item in enumerate(data_list):
        entry = {
            "index": i,
            "path": item["path"],
            "label": item["label"],
            f"{model_name}_embedding": embedding_array[i]  # Embedding salvato nel dizionario
        }
        embedding_db.append(entry)

    # Salva lista Python su disco
    db_path = os.path.join(save_dir, f"{model_name}_python_list.pkl")
    with open(db_path, "wb") as f:
        pickle.dump(embedding_db, f)

    print(f"✅ Salvato database Python list per {model_name} in {save_dir}")


