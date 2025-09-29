import os
import sys
import time
from tqdm import tqdm
import numpy as np
import faiss
import pickle
from PIL import Image

# Aggiungi le cartelle del progetto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "indexing")))

from indexing.clip_embedding import ClipEmbedding
from indexing.dino_embedding import DinoEmbedding
from indexing.image_loader import ImageLoader
from indexing.search_logic import SearchLogic

import wandb

# Inizializza W&B
wandb.init(project="visual-search-pipeline", config={
    "model": "clip",
    "index_type": "faiss",
    "top_k": 5
})
config = wandb.config

# Caricamento immagini
data_root = "../data/test"
image_loader = ImageLoader(root=data_root)
data_list = image_loader.loader()

# Creazione embeddings
if config.model == "clip":
    embedding_creator = ClipEmbedding(data_list, trained_model=False)
elif config.model == "clip_finetuned":
    embedding_creator = ClipEmbedding(data_list, trained_model=True)
elif config.model == "dino":
    embedding_creator = DinoEmbedding(data_list)
else:
    raise ValueError("Modello non supportato")

print(f"Creazione embeddings ({config.model})...")
embedding_array = embedding_creator.create_embeddings(batch_size=8)
embedding_array = np.array(embedding_array).astype(np.float32)

# Costruzione del database
if config.index_type == "python_list":
    embedding_db = []
    for i, item in enumerate(data_list):
        entry = {
            "index": i,
            "path": item["path"],
            "label": item["label"],
            f"{config.model}_embedding": embedding_array[i]
        }
        embedding_db.append(entry)
    faiss_index, metadata = None, None

elif config.index_type == "faiss":
    embedding_dim = embedding_array.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    metadata = []
    for i, item in enumerate(data_list):
        faiss_index.add(embedding_array[i].reshape(1, -1))
        metadata.append({
            "path": item["path"],
            "label": item["label"]
        })
    embedding_db = None
else:
    raise ValueError("Index type non supportato")

# Inizializza la logica di ricerca
searcher = SearchLogic(
    mask_embedding=None,
    embedding_db=embedding_db,
    faiss_index=faiss_index,
    metadata=metadata,
    embedding_type=config.model
)

# Valutazione
n_queries = len(data_list)
correct_top1, correct_top5 = 0, 0
start_time = time.time()

for i, item in tqdm(enumerate(data_list), total=n_queries, desc=f"{config.model}-{config.index_type}"):
    query_emb = embedding_array[i]
    gt_label = item["label"]
    query_path = item["path"]

    searcher.mask_embedding = query_emb

    if config.index_type == "python_list":
        similarities = searcher.similarity()
        top_preds = searcher.top_predictions(similarities, top_k=config.top_k)
    else:
        top_preds = searcher.similarity_faiss(top_k=config.top_k, metric="cosine")

    labels_pred = [pred[1] for pred in top_preds]

    if gt_label == labels_pred[0]:
        correct_top1 += 1
    if gt_label in labels_pred:
        correct_top5 += 1

    # Logging immagini su W&B
    query_img = Image.open(query_path).convert("RGB")
    result_imgs = []
    for idx, label, score in top_preds[:min(3, len(top_preds))]:
        if config.index_type == "faiss":
            img_path = metadata[idx]["path"]
        else:
            img_path = embedding_db[idx]["path"]
        img = Image.open(img_path).convert("RGB")
        result_imgs.append(wandb.Image(img, caption=f"{label} ({score:.2f})"))

    wandb.log({
        "query": wandb.Image(query_img, caption=f"Ground Truth: {gt_label}"),
        "results": result_imgs,
        "progress": i / n_queries
    })

# Metriche finali
end_time = time.time()
query_time = (end_time - start_time) / n_queries
accuracy_top1 = correct_top1 / n_queries
accuracy_top5 = correct_top5 / n_queries

wandb.log({
    "accuracy_top1": accuracy_top1,
    "accuracy_top5": accuracy_top5,
    "query_time": query_time,
    "index_size": n_queries
})

print(f"[{config.model} - {config.index_type}] "
      f"Top-1: {accuracy_top1:.3f}, Top-5: {accuracy_top5:.3f}, Tempo medio: {query_time:.4f}s")
