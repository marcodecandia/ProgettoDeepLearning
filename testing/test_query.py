import os
import pickle
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.special import softmax

from indexing.image_loader import ImageLoader
from indexing.search_logic import SearchLogic

# ---------------------------
# Configurazioni
# ---------------------------
MODELS = ["clip", "clip_finetuned", "dino"]
INDEX_TYPES = ["python_list", "faiss"]
TOP_K = 5

data_root = "../data/test"
db_dir = "../embeddings"
os.makedirs("./comparisons", exist_ok=True)

# Carica immagini test
image_loader = ImageLoader(root=data_root)
data_list = image_loader.loader()
n_queries = len(data_list)

# Per grafici comparativi
results_metrics = {metric: {} for metric in ["accuracy_top1", "accuracy_top5", "mrr", "cross_entropy", "query_time"]}

# ---------------------------
# Loop su configurazioni
# ---------------------------
for model_name in MODELS:
    for index_type in INDEX_TYPES:
        config_name = f"{model_name}_{index_type}"
        print(f"\n=== Test {config_name} ===")

        # Caricamento database
        embedding_db, faiss_index, metadata = None, None, None
        if index_type == "faiss":
            faiss_path = os.path.join(db_dir, f"{config_name}.index")
            meta_path = os.path.join(db_dir, f"{config_name}_metadata.pkl")
            if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
                print(f"[WARNING] {config_name} non trovato, salto.")
                continue
            import faiss
            faiss_index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
        else:
            db_path = os.path.join(db_dir, f"{config_name}.pkl")
            if not os.path.exists(db_path):
                print(f"[WARNING] {config_name} non trovato, salto.")
                continue
            with open(db_path, "rb") as f:
                embedding_db = pickle.load(f)

        # Inizializza SearchLogic
        searcher = SearchLogic(
            mask_embedding=None,
            embedding_db=embedding_db,
            faiss_index=faiss_index,
            metadata=metadata,
            embedding_type=model_name
        )

        # Imposta embedding key per clip_finetuned
        if model_name == "clip_finetuned":
            searcher.embedding_key = "clip_finetuned_embedding"

        # ---------------------------
        # Metriche
        # ---------------------------
        correct_top1, correct_top5, mrr_list, cross_entropy_list, query_times = 0, 0, [], [], []

        # ---------------------------
        # Loop query
        # ---------------------------
        for i, item in enumerate(data_list):
            gt_label = item["label"]
            query_path = item["path"]
            query_img = Image.open(query_path).convert("RGB")

            # ---- Creazione embedding query ----
            if model_name.startswith("clip"):
                # CLIP
                query_emb = searcher.encode_image(query_path)
            elif model_name.startswith("dino"):
                # DINOv2
                from indexing.dino_embedding import DinoEmbedding

                dino_embedder = DinoEmbedding(data=query_img)
                query_emb = dino_embedder.create_embeddings()
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            searcher.mask_embedding = query_emb

            # ---- Ricerca ----
            start = time.time()
            if index_type == "python_list":
                similarities = searcher.similarity()
                top_preds = searcher.top_predictions(similarities, top_k=TOP_K)
                scores_all = np.array([sim[2] for sim in similarities])
                labels_all = [sim[1] for sim in similarities]
            else:
                if model_name.startswith("clip"):
                    top_preds = searcher.similarity_faiss(top_k=TOP_K, metric="cosine")
                elif model_name.startswith("dino"):
                    top_preds = searcher.similarity_faiss_dino(top_k=TOP_K)
                scores_all = np.array([pred[2] for pred in top_preds])
                labels_all = [pred[1] for pred in top_preds]
            end = time.time()
            query_times.append(end - start)

            if not top_preds:
                continue

            # ---- Metriche ----
            labels_pred = [pred[1] for pred in top_preds]
            correct_top1 += int(gt_label == labels_pred[0])
            correct_top5 += int(gt_label in labels_pred)

            ranks = [j + 1 for j, (_, lbl, _) in enumerate(top_preds) if lbl == gt_label]
            mrr_list.append(1 / ranks[0] if ranks else 0)

            probs = softmax(scores_all)
            try:
                true_idx = labels_all.index(gt_label)
                cross_entropy_list.append(-np.log(probs[true_idx] + 1e-10))
            except ValueError:
                cross_entropy_list.append(-np.log(1e-10))

        # ---------------------------
        # Metriche finali
        # ---------------------------
        n = len(data_list)
        accuracy_top1 = correct_top1 / n
        accuracy_top5 = correct_top5 / n
        avg_query_time = np.mean(query_times)
        mrr = np.mean(mrr_list)
        cross_entropy = np.mean(cross_entropy_list)

        print(f"[{config_name}] Top-1: {accuracy_top1:.3f}, Top-5: {accuracy_top5:.3f}, "
              f"MRR: {mrr:.3f}, CE: {cross_entropy:.4f}, Tempo medio: {avg_query_time:.4f}s")

        # Salvataggio metriche per grafici
        results_metrics["accuracy_top1"][config_name] = accuracy_top1
        results_metrics["accuracy_top5"][config_name] = accuracy_top5
        results_metrics["mrr"][config_name] = mrr
        results_metrics["cross_entropy"][config_name] = cross_entropy
        results_metrics["query_time"][config_name] = avg_query_time

# ---------------------------
# Grafici comparativi
# ---------------------------
for metric, values in results_metrics.items():
    plt.figure(figsize=(8,5))
    names = list(values.keys())
    scores = list(values.values())
    plt.bar(names, scores, color='skyblue')
    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"./comparisons/{metric}_comparison.png")
    plt.close()

print("Testing completato. Grafici salvati in ./comparisons/.")


