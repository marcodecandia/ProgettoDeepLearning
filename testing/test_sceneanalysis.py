import pickle
import faiss
import seaborn as sns
from PIL import Image
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from indexing.clip_embedding import ClipEmbedding
from indexing.dino_embedding import DinoEmbedding
from scene_analysis.mtcnn_masks import FaceMasks
from indexing.search_logic import SearchLogic
from test_utils import load_annotations, iou, evaluate_predictions, plot_gt_vs_pred

import matplotlib
matplotlib.use("TkAgg")

# ======================
# Setup
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
mask_maker = FaceMasks(device=device)

# --- Configurazione modello ---
MODEL_NAME = "CLIP fine-tuned"  # "CLIP base" | "CLIP fine-tuned" | "DINOv2"
INDEX_TYPE = "faiss"            # "faiss" | "naive"

# --- Caricamento embedding DB + FAISS ---
embedding_db, faiss_index, metadata = None, None, None

if MODEL_NAME == "CLIP fine-tuned":
    if INDEX_TYPE == "naive":
        with open("../embeddings/embedding_database_cliptrained_list.pkl", "rb") as f:
            embedding_db = pickle.load(f)
    elif INDEX_TYPE == "faiss":
        faiss_index = faiss.read_index("../embeddings/embedding_database_cliptrained_faiss.faiss")
        with open("../embeddings/clip_embeddings_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

elif MODEL_NAME == "CLIP base":
    if INDEX_TYPE == "naive":
        with open("../embeddings/embedding_database_clipbase_list.pkl", "rb") as f:
            embedding_db = pickle.load(f)
    elif INDEX_TYPE == "faiss":
        faiss_index = faiss.read_index("../embeddings/embedding_database_clipbase_faiss.faiss")
        with open("../embeddings/clipbase_embeddings_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

elif MODEL_NAME == "DINOv2":
    if INDEX_TYPE == "naive":
        with open("../embeddings/embedding_database_dino_list.pkl", "rb") as f:
            embedding_db = pickle.load(f)
    elif INDEX_TYPE == "faiss":
        faiss_index = faiss.read_index("../embeddings/embedding_database_dino_faiss.faiss")
        with open("../embeddings/dino_embeddings_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
else:
    raise ValueError("MODEL_NAME non valido!")

# ======================
# Load ground truth annotations (COCO JSON)
# ======================
annotations_gt = load_annotations(
    database_json_path="../data/complex_scenes/train/_annotations.coco.json",
    images_dir="../data/complex_scenes/train",
    category_filter=["Naruto", "Gara", "Sakura", "Tsunade"]
)

# Conversione COCO box [x,y,w,h] -> [x1,y1,x2,y2]
def coco_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

for ann in annotations_gt:
    ann["bbox"] = coco_to_xyxy(ann["bbox"])

# Organizza GT per immagine
gt_by_image = {}
for ann in annotations_gt:
    gt_by_image.setdefault(ann["image_path"], []).append(ann)

# ======================
# Predizioni
# ======================
predictions = []

for img_path, gts in tqdm(gt_by_image.items(), desc="Processing images"):

    image_pil = Image.open(img_path).convert("RGB")
    faces = mask_maker.detect_faces(image_pil)

    if not faces:
        continue

    for face in faces:
        # --- Creazione embedding ---
        if MODEL_NAME.startswith("CLIP"):
            clip = ClipEmbedding(
                data=[{"image": face["image"]}],
                mode="image",
                trained_model=(MODEL_NAME == "CLIP fine-tuned")
            )
            embedding = clip.create_embeddings(batch_size=1)[0]

        elif MODEL_NAME == "DINOv2":
            dino = DinoEmbedding(data=face["image"])
            embedding = dino.create_embeddings(batch_size=1)

        else:
            raise ValueError(f"Modello {MODEL_NAME} non supportato")

        # --- Ricerca top-K ---
        searcher = SearchLogic(
            mask_embedding=embedding,
            embedding_db=embedding_db if INDEX_TYPE=="naive" else None,
            faiss_index=faiss_index if INDEX_TYPE=="faiss" else None,
            metadata=metadata if INDEX_TYPE=="faiss" else None,
            embedding_type="clip" if MODEL_NAME.startswith("CLIP") else "dino"
        )

        if INDEX_TYPE=="faiss" and MODEL_NAME.startswith("CLIP"):
            sims = searcher.similarity_faiss(top_k=10, metric="cosine")
        else:
            sims = searcher.similarity()

        top_pred = searcher.top_predictions(sims, top_k=1)
        character_pred = top_pred[0][1]

        pred_box = face.get("bbox", None)

        if pred_box:
            predictions.append({
                "image_path": img_path,
                "bbox": pred_box,
                "label_pred": character_pred
            })

# ======================
# Debug: stampa alcuni esempi GT vs Pred
# ======================
print("\nEsempi di match GT vs Pred con IoU:\n")
for ann, pred in zip(annotations_gt, predictions):
    print("GT:", ann["label"], ann["bbox"])
    print("Pred:", pred["label_pred"], pred["bbox"])
    print("IoU:", iou(ann["bbox"], pred["bbox"]))
    print("-" * 50)

# Visualizza prime 5 immagini con predizioni vs GT
for img_path, gts in list(gt_by_image.items())[:5]:
    preds_for_img = [p for p in predictions if p["image_path"] == img_path]
    plot_gt_vs_pred(img_path, gts, preds_for_img)

# ======================
# Etichette dei personaggi
# ======================
labels = ["Gara", "Sakura", "Tsunade", "Unlabeled", "Naruto"]

# ======================
# Valutazione
# ======================
cm, report, accuracy = evaluate_predictions(
    annotations=annotations_gt,
    predictions=predictions,
    labels=labels,
    iou_threshold=0.2
)



# ======================
# Plot Confusion Matrix
# ======================
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - {MODEL_NAME} Character Recognition")
plt.show()

# ======================
# Stampa risultati
# ======================
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
print(f"Accuracy: {accuracy*100:.2f}%")


