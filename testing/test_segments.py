import pickle
import faiss
from PIL import Image
import torch
from tqdm import tqdm

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

# Carica embedding database + FAISS
embedding_db_path = "../embeddings/embedding_database_trained.pkl"
with open(embedding_db_path, "rb") as f:
    embedding_database = pickle.load(f)

faiss_index = faiss.read_index("../embeddings/embedding_database_trained_faiss.faiss")
with open("../embeddings/clip_embeddings_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

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

for img_path, gts in tqdm(gt_by_image.items()):

    image_pil = Image.open(img_path).convert("RGB")
    faces = mask_maker.detect_faces(image_pil)

    if faces is None:
        continue

    for face in faces:
        clip_embedding = mask_maker.create_clip_embedding(face["image"])
        search_logic = SearchLogic(mask_embedding=clip_embedding, faiss_index=faiss_index, metadata=metadata)
        similarities = search_logic.similarity_faiss(10, "cosine")
        top_pred = search_logic.top_predictions(similarities, 1)
        character_pred = top_pred[0][1]

        pred_box = face.get("bbox", None)  # bbox in formato [x1,y1,x2,y2]

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

for img_path, gts in list(gt_by_image.items())[:5]:  # primi 5 esempi
    preds_for_img = [p for p in predictions if p["image_path"] == img_path]
    plot_gt_vs_pred(img_path, gts, preds_for_img)

# ======================
# Valutazione
# ======================
cm, report, accuracy = evaluate_predictions(
    annotations=annotations_gt,
    predictions=predictions,
    iou_threshold=0.2  # abbassata per tollerare differenze tra COCO e MTCNN
)

print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
print(f"Accuracy: {accuracy*100:.2f}%")
