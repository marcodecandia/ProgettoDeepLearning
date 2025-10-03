import json
import os
from pathlib import Path
import matplotlib
from PIL import Image

matplotlib.use("TkAgg")
from matplotlib import patches, pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ======================
# Caricamento annotazioni COCO JSON
# ======================
def load_annotations(database_json_path, images_dir, category_filter=None):
    """Carica annotazioni da JSON COCO e restituisce lista di dict {image_path, bbox, label}"""
    with open(database_json_path, "r") as f:
        coco = json.load(f)

    # mapping category_id → nome
    cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # mapping image_id → percorso file
    img_map = {
        img["id"]: os.path.join(images_dir, img["file_name"])
        for img in coco["images"]
    }

    annotations = []
    for ann in coco["annotations"]:
        label = cat_map[ann["category_id"]]

        if category_filter and label not in category_filter:
            continue  # ignora categorie non rilevanti

        annotations.append({
            "image_path": img_map[ann["image_id"]],
            "bbox": ann["bbox"],  # formato COCO: [x, y, w, h]
            "label": label
        })

    return annotations

# ======================
# Calcolo Intersection over Union (IoU)
# ======================
def iou(boxA, boxB):
    """Calcola l'IoU tra due bounding box in formato [x,y,w,h]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou_value = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou_value

# ======================
# Valutazione predizioni
# ======================
def evaluate_predictions(annotations, predictions, labels, iou_threshold=0.5):
    """
    Confronta predizioni vs annotazioni, calcola confusion matrix, classification report e accuracy
    """
    y_true = []
    y_pred = []

    # Organizza annotazioni per immagine
    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_path"], []).append(ann)

    for pred in predictions:
        img_path = pred["image_path"]
        pred_box = pred["bbox"]
        pred_label = pred["label_pred"]

        ann_boxes = ann_by_image.get(img_path, [])
        matched = False

        # Matching con IoU
        for ann in ann_boxes:
            ann_box = ann["bbox"]
            ann_label = ann["label"]

            if iou(pred_box, ann_box) >= iou_threshold:
                y_true.append(ann_label)
                y_pred.append(pred_label)
                matched = True
                ann_boxes.remove(ann)
                break

        if not matched:
            y_true.append("None")
            y_pred.append(pred_label)

    # Aggiunge annotazioni non predette
    for img_path, anns in ann_by_image.items():
        for ann in anns:
            y_true.append(ann["label"])
            y_pred.append("None")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels)

    # Accuracy considerando solo annotazioni effettive
    tp = sum([y_t == y_p for y_t, y_p in zip(y_true, y_pred) if y_t != "None"])
    accuracy = tp / len([y_t for y_t in y_true if y_t != "None"])

    return cm, report, accuracy

# ======================
# Visualizzazione GT vs Predizioni
# ======================
def plot_gt_vs_pred(img_path, gts, preds, show=True):
    """Mostra immagine con bounding box GT (verde) e Predizioni (rosso)"""
    image = Image.open(img_path).convert("RGB")
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot GT
    for ann in gts:
        bbox = ann["bbox"]  # [x1, y1, x2, y2]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2]-bbox[0],
            bbox[3]-bbox[1],
            linewidth=2, edgecolor='g', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1]-5, ann["label"], color='g', fontsize=10)

    # Plot Predictions
    for pred in preds:
        bbox = pred["bbox"]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2]-bbox[0],
            bbox[3]-bbox[1],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[3]+10, pred["label_pred"], color='r', fontsize=10)

    if show:
        plt.show()

    return fig

