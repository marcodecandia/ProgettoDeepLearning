import json
import os
from pathlib import Path
import matplotlib
from PIL import Image

matplotlib.use("TkAgg")
from matplotlib import patches, pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def load_annotations(database_json_path, images_dir, category_filter=None):
    with open(database_json_path, "r") as f:
        coco = json.load(f)

    # mapping category_id → nome
    cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # mapping image_id → file path
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


def iou(boxA, boxB):
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


def evaluate_predictions(annotations, predictions, labels, iou_threshold=0.5):
    y_true = []
    y_pred = []

    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_path"], []).append(ann)

    for pred in predictions:
        img_path = pred["image_path"]
        pred_box = pred["bbox"]
        pred_label = pred["label_pred"]

        ann_boxes = ann_by_image.get(img_path, [])
        matched = False

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

    for img_path, anns in ann_by_image.items():
        for ann in anns:
            y_true.append(ann["label"])
            y_pred.append("None")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels)

    tp = sum([y_t == y_p for y_t, y_p in zip(y_true, y_pred) if y_t != "None"])
    accuracy = tp / len([y_t for y_t in y_true if y_t != "None"])

    return cm, report, accuracy


def plot_gt_vs_pred(img_path, gts, preds):
    """
    Mostra fianco a fianco l'immagine con GT e l'immagine con predizioni.
    gts: lista di annotazioni GT per l'immagine
    preds: lista di predizioni per l'immagine
    """
    image_pil = Image.open(img_path).convert("RGB")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ======================
    # Immagine con GT
    # ======================
    axes[0].imshow(image_pil)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    for gt in gts:
        x1, y1, x2, y2 = gt["bbox"]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='green', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 5, gt["label"], color='green', fontsize=12, weight='bold')

    # ======================
    # Immagine con Predizioni
    # ======================
    axes[1].imshow(image_pil)
    axes[1].set_title("Predictions")
    axes[1].axis("off")
    for pred in preds:
        x1, y1, x2, y2 = pred["bbox"]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x1, y1 - 5, pred["label_pred"], color='red', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.show()