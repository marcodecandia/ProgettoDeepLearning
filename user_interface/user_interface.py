import pickle
import faiss
import gradio as gr
import torch
from PIL import Image
import json
import numpy as np

from indexing.clip_embedding import ClipEmbedding
from indexing.dino_embedding import DinoEmbedding
from scene_analysis.mtcnn_masks import FaceMasks
from scene_analysis.sam_masks import SAMMasks
from indexing.search_logic import SearchLogic

# ----------------- Caricamento database -----------------
with open("../embeddings/embedding_database_clipbase_list.pkl", "rb") as f:
    embedding_database_base = pickle.load(f)
with open("../embeddings/embedding_database_cliptrained_list.pkl", "rb") as f:
    embedding_database = pickle.load(f)
with open("../embeddings/embedding_database_dino_list.pkl", "rb") as f:
    embedding_database_dino = pickle.load(f)
faiss_index = faiss.read_index("../embeddings/embedding_database_cliptrained_faiss.faiss")
with open("../embeddings/clip_embeddings_metadata.pkl", "rb") as f:
    faiss_metadata = pickle.load(f)
faiss_index_dino = faiss.read_index("../embeddings/embedding_database_dino_faiss.faiss")



device = "cuda" if torch.cuda.is_available() else "cpu"
mask_maker = FaceMasks(device=device)

# ----------------- Funzioni principali -----------------
"""
def query_database(image=None, text=None, top_k=5, model_name="CLIP fine-tuned", db_type="faiss"):
    # ----------------- Selezione modello e database -----------------
    if model_name.startswith("CLIP"):
        clip_version = "fine-tuned" if model_name == "CLIP fine-tuned" else "base"
        embedding_db = embedding_database if clip_version == "fine-tuned" else embedding_database_base

        if image is not None:
            clip = ClipEmbedding(data=image, mode="image", trained_model=(clip_version=="fine-tuned"))
            embedding = clip.create_embeddings()
        elif text is not None and text.strip() != "":
            clip = ClipEmbedding(data=text, mode="text", trained_model=(clip_version=="fine-tuned"))
            embedding = clip.create_embeddings()
        else:
            return "Load an image or write a description.", None

        embedding_type = "clip"

    elif model_name == "DINOv2":
        if image is None:
            return "DINOv2 only supports image search", None
        embedding_db = embedding_database_dino
        dino = DinoEmbedding(image)
        embedding = dino.create_embeddings()
        embedding_type = "dino"
    else:
        return "Invalid model selected.", None

    # ----------------- Logica di ricerca -----------------
    search_logic = SearchLogic(
        mask_embedding=embedding,
        embedding_db=embedding_db,
        faiss_index=faiss_index if embedding_type=="clip" else None,
        metadata=faiss_metadata if embedding_type=="clip" else None,
        embedding_type=embedding_type
    )

    if db_type=="faiss" and embedding_type=="clip":
        top_preds = search_logic.similarity_faiss(top_k=top_k, metric="cosine")
    else:
        similarities = search_logic.similarity()
        top_preds = search_logic.top_predictions(similarities, top_k)

    # ----------------- Costruzione risultati -----------------
    results = []
    for i, label, score in top_preds:
        db_image_path = faiss_metadata[i]["path"] if embedding_type=="clip" else embedding_db[i]["path"]
        db_image = Image.open(db_image_path).convert("RGB")

        # Percentuali solo per visualizzazione
        if embedding_type=="clip":
            top_labels = search_logic.character_similarity([(i,label,score)], mode="mean")[:3]
            label_str = " / ".join([f"{l[0]} ({l[1]*100:.1f}%)" for l in top_labels])
        else:
            label_str = f"{label} ({score*100:.1f}%)"

        results.append((db_image, label_str))

    return results
"""

def query_database(image=None, text=None, top_k=5, model_name="CLIP fine-tuned", db_type="faiss"):
    # ----------------- Selezione modello -----------------
    if model_name.startswith("CLIP"):
        clip_version = "fine-tuned" if model_name == "CLIP fine-tuned" else "base"

        if image is not None:
            clip = ClipEmbedding(data=image, mode="image", trained_model=(clip_version == "fine-tuned"))
            embedding = clip.create_embeddings()
        elif text is not None and text.strip() != "":
            clip = ClipEmbedding(data=text, mode="text", trained_model=(clip_version == "fine-tuned"))
            embedding = clip.create_embeddings()
        else:
            return "Load an image or write a description.", None

        embedding_db = embedding_database if clip_version == "fine-tuned" else embedding_database_base
        embedding_type = "clip"

    elif model_name == "DINOv2":
        if image is None:
            return "DINOv2 only supports image search", None
        emb_creator = DinoEmbedding(data=image)
        embedding = emb_creator.create_embeddings()
        embedding_db = embedding_database_dino
        embedding_type = "dino"

    else:
        return "Invalid model selected.", None

    # ----------------- Ricerca -----------------
    searcher = SearchLogic(
        mask_embedding=embedding,
        embedding_db=embedding_db if db_type == "naive" else None,
        faiss_index=faiss_index if (db_type == "faiss" and embedding_type == "clip") else None,
        metadata=faiss_metadata if (db_type == "faiss" and embedding_type == "clip") else None,
        embedding_type=embedding_type
    )

    if db_type == "faiss" and embedding_type == "clip":
        top_preds = searcher.similarity_faiss(top_k=top_k, metric="cosine")
    else:
        sims = searcher.similarity()
        top_preds = searcher.top_predictions(sims, top_k)

    # ----------------- Costruzione risultati -----------------
    results = []
    for i, label, score in top_preds:
        if embedding_type == "clip" and db_type == "faiss":
            db_image_path = faiss_metadata[i]["path"]
        else:
            db_image_path = embedding_db[i]["path"]

        db_image = Image.open(db_image_path).convert("RGB")
        label_str = f"{label} ({score*100:.1f}%)"

        results.append((db_image, label_str))

    return results


"""
def analyze_scene(image, method="MTCNN", clip_version="fine-tuned", interactive=False, bbox=None):
    embedding_db = embedding_database if clip_version=="fine-tuned" else embedding_database_base
    segments = []

    # ----------------- Modalità interattiva -----------------
    if interactive and bbox is not None:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        roi = image.crop((x, y, x + w, y + h))
        if method=="MTCNN":
            faces = mask_maker.detect_faces(roi)
            if faces:
                segments.extend([{"image": f["image"], "bbox": f["bbox"]} for f in faces])
        elif method=="SAM":
            sam_maker = SAMMasks(device=device)
            predictor = sam_maker.load_sam()
            masks = sam_maker.generate_masks(predictor, roi, bbox=[0,0,w,h], mode="auto")
            for m in masks[:5]:
                seg_img = sam_maker.isolate_segment_rgba(roi, m["segmentation"])
                if seg_img:
                    segments.append({"image": seg_img})
    else:
        if method=="MTCNN":
            faces = mask_maker.detect_faces(image)
            if faces:
                segments.extend([{"image": f["image"], "bbox": f.get("bbox",None)} for f in faces])
        elif method=="SAM":
            sam_maker = SAMMasks(device=device)
            predictor = sam_maker.load_sam()
            masks = sam_maker.generate_masks(predictor, image, mode="auto")
            for m in masks[:5]:
                seg_img = sam_maker.isolate_segment_rgba(image, m["segmentation"])
                if seg_img:
                    segments.append({"image": seg_img})

    segments = [s for s in segments if s.get("image") is not None]
    if not segments:
        return "No faces or objects detected", None

    outputs = []
    predictions_json = []
    for seg in segments:
        img = seg["image"]
        clip_embedding = ClipEmbedding(img, mode="image", trained_model=(clip_version=="fine-tuned")).create_embeddings()
        search_logic = SearchLogic(mask_embedding=clip_embedding, embedding_db=embedding_db)

        similarities = search_logic.similarity()
        top_labels = search_logic.character_similarity(similarities, mode="mean")[:3]

        label_str = " / ".join([f"{l[0]} ({l[1]*100:.1f}%)" for l in top_labels])
        outputs.append((img, label_str))
        predictions_json.append({l[0]: float(f"{l[1]*100:.2f}") for l in top_labels})

    return outputs, json.dumps(predictions_json, indent=2)
"""

def analyze_scene(image, method="MTCNN", clip_version="fine-tuned", interactive=False, bbox=None):
    embedding_db = embedding_database if clip_version == "fine-tuned" else embedding_database_base
    segments = []

    # ----------------- Segmentazione -----------------
    if interactive and bbox is not None:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        roi = image.crop((x, y, x + w, y + h))
        if method == "MTCNN":
            faces = mask_maker.detect_faces(roi)
            if faces:
                segments.extend([{"image": f["image"], "bbox": f["bbox"]} for f in faces])
        elif method == "SAM":
            sam_maker = SAMMasks(device=device)
            predictor = sam_maker.load_sam()
            masks = sam_maker.generate_masks(predictor, roi, bbox=[0, 0, w, h], mode="auto")
            for m in masks[:5]:
                seg_img = sam_maker.isolate_segment_rgba(roi, m["segmentation"])
                if seg_img:
                    segments.append({"image": seg_img})
    else:
        if method == "MTCNN":
            faces = mask_maker.detect_faces(image)
            if faces:
                segments.extend([{"image": f["image"], "bbox": f.get("bbox", None)} for f in faces])
        elif method == "SAM":
            sam_maker = SAMMasks(device=device)
            predictor = sam_maker.load_sam()
            masks = sam_maker.generate_masks(predictor, image, mode="auto")
            for m in masks[:5]:
                seg_img = sam_maker.isolate_segment_rgba(image, m["segmentation"])
                if seg_img:
                    segments.append({"image": seg_img})

    segments = [s for s in segments if s.get("image") is not None]
    if not segments:
        return "No faces or objects detected", None

    # ----------------- Embedding + Ricerca -----------------
    outputs = []
    predictions_json = []

    for seg in segments:
        img = seg["image"]

        emb_creator = ClipEmbedding(data=img, mode="image", trained_model=(clip_version == "fine-tuned"))
        clip_embedding = emb_creator.create_embeddings()

        searcher = SearchLogic(
            mask_embedding=clip_embedding,
            embedding_db=embedding_db,
            embedding_type="clip"
        )

        sims = searcher.similarity()
        top_preds = searcher.top_predictions(sims, top_k=3)

        label_str = " / ".join([f"{l[1]} ({l[2]*100:.1f}%)" for l in top_preds])
        outputs.append((img, label_str))
        predictions_json.append({l[1]: float(f"{l[2]*100:.2f}") for l in top_preds})

    return outputs, json.dumps(predictions_json, indent=2)


# ----------------- Gradio UI -----------------
with gr.Blocks(title="Naruto Retrieval & Scene Analysis") as demo:
    gr.Markdown("## 🔍 Naruto Character Retrieval and Scene Analysis")

    # ----------------- Database Query Tab -----------------
    with gr.Tab("Database Query"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Load image", interactive=True)
                text_input = gr.Text(label="Or write a textual description")
                model_selector = gr.Dropdown(
                    choices=["CLIP base", "CLIP fine-tuned", "DINOv2"],
                    value="CLIP fine-tuned",
                    label="Model"
                )
                search_method = gr.Dropdown(
                    choices=["faiss", "naive"],
                    value="faiss",
                    label="Search method"
                )
                top_k = gr.Slider(1,10,value=5,step=1,label="Top K results")
                query_btn = gr.Button("Find in Database")
                output_gallery = gr.Gallery(label="Results", columns=2, height="auto")

                model_selector.change(
                    fn=lambda name: gr.update(visible=(name!="DINOv2")),
                    inputs=[model_selector],
                    outputs=[text_input]
                )

                query_btn.click(
                    fn=query_database,
                    inputs=[image_input, text_input, top_k, model_selector, search_method],
                    outputs=[output_gallery]
                )

    # ----------------- Scene Analysis Tab -----------------
    with gr.Tab("Scene Analysis"):
        with gr.Row():
            interactive_mode = gr.Checkbox(label="Enable interactive bounding box", value=False)
        with gr.Row():
            scene_input = gr.Image(type="pil", label="Load scene")
            scene_editor = gr.ImageEditor(type="pil", label="Select object", visible=False)
        interactive_mode.change(
            lambda checked: (gr.update(visible=not checked), gr.update(visible=checked)),
            inputs=[interactive_mode],
            outputs=[scene_input, scene_editor]
        )
        clip_selector = gr.Dropdown(choices=["base","fine-tuned"], value="fine-tuned", label="CLIP version")
        analyzer_selector = gr.Dropdown(choices=["MTCNN","SAM"], value="MTCNN", label="Scene analyzer")
        analyze_btn = gr.Button("Analyze scene")
        scene_output = gr.Gallery(label="Faces and Predictions", columns=3, show_label=True)
        predictions_json_output = gr.Textbox(label="Predictions JSON", lines=10)

        analyze_btn.click(
            fn=lambda image, editor, interactive, method, clip: analyze_scene(
                editor if interactive else image,
                method,
                clip,
                interactive=interactive,
                bbox=None
            ),
            inputs=[scene_input, scene_editor, interactive_mode, analyzer_selector, clip_selector],
            outputs=[scene_output, predictions_json_output]
        )

demo.queue()
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


