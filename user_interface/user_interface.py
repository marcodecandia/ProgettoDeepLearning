import pickle

import faiss
import gradio as gr
import torch.cuda
from PIL import Image

from indexing.clip_embedding import ClipEmbedding
from indexing.dino_embedding import DinoEmbeddding
from scene_analysis.mtcnn_masks import FaceMasks
from scene_analysis.sam_masks import SAMMasks
from scene_analysis.search_logic import SearchLogic

with open("../embeddings/embedding_database.pkl", "rb") as f:
    embedding_database_base = pickle.load(f)

with open("../embeddings/embedding_database_trained.pkl", "rb") as f:
    embedding_database = pickle.load(f)

with open("../embeddings/embedding_database_dino.pkl", "rb") as f:
    embedding_database_dino = pickle.load(f)

faiss_index = faiss.read_index("../embeddings/embedding_database_trained_faiss.faiss")
with open("../embeddings/clip_embeddings_metadata.pkl", "rb") as f:
    faiss_metadata = pickle.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
mask_maker = FaceMasks(device=device)


def query_database(image=None, text=None, top_k=5, model_name="CLIP fine-tuned", db_type="faiss"):
    if model_name.startswith("CLIP"):
        clip_version = "fine-tuned" if model_name == "CLIP fine-tuned" else "base"
        embedding_db = embedding_database if clip_version == "fine-tuned" else embedding_database_base

        if image is not None:
            clip = ClipEmbedding(data=image)
            embedding = clip.create_embeddings(trained_model=(clip_version == "fine-tuned"))
        elif text is not None and text.strip() != "":
            clip = ClipEmbedding(data=text, mode="text")
            embedding = clip.create_embeddings(trained_model=(clip_version == "fine-tuned"))

        else:
            return "Load an image or write a description.", None

    elif model_name == "DINOv2":
        if image is None:
            return "DINOv2 only supports image search", None
        embedding_db = embedding_database_dino
        dino = DinoEmbeddding(image)
        embedding = dino.create_embeddings()

    search_logic = SearchLogic(mask_embedding=embedding,
                               embedding_db=embedding_db,
                               faiss_index=faiss_index if model_name.startswith("CLIP") else None,
                               metadata=faiss_metadata if model_name.startswith("CLIP") else None)

    results = []

    if db_type == "faiss" and model_name.startswith("CLIP"):
        similarities = search_logic.similarity_faiss(top_k=top_k,
                                                     metric="cosine")
        for i, label, score in similarities:
            db_image_path = faiss_metadata[i]["path"]
            db_image = Image.open(db_image_path).convert("RGB")
            results.append((db_image, f"{label} ({score:.2f})"))
            print("Using faiss")

    elif db_type == "naive":
        similarities = search_logic.similarity()
        top_preds = search_logic.top_predictions(similarities, top_k)

        for i, label, score in top_preds:
            db_image_path = embedding_db[i]["path"]
            db_image = Image.open(db_image_path).convert("RGB")
            results.append((db_image, f"{label} ({score:.2f})"))
            print("Using naive")

    return results


def analyze_scene(image, method="MTCNN", clip_version="fine-tuned", interactive=False, bbox=None):
    embedding_db = embedding_database if clip_version == "fine-tuned" else embedding_database_base
    segments = []

    # ----------------- modalità interattiva -----------------
    if interactive and bbox is not None:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        x1, y1 = x + w, y + h
        roi = image.crop((x, y, x1, y1))

        if method == "MTCNN":
            mask_maker = FaceMasks(device=device)
            faces = mask_maker.detect_faces(roi)
            if faces:
                for f in faces:
                    if isinstance(f, dict) and "image" in f:
                        segments.append(f)
                    else:
                        segments.append({"image": f})

        elif method == "SAM":
            sam_maker = SAMMasks(device=device)
            predictor = sam_maker.load_sam()
            masks = sam_maker.generate_masks(
                predictor=predictor,
                image=image,
                bbox=[x, y, x1, y1],
                points=None,
                params=None,
                point_labels=None,
                mode="auto"
            )
            for m in masks[:5]:
                seg_img = sam_maker.isolate_segment_rgba(image, m["segmentation"])
                if seg_img:
                    segments.append({"image": seg_img})

    # ----------------- modalità standard -----------------
    else:
        if method == "MTCNN":
            mask_maker = FaceMasks(device=device)
            faces = mask_maker.detect_faces(image)
            if faces:
                for f in faces:
                    if isinstance(f, dict) and "image" in f:
                        segments.append(f)
                    else:
                        segments.append({"image": f})

        elif method == "SAM":
            sam_maker = SAMMasks(device=device)
            predictor = sam_maker.load_sam()
            masks = sam_maker.generate_masks(
                predictor=predictor,
                image=image,
                bbox=None,
                points=None,
                params=None,
                point_labels=None,
                mode="auto"
            )
            for m in masks[:5]:
                seg_img = sam_maker.isolate_segment_rgba(image, m["segmentation"])
                if seg_img:
                    segments.append({"image": seg_img})

    # ----------------- nessun segmento rilevato -----------------
    if not segments:
        return "No face/object detected", None

    # ----------------- calcolo embeddings e predizioni -----------------
    outputs = []
    for seg in segments:
        img = seg.get("image", None)
        if img is None:
            continue

        clip_embedding = ClipEmbedding(img).create_embeddings()
        search_logic = SearchLogic(mask_embedding=clip_embedding,
                                   embedding_db=embedding_db)
        similarities = search_logic.similarity()
        top_pred = search_logic.top_predictions(similarities, 1)[0]
        outputs.append((img, f"Prediction: {top_pred[1]} ({top_pred[2]:.2f})"))

    if not outputs:
        return "No valid segments found"

    return outputs



with gr.Blocks(title="Naruto Retrieval & Scene Analysis") as demo:
    gr.Markdown("Naruto Character Retrieval and Scene Analysis")

    with gr.Tab("Database Query"):
        with gr.Row():
            image_input = gr.Image(type="pil", label="Load image:", interactive=True)
            text_input = gr.Text(label="Or write a textual description")

        model_selector = gr.Dropdown(choices=["CLIP base", "CLIP fine-tuned", "DINOv2"],
                                     value="CLIP fine-tuned",
                                     label="Select embedding model")
        search_method = gr.Dropdown(choices=["faiss", "naive"],
                                    value="faiss",
                                    label="Search method")
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top K results")
        query_btn = gr.Button("Find in the Database")
        output_gallery = gr.Gallery(label="Results")


        def remove_text_input(model_name):
            if model_name == "DINOv2":
                return gr.update(visible=False,
                                 placeholder="Text search disabled")
            else:
                return gr.update(visible=True,
                                 placeholder="Or write a textual description")

        model_selector.change(
            fn=remove_text_input,
            inputs=[model_selector],
            outputs=[text_input]
        )

        query_btn.click(
            fn=query_database,
            inputs=[image_input, text_input, top_k, model_selector, search_method],
            outputs=[output_gallery]
        )

    with gr.Tab("Scene Analysis"):
        with gr.Row():
            interactive_mode = gr.Checkbox(label="Enable interactive bounding box mode", value=False)

        with gr.Row():
            # Caricamento standard
            scene_input = gr.Image(type="pil", label="Load scene", visible=True)

            # Editor interattivo (inizialmente nascosto)
            scene_editor = gr.ImageEditor(type="pil", label="Select object", visible=False)

        # Quando spunti la checkbox, nasconde l'immagine normale e mostra l’editor
        interactive_mode.change(
            lambda checked: (
                gr.update(visible=not checked),  # scene_input
                gr.update(visible=checked)  # scene_editor
            ),
            inputs=interactive_mode,
            outputs=[scene_input, scene_editor]
        )

        clip_selector = gr.Dropdown(choices=["base", "fine-tuned"], value="fine-tuned", label="CLIP version")
        analyzer_selector = gr.Dropdown(choices=["MTCNN", "SAM"], value="MTCNN", label="Scene analyzer")

        analyze_btn = gr.Button("Analyze scene")
        scene_output = gr.Gallery(label="Faces and Predictions")


        # Risolve l'input corretto da passare a analyze_scene
        def resolve_inputs(image, editor, interactive, method, clip):
            if interactive:
                # estrai l'immagine dal dizionario restituito da ImageEditor
                pil_image = editor.get("image") if isinstance(editor, dict) else editor
                return analyze_scene(pil_image, method, clip, interactive=True)
            else:
                return analyze_scene(image, method, clip, interactive=False)


        analyze_btn.click(
            fn=resolve_inputs,
            inputs=[scene_input, scene_editor, interactive_mode, analyzer_selector, clip_selector],
            outputs=scene_output
        )

demo.queue()

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)