import pickle
import faiss
import gradio as gr
import torch
from PIL import Image
import json

from indexing.clip_embedding import ClipEmbedding
from indexing.dino_embedding import DinoEmbedding
from scene_analysis.mtcnn_masks import FaceMasks
from scene_analysis.sam_masks import SAMMasks
from indexing.search_logic import SearchLogic

# ----------------- Caricamento di tutti i database -----------------
dbs = {}  # Dizionario per contenere tutti i database caricati (Python list o FAISS)

# CLIP base
with open("../embeddings/clip_python_list.pkl", "rb") as f:
    dbs["CLIP base - Python list"] = {"type": "naive", "db": pickle.load(f)}  # Caricamento database "naive"
faiss_index = faiss.read_index("../embeddings/clip_faiss.index")  # Caricamento index FAISS
with open("../embeddings/clip_faiss_metadata.pkl", "rb") as f:
    dbs["CLIP base - FAISS"] = {"type": "faiss", "index": faiss_index, "metadata": pickle.load(f)}  # Salva FAISS con metadata

# CLIP fine-tuned
with open("../embeddings/clip_finetuned_python_list.pkl", "rb") as f:
    dbs["CLIP fine-tuned - Python list"] = {"type": "naive", "db": pickle.load(f)}
faiss_index = faiss.read_index("../embeddings/clip_finetuned_faiss.index")
with open("../embeddings/clip_faiss_metadata.pkl", "rb") as f:
    dbs["CLIP fine-tuned - FAISS"] = {"type": "faiss", "index": faiss_index, "metadata": pickle.load(f)}

# DINO
with open("../embeddings/dino_python_list.pkl", "rb") as f:
    dbs["DINOv2 - Python list"] = {"type": "naive", "db": pickle.load(f)}
faiss_index = faiss.read_index("../embeddings/dino_faiss.index")
with open("../embeddings/dino_faiss_metadata.pkl", "rb") as f:
    dbs["DINOv2 - FAISS"] = {"type": "faiss", "index": faiss_index, "metadata": pickle.load(f)}

# ----------------- Configurazioni -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"  # Usa GPU se disponibile
mask_maker = FaceMasks(device=device)  # Inizializza MTCNN per rilevamento volti

# ----------------- Funzione query database -----------------
def query_database(image=None, text=None, top_k=5, model_choice="CLIP fine-tuned - FAISS"):
    """
    Funzione per fare query sui database caricati.
    Supporta query da immagine o testo (solo CLIP).
    """
    config = dbs[model_choice]  # Configurazione scelta dall'utente
    embedding_type = "clip" if model_choice.startswith("CLIP") else "dino"
    clip_version = "fine-tuned" if "fine-tuned" in model_choice else "base"

    # ---- Creazione embedding per input ----
    if embedding_type == "clip":
        if image is not None:  # Se l'utente ha caricato un'immagine
            clip = ClipEmbedding(
                data=image,
                mode="image",
                trained_model=(clip_version == "fine-tuned")
            )
            embedding = clip.create_embeddings()
        elif text is not None and text.strip() != "":  # Query testuale
            clip = ClipEmbedding(
                data=text,
                mode="text",
                trained_model=(clip_version == "fine-tuned")
            )
            embedding = clip.create_embeddings()
        else:
            return "⚠️ Provide an image or text.", None

    elif embedding_type == "dino":  # DINO supporta solo immagini
        if image is None:
            return "⚠️ DINOv2 only supports image queries.", None
        dino = DinoEmbedding(data=image)
        embedding = dino.create_embeddings()

    else:
        return "⚠️ Unsupported model.", None

    # ---- Ricerca nei database ----
    searcher = SearchLogic(
        mask_embedding=embedding,
        embedding_db=config.get("db"),
        faiss_index=config.get("index"),
        metadata=config.get("metadata"),
        embedding_type=embedding_type
    )

    if config["type"] == "faiss":
        # Ricerca tramite FAISS
        if embedding_type == "clip":
            top_preds = searcher.similarity_faiss(top_k=top_k, metric="cosine")
        elif embedding_type == "dino":
            top_preds = searcher.similarity_faiss_dino(top_k=top_k)
        else:
            return "⚠️ Invalid FAISS model type.", None
    else:
        # Ricerca naive tramite lista Python
        sims = searcher.similarity()
        top_preds = searcher.top_predictions(sims, top_k)

    # ---- Preparazione output immagini + label ----
    results = []
    for i, label, score in top_preds:
        if config["type"] == "faiss":
            db_image_path = config["metadata"][i]["path"]  # Path immagine nel FAISS
        else:
            db_image_path = config["db"][i]["path"]  # Path immagine nella lista

        db_image = Image.open(db_image_path).convert("RGB")
        results.append((db_image, f"{label} ({score*100:.1f}%)"))

    return results


def analyze_scene(image, method="MTCNN", model_choice="CLIP fine-tuned - FAISS", interactive=False, bbox=None, top_k=3):
    """
    Funzione per analizzare una scena complessa:
    - Rileva volti con MTCNN o segmenti con SAM
    - Genera embedding per ogni segmento
    - Effettua ricerca top-K per ogni segmento
    """
    config = dbs[model_choice]
    clip_version = "fine-tuned" if "fine-tuned" in model_choice else "base"

    segments = []

    # ----------------- Segmentazione -----------------
    if interactive and bbox is not None:  # Se l'utente ha selezionato bounding box
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        target_image = image.crop((x, y, x + w, y + h))
    else:
        target_image = image

    if method == "MTCNN":  # Segmentazione volti
        faces = mask_maker.detect_faces(target_image)
        if faces:
            segments.extend([{"image": f["image"]} for f in faces])
    elif method == "SAM":  # Segmentazione oggetti con SAM
        sam_maker = SAMMasks(device=device)
        predictor = sam_maker.load_sam()
        masks = sam_maker.generate_masks(
            predictor=predictor,
            image=target_image,
            points=None,
            point_labels=None,
            bbox=None,
            params=None,
            mode="auto"
        )
        for m in masks[:5]:
            seg_img = sam_maker.isolate_segment_rgba(target_image, m["segmentation"])
            if seg_img:
                segments.append({"image": seg_img})

    segments = [s for s in segments if s.get("image") is not None]
    if not segments:
        return "No faces or objects detected"

    # ----------------- Embedding + Ricerca CLIP-FAISS -----------------
    outputs = []
    for seg in segments:
        img = seg["image"]

        # Creazione embedding CLIP
        emb_creator = ClipEmbedding(data=img, mode="image", trained_model=(clip_version=="fine-tuned"))
        embedding = emb_creator.create_embeddings()

        # Ricerca FAISS
        searcher = SearchLogic(
            mask_embedding=embedding,
            faiss_index=config.get("index"),
            metadata=config.get("metadata"),
            embedding_type="clip"
        )
        top_preds = searcher.similarity_faiss(top_k=top_k, metric="cosine")

        # Prepara stringa con label multiple sotto immagine
        label_str = " / ".join([f"{l[1]} ({l[2]*100:.1f}%)" for l in top_preds])
        outputs.append((img, label_str))

    return outputs


# ----------------- Gradio UI -----------------
with gr.Blocks(title="Naruto Retrieval & Scene Analysis") as demo:
    gr.Markdown("## 🔍 Naruto Character Retrieval and Scene Analysis")

    # ----------------- Database Query Tab -----------------
    with gr.Tab("Database Query"):
        with gr.Row():
            # Colonna sinistra = input
            with gr.Column(scale=1, min_width=300):
                image_input = gr.Image(type="pil", label="Upload image", height=250)
                text_input = gr.Text(label="Or enter description")
                model_selector = gr.Dropdown(
                    choices=list(dbs.keys()),
                    value="CLIP fine-tuned - FAISS",
                    label="Model + DB type"
                )
                top_k = gr.Slider(1, 10, value=5, step=1, label="Top K")
                query_btn = gr.Button("Find", variant="primary")

            # Colonna destra = risultati
            with gr.Column(scale=2, min_width=500):
                output_gallery = gr.Gallery(
                    label="Results", columns=2, height=500, show_label=True
                )

        # Nasconde campo testo se modello non supporta query testuale
        model_selector.change(
            fn=lambda name: gr.update(visible=not name.startswith("DINOv2")),
            inputs=[model_selector],
            outputs=[text_input]
        )

        query_btn.click(
            fn=query_database,
            inputs=[image_input, text_input, top_k, model_selector],
            outputs=[output_gallery]
        )

    # ----------------- Scene Analysis Tab -----------------
    with gr.Tab("Scene Analysis"):
        with gr.Row():
            with gr.Column(scale=1, min_width=400):
                interactive_mode = gr.Checkbox(label="Enable interactive bounding box", value=False)
                scene_input = gr.Image(type="pil", label="Load scene", height=300)
                scene_editor = gr.ImageEditor(type="pil", label="Select object", visible=False, height=300)
                analyzer_selector = gr.Dropdown(choices=["MTCNN", "SAM"], value="MTCNN", label="Scene analyzer")
                analyze_btn = gr.Button("Analyze scene", variant="primary")

            with gr.Column(scale=2, min_width=500):
                scene_output = gr.Gallery(
                    label="Faces and Predictions", columns=3, height=500, show_label=True
                )

        # Mostra scene_input o scene_editor in base alla checkbox
        interactive_mode.change(
            lambda checked: (gr.update(visible=not checked), gr.update(visible=checked)),
            inputs=[interactive_mode],
            outputs=[scene_input, scene_editor]
        )

        analyze_btn.click(
            fn=lambda image, editor, interactive, method: analyze_scene(
                image=editor if interactive else image,
                method=method,
                model_choice="CLIP fine-tuned - FAISS",  # Forzato a CLIP-FAISS
                interactive=interactive,
                bbox=None
            ),
            inputs=[scene_input, scene_editor, interactive_mode, analyzer_selector],
            outputs=[scene_output]
        )

demo.queue()  # Attiva la coda di richieste Gradio
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)  # Avvia app Gradio
