import os
import pickle

import gradio as gr
import torch.cuda
from PIL import Image

from indexing.clip_embedding import ClipEmbedding
from scene_analysis.mtcnn_masks import FaceMasks
from scene_analysis.search_logic import SearchLogic

with open("../embeddings/embedding_database_trained.pkl", "rb") as f:
    embedding_database = pickle.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
mask_maker = FaceMasks(device=device)


def query_database(image=None, text=None, top_k=5):
    if image is not None:
        clip_embedding = mask_maker.create_clip_embedding(image)
    elif text is not None and text.strip() != "":
        clip = ClipEmbedding(data=text, mode="text")
        clip_embedding = clip.create_embeddings()

    else:
        return "Load an image or write a description.", None

    search_logic = SearchLogic(mask_embedding=clip_embedding,
                               embedding_db=embedding_database)

    similarities = search_logic.similarity()
    top_preds = search_logic.top_predictions(similarities, top_k)

    results = []

    for idx, (i, character, score) in enumerate(top_preds):
        db_image_path = embedding_database[i]["path"]
        db_image = Image.open(db_image_path).convert("RGB")
        results.append((db_image, f"{character} ({score:.2f}"))

    return results


def analyze_scene(image):
    faces = mask_maker.detect_faces(image)

    if faces is None:
        return "No face detected", None

    outputs = []
    for i, face in enumerate(faces):
        clip_embedding = mask_maker.create_clip_embedding(face["image"])
        search_logic = SearchLogic(mask_embedding=clip_embedding,
                                   embedding_db=embedding_database)
        similarities = search_logic.similarity()
        top_pred = search_logic.top_predictions(similarities, 1)[0]
        outputs.append((face["image"], f"Prediction: {top_pred[1]} ({top_pred[2]:.2f}"))

    return outputs


with gr.Blocks(title="Naruto Retrieval & Scene Analysis") as demo:
    gr.Markdown("Naruto Character Retrieval and Scene Analysis")

    with gr.Tab("Database Query"):
        with gr.Row():
            image_input = gr.Image(type="pil", label="Load image:", interactive=True)
            text_input = gr.Text(label="Or write a textual description")

        top_k = gr.Slider(1, 10, value=5, step=1, label="Top K results")
        query_btn = gr.Button("Find in the Database")
        output_gallery = gr.Gallery(label="Results")

        query_btn.click(
            fn=query_database,
            inputs=[image_input, text_input, top_k],
            outputs=[output_gallery]
        )

    with gr.Tab("Scene Analysis"):
        scene_input = gr.Image(type="pil", label="Load scene", interactive=True)
        analyze_btn = gr.Button("Analyze scene")
        scene_output = gr.Gallery(label="Faces and Predictions")

        analyze_btn.click(
            fn=analyze_scene,
            inputs=[scene_input],
            outputs=[scene_output]
        )


demo.queue()

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)