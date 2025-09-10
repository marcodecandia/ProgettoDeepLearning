import pickle
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from facenet_pytorch import MTCNN
from indexing.clip_embedding import ClipEmbedding
from scene_analysis.search_logic import SearchLogic
import matplotlib

from scene_analysis.mtcnn_masks import FaceMasks
from scene_analysis.search_logic import SearchLogic

matplotlib.use("TkAgg")


# --- Config ---
image_root = 'C:/Users/utente/PycharmProjects/ProgettoDeepLearning/data/test/Naruto/43596_jpg.rf.e68f4774e18d30d2e7b663288b7bfae9.jpg'
embedding_db_path = "../indexing/embedding_database.pkl"

# --- Carica immagine ---
image_pil = Image.open(image_root)
plt.figure(figsize=(8, 8))
plt.imshow(image_pil)
plt.axis("off")
plt.title("Immagine originale")
plt.show()

# --- Carica database embedding ---
with open(embedding_db_path, "rb") as f:
    embedding_database = pickle.load(f)

# --- Inizializza MTCNN ---
device = "cuda" if torch.cuda.is_available() else "cpu"
mask_maker = FaceMasks(device=device)

# --- Rileva volti ---
faces = mask_maker.detect_faces(image_pil)
if faces is not None:
    for i, face in enumerate(faces):
        plt.figure(figsize=(4, 4))
        plt.imshow(face["image"])
        plt.axis("off")
        plt.title(f"Volto {i + 1}")
        plt.show()


# --- Crea embedding CLIP per ogni volto e confronta ---
for i, face in enumerate(faces):
    clip_embedding = mask_maker.create_clip_embedding(face["image"])

    search_logic = SearchLogic(clip_embedding, embedding_database)

    similarities = search_logic.similarity()

    top_preds = search_logic.top_predictions(similarities, 10)
    character_preds = search_logic.character_similarity(similarities)
    print(f"Volto {i+1} - Top predictions: {top_preds}")
    print(f"Top character predictions (%): {character_preds}")

