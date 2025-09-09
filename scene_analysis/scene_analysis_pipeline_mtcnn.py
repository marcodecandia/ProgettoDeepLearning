import pickle
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from facenet_pytorch import MTCNN
from indexing.clip_embedding import ClipEmbedding
import matplotlib
matplotlib.use("TkAgg")

# --- Config ---
image_root = 'C:/Users/utente/PycharmProjects/ProgettoDeepLearning/data/train/Gara/3596_jpg.rf.058ac899789bb1e6d9a3b1b7a3a5b4d3.jpg'
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
mtcnn = MTCNN(keep_all=True, device=device)

# --- Rileva volti ---
boxes, probs = mtcnn.detect(image_pil)
faces = []
if boxes is not None:
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = [int(b) for b in box]
        face_crop = image_pil.crop((x0, y0, x1, y1))
        faces.append(face_crop)

        # Visualizza il volto
        plt.figure(figsize=(4, 4))
        plt.imshow(face_crop)
        plt.axis("off")
        plt.title(f"Volto {i+1}")
        plt.show()

# --- Crea embedding CLIP per ogni volto e confronta ---
for i, face in enumerate(faces):
    clip = ClipEmbedding([{"image": face}])
    face_embedding = clip.create_embeddings(batch_size=1)[0]

    # Confronta con il database
    similarities = []
    for entry in embedding_database:
        db_emb = np.array(entry["clip_embedding"]).reshape(1, -1)
        # assicurati che face_embedding abbia la stessa shape (1, 512)
        face_emb = face_embedding.reshape(1, -1)
        cos_sim = np.dot(face_emb, db_emb.T)[0][0]
        similarities.append((entry["label"], cos_sim))

    # Ordina e prendi i top 5
    top_preds = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    print(f"Volto {i+1} - Top predictions: {top_preds}")