import pickle

import cv2
import torch
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib

from scene_analysis.search_logic import SearchLogic

matplotlib.use("TkAgg")

from scene_analysis.mtcnn_masks import FaceMasks

image1_root = './data/0c9ce4e037546965d6b1f3807e9f8f549a113d32066b2bdb22ada5d179c0d89a.jpg'
image2_root = './data/personaggi-anime-naruto.jpg'

embedding_db_path = "./embeddings/embedding_database_trained.pkl"

with open(embedding_db_path, "rb") as f:
    embedding_database = pickle.load(f)


image = cv2.imread(image2_root)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

device = "cuda" if torch.cuda.is_available() else "cpu"
mask_maker = FaceMasks(device=device)

image_pil = Image.fromarray(image)

plt.figure(figsize=(8, 8))
plt.imshow(image_pil)
plt.axis("off")
plt.title("Immagine originale")
plt.show()

faces = mask_maker.detect_faces(image_pil)


for i, face in enumerate(faces):
    plt.figure(figsize=(4, 4))
    plt.imshow(face["image"])
    plt.axis("off")
    plt.title(f"Volto {i + 1}")
    plt.show()

for i, face in enumerate(faces):
    clip_embedding = mask_maker.create_clip_embedding(face["image"])

    search_logic = SearchLogic(clip_embedding, embedding_database)
    similarities = search_logic.similarity()

    top_pred = search_logic.top_predictions(similarities, 1)
    character_pred = top_pred[0][1]
    print(character_pred)
    print(f"Volto {i + 1} - Top prediction: {top_pred}")
    print(f"Top character prediction (%): {character_pred}")


