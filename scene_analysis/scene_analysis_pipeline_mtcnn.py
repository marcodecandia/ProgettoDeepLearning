import os
import pickle

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from facenet_pytorch import MTCNN
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from indexing.clip_embedding import ClipEmbedding
from scene_analysis.search_logic import SearchLogic
import matplotlib

from scene_analysis.mtcnn_masks import FaceMasks
from scene_analysis.search_logic import SearchLogic

matplotlib.use("TkAgg")


class NarutoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for idx, cls in enumerate(self.classes):
            class_folder = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_folder):
                self.image_paths.append(os.path.join(class_folder, img_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]
        class_name = self.classes[label]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, class_name


embedding_db_path = "../embeddings/embedding_database_trained.pkl"
test_data = NarutoDataset("../data/test")
test_loader = DataLoader(test_data)

with open(embedding_db_path, "rb") as f:
    embedding_database = pickle.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
mask_maker = FaceMasks(device=device)

correct = 0
total = 0

for image, text in tqdm(test_loader):

    image_pil = Image.fromarray(image.squeeze(0).numpy())

    """
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image_pil)
    plt.axis("off")
    plt.title("Immagine originale")
    plt.show()
    """
    faces = mask_maker.detect_faces(image_pil)

    if faces is None:
        continue
    """
    if faces is not None:
        for i, face in enumerate(faces):
            plt.figure(figsize=(4, 4))
            plt.imshow(face["image"])
            plt.axis("off")
            plt.title(f"Volto {i + 1}")
            plt.show()
    """
    for i, face in enumerate(faces):
        clip_embedding = mask_maker.create_clip_embedding(face["image"])

        search_logic = SearchLogic(clip_embedding, embedding_database)
        similarities = search_logic.similarity()

        top_pred = search_logic.top_predictions(similarities, 1)
        character_pred = top_pred[0][1]
        print(character_pred)
        print(f"Volto {i + 1} - Top prediction: {top_pred}")
        print(f"Top character prediction (%): {character_pred}")

        if character_pred == text[0]:
            correct += 1
        total += 1

accuracy = correct / total if total > 0 else 0
print(f"Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")




"""
# --- Config ---
image_root = 'C:/Users/utente/PycharmProjects/ProgettoDeepLearning/data/test/Naruto/43596_jpg.rf.e68f4774e18d30d2e7b663288b7bfae9.jpg'
embedding_db_path = "../embeddings/embedding_database_trained.pkl"

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

"""
