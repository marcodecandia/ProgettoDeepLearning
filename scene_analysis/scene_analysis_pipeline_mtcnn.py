import os
import pickle

import cv2
import faiss
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib

from scene_analysis.mtcnn_masks import FaceMasks
from indexing.search_logic import SearchLogic

matplotlib.use("TkAgg")


class NarutoDataset(Dataset):
    """
    Dataset custom per immagini organizzate in cartelle (una per classe).
    Restituisce coppia (immagine, etichetta).
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Carica i percorsi delle immagini e associa etichette
        for idx, cls in enumerate(self.classes):
            class_folder = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_folder):
                self.image_paths.append(os.path.join(class_folder, img_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Legge immagine con OpenCV e la converte in RGB
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]
        class_name = self.classes[label]

        # Applica trasformazioni se definite
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, class_name


# -----------------------------
# Caricamento embeddings e indice FAISS
# -----------------------------
embedding_db_path = "../embeddings/embedding_database_cliptrained_list.pkl"
test_data = NarutoDataset("../data/test")
test_loader = DataLoader(test_data)

with open(embedding_db_path, "rb") as f:
    embedding_database = pickle.load(f)

faiss_index = faiss.read_index("../embeddings/embedding_database_cliptrained_faiss.faiss")
with open("../embeddings/clip_embeddings_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Inizializza rilevatore di volti
device = "cuda" if torch.cuda.is_available() else "cpu"
mask_maker = FaceMasks(device=device)

correct = 0
total = 0

# -----------------------------
# Loop di test
# -----------------------------
for image, text in tqdm(test_loader):
    # Converte immagine in formato PIL
    image_pil = Image.fromarray(image.squeeze(0).numpy())

    # Rilevamento volti
    faces = mask_maker.detect_faces(image_pil)
    if faces is None:
        continue

    # Per ogni volto trovato → calcola embedding e cerca nel database
    for i, face in enumerate(faces):
        clip_embedding = mask_maker.create_clip_embedding(face["image"])

        search_logic = SearchLogic(mask_embedding=clip_embedding, faiss_index=faiss_index, metadata=metadata)
        similarities = search_logic.similarity_faiss(10, "cosine")

        # Top-1 predizione
        top_pred = search_logic.top_predictions(similarities, 1)
        character_pred = top_pred[0][1]

        print(character_pred)
        print(f"Volto {i + 1} - Top prediction: {top_pred}")

        if character_pred == text[0]:
            correct += 1
        total += 1

# -----------------------------
# Calcolo accuratezza finale
# -----------------------------
accuracy = correct / total if total > 0 else 0
print(f"Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")




