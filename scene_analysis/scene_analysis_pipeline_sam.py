import pickle

from PIL import Image
from scene_analysis.sam_masks import SAMMasks
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
from indexing.search_logic import SearchLogic

"""
import requests
from tqdm import tqdm
import os

os.makedirs("models", exist_ok=True)

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
output = "models/sam_vit_b_01ec64.pth"

response = requests.head(url)
total_size = int(response.headers.get('content-length', 0))

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(output, "wb") as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc="Scaricamento SAM", ncols=100
    ) as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

print(f"Download completato: {output}, dimensione: {os.path.getsize(output)/1e9:.2f} GB")
"""

image_root = 'C:/Users/utente/PycharmProjects/ProgettoDeepLearning/data/train/Gara/3596_jpg.rf.058ac899789bb1e6d9a3b1b7a3a5b4d3.jpg'

mask_maker = SAMMasks()
predictor = mask_maker.load_sam()

image_pil = Image.open(image_root)

plt.figure(figsize=(8, 8))
plt.imshow(image_pil)
plt.axis("off")
plt.title("Immagine originale")
plt.show()

image_masks = mask_maker.generate_masks(predictor, image_pil, points=None, point_labels=None, bbox=None, params=None,
                                        mode="auto")

with open("../embeddings/embedding_database.pkl", "rb") as f:
    embedding_database = pickle.load(f)

print(f"Numero maschere trovate: {len(image_masks)}")

for i, mask_dict in enumerate(image_masks):
    mask = mask_dict["segmentation"]

    plt.figure(figsize=(8, 8))
    plt.imshow(image_pil)
    plt.imshow(mask, alpha=0.5, cmap="jet")
    plt.axis("off")
    plt.title(f"Maschera {i + 1} (area={mask_dict['area']})")
    plt.show()

    segment_img = mask_maker.isolate_segment_rgba(image_pil, mask)
    mask_embedding = mask_maker.create_clip_embedding(segment_img)

    search_engine = SearchLogic(mask_embedding, embedding_database)

    mask_similarities = search_engine.similarity()
    character_similarities = search_engine.character_similarity(mask_similarities)

    print(f"Top predictions: {search_engine.top_predictions(mask_similarities, 10)}")
    print(f"Top character predictions (%): {search_engine.top_predictions(character_similarities, 10)}")