import pickle
from PIL import Image
from scene_analysis.sam_masks import SAMMasks
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
from indexing.search_logic import SearchLogic


# -----------------------------
# Caricamento immagine di test
# -----------------------------
image_root = 'C:/Users/utente/PycharmProjects/ProgettoDeepLearning/data/train/Gara/3596_jpg.rf.058ac899789bb1e6d9a3b1b7a3a5b4d3.jpg'

mask_maker = SAMMasks()
predictor = mask_maker.load_sam()

image_pil = Image.open(image_root)

# Mostra immagine originale
plt.figure(figsize=(8, 8))
plt.imshow(image_pil)
plt.axis("off")
plt.title("Immagine originale")
plt.show()

# -----------------------------
# Generazione maschere con SAM
# -----------------------------
image_masks = mask_maker.generate_masks(
    predictor,
    image_pil,
    points=None,
    point_labels=None,
    bbox=None,
    params=None,
    mode="auto"
)

# Carica database embeddings CLIP
with open("../embeddings/embedding_database_clipbase_list.pkl", "rb") as f:
    embedding_database = pickle.load(f)

print(f"Numero maschere trovate: {len(image_masks)}")

# -----------------------------
# Loop sulle maschere generate
# -----------------------------
for i, mask_dict in enumerate(image_masks):
    mask = mask_dict["segmentation"]

    # Visualizza maschera sovrapposta all’immagine
    plt.figure(figsize=(8, 8))
    plt.imshow(image_pil)
    plt.imshow(mask, alpha=0.5, cmap="jet")
    plt.axis("off")
    plt.title(f"Maschera {i + 1} (area={mask_dict['area']})")
    plt.show()

    # Isola il segmento e calcola embedding CLIP
    segment_img = mask_maker.isolate_segment_rgba(image_pil, mask)
    mask_embedding = mask_maker.create_clip_embedding(segment_img)

    # -----------------------------
    # Ricerca di similarità
    # -----------------------------
    search_engine = SearchLogic(mask_embedding, embedding_database)

    mask_similarities = search_engine.similarity()
    character_similarities = search_engine.character_similarity(mask_similarities)

    print(f"Top predictions: {search_engine.top_predictions(mask_similarities, 10)}")
    print(f"Top character predictions (%): {search_engine.top_predictions(character_similarities, 10)}")
