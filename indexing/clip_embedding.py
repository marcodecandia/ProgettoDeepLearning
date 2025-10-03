from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from tqdm import tqdm

class ClipEmbedding:
    """
    Classe per creare embedding CLIP da immagini o testo.
    Supporta sia il modello base che un modello fine-tuned caricato localmente.
    """
    def __init__(self, data, mode="image", trained_model=True):
        """
        Inizializza la classe.

        Args:
            data: immagine PIL, lista di immagini, stringa o lista di stringhe.
            mode: 'image' o 'text', specifica il tipo di input.
            trained_model: se True, carica un modello CLIP fine-tuned.
        """
        self.data = data
        self.mode = mode
        self.trained_model = trained_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Usa GPU se disponibile

        # Caricamento modello e processor CLIP
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def create_embeddings(self, batch_size=8):
        """
        Crea gli embedding CLIP per immagini o testo in batch.

        Args:
            batch_size: numero di elementi da processare insieme (per efficienza).

        Returns:
            Numpy array con gli embedding calcolati.
        """

        # Carica i pesi del modello fine-tuned se specificato
        if self.trained_model:
            self.model.load_state_dict(torch.load("../models/trained_clip_model.pth", map_location="cpu"))

        # -----------------------
        # Modalità testo
        # -----------------------
        if self.mode == "text":
            # Gestione input singolo o lista
            if isinstance(self.data, str):
                texts = [self.data]
                single_text = True  # flag per sapere se ritornare singolo embedding
            else:
                texts = self.data
                single_text = False

            all_embeddings = []

            # Elaborazione a batch
            for i in tqdm(range(0, len(texts), batch_size), desc="Creating text embedding"):
                batch = texts[i:i + batch_size]

                # Tokenizzazione e trasformazione in tensori
                inputs = self.processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                # Disattiva il calcolo del gradiente per velocizzare
                with torch.no_grad():
                    embeddings = self.model.get_text_features(**inputs)
                    # Normalizzazione L2
                    embeddings /= embeddings.norm(p=2, dim=-1, keepdim=True)

                embeddings = embeddings.cpu().numpy()  # Trasferisci su CPU

                all_embeddings.extend(embeddings)

            return np.array(all_embeddings) if not single_text else all_embeddings[0]

        # -----------------------
        # Modalità immagine
        # -----------------------
        elif self.mode == "image":
            # Gestione input singolo o lista
            if isinstance(self.data, Image.Image):
                images = [self.data]
                single_image = True
            else:
                # Lista di dizionari con chiave 'image'
                images = [item["image"] for item in self.data]
                single_image = False

            all_embeddings = []

            # Elaborazione a batch
            for i in tqdm(range(0, len(images), batch_size), desc="Creazione embedding"):
                batch = images[i:i + batch_size]

                # Preprocessamento batch di immagini
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    embeddings = self.model.get_image_features(**inputs)
                    # Normalizzazione L2
                    embeddings /= embeddings.norm(p=2, dim=-1, keepdim=True)

                embeddings = embeddings.cpu().numpy()

                # Salvataggio embedding nel caso di input lista di dizionari
                if single_image:
                    all_embeddings.extend(embeddings)
                else:
                    for item, emb in zip(self.data[i:i+batch_size], embeddings):
                        item["clip_embedding"] = emb  # salva embedding nel dizionario
                        all_embeddings.append(emb)

            return np.array(all_embeddings) if not single_image else all_embeddings[0]

        else:
            # Errore se la modalità non è valida
            raise ValueError(f"Invalid mode: {self.mode}. Use 'image' or 'text'")
