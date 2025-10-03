import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class DinoEmbedding:
    """
    Classe per creare embedding DINOv2 da immagini.
    Supporta input singoli o liste di immagini.
    """
    def __init__(self, data):
        """
        Inizializza la classe.

        Args:
            data: immagine PIL singola o lista di dizionari con chiave 'image'.
        """
        self.data = data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Usa GPU se disponibile

        # Caricamento modello DINOv2 pre-addestrato (ViT base)
        self.model = timm.create_model("vit_base_patch14_dinov2", pretrained=True)
        self.model.eval().to(self.device)  # Modalità eval per disabilitare dropout, batchnorm, ecc.

        # Trasformazioni immagini per il modello
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),  # Ridimensiona a 518x518
            transforms.ToTensor(),           # Converti in tensore
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalizzazione standard ImageNet
                                 std=[0.229, 0.224, 0.225])
        ])

    def create_embeddings(self, batch_size=8):
        """
        Crea embedding DINOv2 per batch di immagini.

        Args:
            batch_size: numero di immagini da processare insieme.

        Returns:
            Numpy array di embedding, o singolo embedding se input singolo.
        """
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
        for i in tqdm(range(0, len(images), batch_size), desc="Creating DINO embeddings"):
            batch_images = images[i:i + batch_size]
            # Applica trasformazioni e crea tensore batch
            batch_tensors = torch.stack([self.transform(img) for img in batch_images if img is not None]).to(self.device)

            with torch.no_grad():  # Disattiva gradiente
                feats = self.model.forward_features(batch_tensors)  # Estrai features

                # DINOv2 può restituire dizionario o tensore
                if isinstance(feats, dict):
                    embeddings = feats["x_norm_clstoken"]      # Estrai token CLS normalizzato (B, 768)
                else:
                    embeddings = feats[:, 0, :]                # Se tensore, prendi primo token CLS

                # Normalizzazione L2 dei vettori
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            embeddings = embeddings.cpu().numpy()  # Trasferisci su CPU

            # Salvataggio embedding nel caso di lista di dizionari
            if single_image:
                all_embeddings.extend(embeddings)
            else:
                for item, emb in zip(self.data[i:i + batch_size], embeddings):
                    item["dino_embedding"] = emb  # Salva embedding nel dizionario
                    all_embeddings.append(emb)

        return np.array(all_embeddings) if not single_image else all_embeddings[0]
