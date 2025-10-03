import os
from PIL import Image
from tqdm import tqdm


class ImageLoader:
    """
    Classe per caricare immagini da una directory organizzata per classi.
    Restituisce una lista di dizionari con percorso, label e oggetto PIL.Image.
    """
    def __init__(self, root):
        """
        Inizializza il loader.

        Args:
            root (str): percorso alla cartella principale del dataset,
                        che contiene sottocartelle per ciascuna classe.
        """
        self.root = root

    def loader(self):
        """
        Scansiona la directory root e carica tutte le immagini valide.

        Restituisce:
            data (list): lista di dizionari con chiavi:
                - 'path': percorso completo dell'immagine
                - 'label': nome della cartella (classe)
                - 'image': oggetto PIL.Image
        """
        data = []  # Lista finale di immagini
        dataset_dir = self.root

        # Itera su ogni sottocartella (classe)
        for label in tqdm(os.listdir(dataset_dir), desc="Caricamento classi"):
            label_dir = os.path.join(dataset_dir, label)

            # Verifica che sia una cartella
            if os.path.isdir(label_dir):
                # Itera su ogni file della cartella
                for filename in os.listdir(label_dir):
                    # Controlla estensioni immagine
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        filepath = os.path.join(label_dir, filename)

                        # Carica immagine e converte in RGB
                        img = Image.open(filepath).convert("RGB")

                        # Aggiunge il dizionario alla lista
                        data.append({
                            "path": filepath,
                            "label": label,
                            "image": img
                        })
        return data
