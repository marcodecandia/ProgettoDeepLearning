import os
from PIL import Image
from tqdm import tqdm


class ImageLoader:
    def __init__(self, root):
        self.root = root

    def loader(self):
        data = []
        dataset_dir = self.root

        for label in tqdm(os.listdir(dataset_dir), desc="Caricamento classi"):
            label_dir = os.path.join(dataset_dir, label)

            if os.path.isdir(label_dir):
                for filename in os.listdir(label_dir):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        filepath = os.path.join(label_dir, filename)

                        img = Image.open(filepath).convert("RGB")

                        data.append({
                            "path": filepath,
                            "label": label,
                            "image": img
                        })
        return data

