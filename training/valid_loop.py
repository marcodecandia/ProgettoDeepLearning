import os
import cv2
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import torch.nn.functional as F


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


val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

val_dataset = NarutoDataset("../data/valid", transform=val_transforms)

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def valid_epoch(model, loader, processor, device):
    model.eval()
    total_loss = 0.0

    for images, texts in tqdm(loader, desc="Training"):
        images_denorm = ((images + 1) * 127.5).clamp(0, 255).byte()
        inputs = processor(
            text=texts,
            images=images_denorm,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, return_loss=True)

            loss = outputs.loss
            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        return average_loss


num_epochs = 5

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    val_loss = valid_epoch(
        model=clip_model,
        loader=val_loader,
        processor=processor,
        device=device
    )

    print(f"Validation loss: {val_loss:.4f}")


