import os
import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


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


train_transforms = A.Compose([
    A.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ColorJitter(p=0.2),
    A.GaussianBlur(p=0.1),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

train_dataset = NarutoDataset("../data/train", transform=train_transforms)
class_counts = torch.bincount(torch.tensor(train_dataset.labels))
class_weights = 1.0 / class_counts.float()
sample_weights = [class_weights[label] for label in train_dataset.labels]
sample_weights = torch.tensor(sample_weights)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
val_dataset = NarutoDataset("../data/valid", transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 🔒 Freezing strategy: sblocca solo le projection heads
for name, param in clip_model.named_parameters():
    if "visual_projection" in name or "text_projection" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


def train_epoch(model, loader, optimizer, processor, device):
    model.train()
    total_loss = 0.0

    for images, texts in tqdm(loader, desc="Training"):
        # Converti batch di immagini tensor → PIL per CLIPProcessor
        pil_images = [transforms.ToPILImage()(img) for img in images]

        inputs = processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, return_loss=True)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def valid_epoch(model, loader, processor, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, texts in tqdm(loader, desc="Validation"):
            pil_images = [transforms.ToPILImage()(img) for img in images]

            inputs = processor(
                text=texts,
                images=pil_images,
                return_tensors="pt",
                padding=True
            ).to(device)

            outputs = model(**inputs, return_loss=True)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(loader)


def training_loop(model, train_loader, val_loader, processor, device, num_epochs, patience=3):
    optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = np.inf
    patience_counter = 0

    # Liste per salvare l'andamento delle loss
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1} / {num_epochs}")

        train_loss = train_epoch(
            model=clip_model,
            loader=train_loader,
            optimizer=optimizer,
            processor=processor,
            device=device
        )

        val_loss = valid_epoch(
            model=clip_model,
            loader=val_loader,
            processor=processor,
            device=device
        )

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        # Salva i valori per il grafico
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "../models/trained_clip_model.pth")
            print("Model saved")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter} / {patience}")
            if patience_counter > patience:
                print("Early stopping")
                break

    # --- Plot finale ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Andamento Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


training_loop(
    model=clip_model,
    train_loader=train_loader,
    val_loader=val_loader,
    processor=processor,
    device=device,
    num_epochs=20,
    patience=3
)
