import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from dataset_manager import get_dataloaders

# Parameters
DATASET_NAME = os.environ.get("KWX_DATASET")
PREV_BACKBONE_PATH = os.environ.get("PREV_BACKBONE_PATH")

NUM_EPOCHS   = 15
LR           = 1e-4 # start with lower training rate for now? not sure what best practice is   
WEIGHT_DECAY = 1e-4
MODEL_SAVE   = f"doc_backbone_{DATASET_NAME}.pt"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * imgs.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * imgs.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def main():
    # Load document dataset dataloaders
    dtr, dval, dte, classes = get_dataloaders(DATASET_NAME, batch_size=32, num_workers=2)
    num_classes = len(classes)

    print(f"Using dataset: {DATASET_NAME}")
    print(f"Number of document classes: {num_classes}")
    print(f"Classes: {classes}")

    model = models.resnet18(weights=None)

    if PREV_BACKBONE_PATH is not None and os.path.isfile(PREV_BACKBONE_PATH):
        print(f"Loading previous backbone from: {PREV_BACKBONE_PATH}")
        state = torch.load(PREV_BACKBONE_PATH, map_location=DEVICE)
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded backbone. Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    else:
        print("No previous backbone provided or file not found, starting from ImageNet.")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    best_val_acc = 0.0
    best_state = model.state_dict()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, dtr, optimizer, criterion)
        val_loss, val_acc = evaluate(model, dval, criterion)
        scheduler.step()

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc * 100:.2f}%")
        print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            torch.save(best_state, MODEL_SAVE)
            print(f"Saved best document backbone to {MODEL_SAVE}")

    # Load best backbone and evaluate once on test set
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, dte, criterion)
    print(f"\nFinal test loss: {test_loss:.4f}")
    print(f"Final test acc:  {test_acc * 100:.2f}%")
    print(f"Best validation acc during training: {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()


# FIRST RUN THROUGH
    # export KWX_DATASET="name_of_dataset"
    # unset PREV_BACKBONE_PATH
    # python3 cnn_part1.py

# GOING FORWARD
    # export KWX_DATASET="name_of_dataset"
    # export PREV_BACKBONE_PATH="prev_dataset.pt"
    # python3 cnn_part1.py

