import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from dataset_manager import get_dataloaders

dataset_name = os.environ["KWX_DATASET"]
font_pretrained_path = os.environ.get("FONT_BACKBONE", "")

if not font_pretrained_path:
    raise ValueError("FONT_BACKBONE environment variable not set.")

dtr, dval, dte, classes = get_dataloaders(
    dataset_name,
    batch_size=32,
    num_workers=2
)

if len(classes) != 2:
    raise ValueError(
        f"Expected 2 classes (real/fake), but got {len(classes)}: {classes}"
    )


# Training parameters

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 25

LR_HEAD      = 1e-3
LR_FULL      = 1e-4
WEIGHT_DECAY = 1e-4

MODEL_SAVE = f"best_{dataset_name}.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def get_class_weights_from_loader(dataloader, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.long)

    for _, labels in dataloader:
        counts += torch.bincount(labels, minlength=num_classes)

    counts = counts.float()

    # Inverse frequency weighting
    weights = counts.sum() / (num_classes * counts)

    return weights


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for imgs, labels in tqdm(dataloader, desc="Training"):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)

        total_loss += loss.item() * imgs.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples


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

            preds = outputs.argmax(1)

            total_loss += loss.item() * imgs.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples


def build_model(num_classes):
    # Start from ImageNet pretrained ResNet-18
    try:
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
    except AttributeError:
        model = models.resnet18(pretrained=True)

    # Replace final layer for real/fake prediction
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def load_font_weights_into_model(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")

    # Remove font classifier head weights
    state.pop("fc.weight", None)
    state.pop("fc.bias", None)

    model.load_state_dict(state, strict=False)


def main():
    model = build_model(len(classes)).to(DEVICE)

    # Load font-pretrained backbone weights
    load_font_weights_into_model(model, font_pretrained_path)

    # Automatically compute class weights from the train loader
    class_weights = get_class_weights_from_loader(dtr, len(classes)).to(DEVICE)

    print("Class order:", classes)
    print("Auto class weights:", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0.0
    best_state = model.state_dict()


    # Phase 1: train only the classifier head

    print("\nPhase 1: Training classifier head")

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the final layer
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(
        model.fc.parameters(),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=PHASE1_EPOCHS,
        eta_min=1e-6
    )

    for epoch in range(PHASE1_EPOCHS):
        print(f"\n[Phase 1] Epoch {epoch + 1}/{PHASE1_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, dtr, optimizer, criterion
        )
        val_loss, val_acc = evaluate(
            model, dval, criterion
        )

        scheduler.step()

        print(f"Train acc: {train_acc * 100:.2f}%")
        print(f"Val acc:   {val_acc * 100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            torch.save(best_state, MODEL_SAVE)
            print("Saved best model so far")


    # Phase 2: fine-tune entire model

    print("\nPhase 2: Fine-tuning full model")

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR_FULL,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=PHASE2_EPOCHS,
        eta_min=1e-6
    )

    for epoch in range(PHASE2_EPOCHS):
        print(f"\n[Phase 2] Epoch {epoch + 1}/{PHASE2_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, dtr, optimizer, criterion
        )
        val_loss, val_acc = evaluate(
            model, dval, criterion
        )

        scheduler.step()

        print(f"Train acc: {train_acc * 100:.2f}%")
        print(f"Val acc:   {val_acc * 100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            torch.save(best_state, MODEL_SAVE)
            print("Saved best model so far")


    # Final evaluation

    model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, dte, criterion)
    print(f"\nFinal test accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()

# Example run:
# export KWX_DATASET="find_it_again_receipts"
# export FONT_BACKBONE="best_pretrain_{dataset_name}.pt"
# python3 fine_tune.py
