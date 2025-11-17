import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os
from dataset_manager import get_dataloaders

# Environment variables define dataset
dataset_name = os.environ["KWX_DATASET"]
dtr, dval, dte, classes = get_dataloaders(dataset_name, batch_size=32, num_workers=2)

# Training parameters
PHASE1_EPOCHS = 5   # Phase 1: train only the final classifier layer
PHASE2_EPOCHS = 10  # Phase 2: fine-tune the full backbone
LR_HEAD        = 3e-4   # higher LR for classifier head
LR_FULL        = 1e-4   # smaller LR when fine-tuning the whole network
WEIGHT_DECAY   = 1e-4
MODEL_SAVE     = "best_real_fake_resnet18.pt"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for imgs, labels in tqdm(dataloader, desc="Training"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
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
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(1)
            total_loss += loss.item() * imgs.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)
    return total_loss / total_samples, total_correct / total_samples


def main():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(DEVICE)

    model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, len(classes))
)

    # Class weights to compensate for dataset imbalance
    # Real ≈ 825, Fake ≈ 163 ⇒ ~5:1 ratio
    # Class weights = [1.0, 5.0] so mistakes on fake class are penalized more
    #  class_weights = torch.tensor([1.0, 5.0], dtype=torch.float32).to(DEVICE)
>>>>>>> 9b7d0ff (remove weights from model)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = model.state_dict()

    # PHASE 1: Train classifier head only
    print("\n Phase 1")

    # Loop through every trainable parameter in the model and freeze it so gradients are not computed and not updated during backpropagation.
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the final fully-connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Optimizer only sees the fully connected parameters in Phase 1
    optimizer = optim.AdamW(model.fc.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS, eta_min=1e-6)

    for epoch in range(PHASE1_EPOCHS):
        print(f"\n[Phase 1] Epoch {epoch+1}/{PHASE1_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, dtr, optimizer, criterion)
        val_loss, val_acc = evaluate(model, dval, criterion)
        scheduler.step()
        print(f"Train acc: {train_acc*100:.2f}% | Val acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            torch.save(best_state, MODEL_SAVE)
            print(" Saved best model checkpoint of Phase 1")

    # PHASE 2: Fine-tune full backbone
    print("\n Phase 2")

    # Unfreeze entire network
    for param in model.parameters():
        param.requires_grad = True

    # New optimizer now updates all parameters
    optimizer = optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS, eta_min=1e-6)

    for epoch in range(PHASE2_EPOCHS):
        print(f"\n[Phase 2] Epoch {epoch+1}/{PHASE2_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, dtr, optimizer, criterion)
        val_loss, val_acc = evaluate(model, dval, criterion)
        scheduler.step()
        print(f"Train acc: {train_acc*100:.2f}% | Val acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            torch.save(best_state, MODEL_SAVE)
            print("Saved best model checkpoint (Phase 2).")

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, dte, criterion)
    print(f"\nFinal test accuracy (best model): {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()

