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

# Parameters
NUM_EPOCHS   = 15
LR           = 1e-4
WEIGHT_DECAY = 1e-4
MODEL_SAVE   = "best_real_fake_resnet18.pt"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_acc = 0.0
    best_state = model.state_dict()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, dtr, optimizer, criterion)
        val_loss, val_acc = evaluate(model, dval, criterion)
        scheduler.step()
        print(f"Train acc: {train_acc*100:.2f}% | Val acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            torch.save(best_state, MODEL_SAVE)
            print("Saved best model checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, dte, criterion)
    print(f"\nFinal test accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()

