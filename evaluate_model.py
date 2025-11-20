import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path, num_classes=2):
    model = models.resnet18(weights=None)  # No pretrained weights here
    model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_dataloader(test_dir):
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(test_dir, transform=tfms)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return loader, dataset.classes


def evaluate(model, loader, class_names):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n===== Classification Report =====")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\n===== Confusion Matrix =====")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    checkpoint = "best_real_fake_resnet18.pt"
    test_dir = "data/full_dataset/casia2_forgery/test"

    print("\nLoading model:", checkpoint)
    model = load_model(checkpoint)

    print("\nLoading test dataset…")
    loader, class_names = get_dataloader(test_dir)

    print("\nEvaluating on test set…")
    evaluate(model, loader, class_names)
