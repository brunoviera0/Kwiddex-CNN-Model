import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#path to dataset bucket
DATA_ROOT = "data/full_dataset"

def get_dataloaders(batch_size=32, num_workers=0):
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=transform_train)
    val   = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"),   transform=transform_eval)
    test  = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),  transform=transform_eval)

    loaders = {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        "val":   DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test":  DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, train.classes
