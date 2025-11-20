import os
import json
import shutil
from pathlib import Path

from google.cloud import storage
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def _required_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(
            f"Environment variable '{name}' is required.\n"
            f'Example:\n'
            f'  export KWX_DATA_BASE="data/full_dataset"\n'
            f'  export KWX_BUCKET="kwiddex-datasets"\n'
        )
    return v


def _sync_from_gcs(dataset_name: str, dst_root: Path):
    """Download the dataset directory from GCS if not already cached locally."""
    bucket_name = _required_env("KWX_BUCKET")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    prefix = f"{dataset_name}/"

    print(f"Checking for dataset in GCS bucket: {bucket_name}/{prefix}")

    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise RuntimeError(
            f"Dataset '{dataset_name}' not found in bucket '{bucket_name}'."
        )

    if dst_root.exists():
        print(f"Local cache found: {dst_root}")
        return

    print(f"Downloading dataset to local directory: {dst_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    for blob in blobs:
        rel = blob.name[len(prefix):]
        if not rel:  
            continue
        local_path = dst_root / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

    print("Download complete.")

def get_dataloaders(dataset_name: str,
                    batch_size: int = 32,
                    num_workers: int = 2,
                    img_size: int = 256):

    base_dir = Path(_required_env("KWX_DATA_BASE"))
    dataset_dir = base_dir / dataset_name

    #Ensure dataset exists locally
    _sync_from_gcs(dataset_name, dataset_dir)

    #Load class mapping (fake = 0, real = 1)
    mapping_path = dataset_dir / "class_mapping.json"
    if not mapping_path.exists():
        raise RuntimeError("class_mapping.json missing — dataset is incomplete.")

    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    classes = mapping["final_order"]   # ['fake', 'real']
    print(f"Classes: {classes}")

    #augmentation and transforms

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomRotation(8),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    #building datasets
    train_path = dataset_dir / "train"
    val_path   = dataset_dir / "val"
    test_path  = dataset_dir / "test"

    if not train_path.exists():
        raise RuntimeError("Training folder missing — pipeline did not complete.")

    d_train = datasets.ImageFolder(train_path, transform=train_tf)
    d_val   = datasets.ImageFolder(val_path, transform=eval_tf)
    d_test  = datasets.ImageFolder(test_path, transform=eval_tf)

    train_loader = DataLoader(
        d_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        d_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        d_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    #Simple summary
    print(f"Train: {len(d_train)} images")
    print(f"Val:   {len(d_val)} images")
    print(f"Test:  {len(d_test)} images")

    return train_loader, val_loader, test_loader, classes

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug Kwiddex Dataloader")
    parser.add_argument("dataset", help="Dataset name (folder in bucket)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)

    args = parser.parse_args()

    get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch,
        num_workers=args.workers,
        img_size=args.size
    )

