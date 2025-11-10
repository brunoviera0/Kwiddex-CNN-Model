import os
import argparse
from pathlib import Path
from typing import Tuple, List

from google.cloud import storage
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _required_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(
            f"Environment variable {name} is required.\n"
            f'  Example:\n'
            f'    export KWX_DATA_BASE="data/full_dataset"\n'
            f'    export KWX_BUCKET="kwiddex-datasets"\n'
        )
    return v


def _has_local_structure(root: Path) -> bool:
    return (root / "train").is_dir() and (root / "val").is_dir() and (root / "test").is_dir()


def _download_dataset_from_gcs(bucket_name: str, dataset_name: str, local_base: Path) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    prefix = f"{dataset_name}/"
    dest_root = local_base / dataset_name
    dest_root.mkdir(parents=True, exist_ok=True)

    # List and download all blobs under the dataset prefix
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise FileNotFoundError(
            f"No objects found in gs://{bucket_name}/{prefix}\n"
            f"Did you run the dataset pipeline for '{dataset_name}'?"
        )

    print(f"Syncing from gs://{bucket_name}/{prefix} to {dest_root}")

    for b in blobs:
        if b.name.endswith("/"):
            continue
        rel = b.name[len(prefix):]
        dest_path = dest_root / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        b.download_to_filename(str(dest_path))

    if not _has_local_structure(dest_root):
        raise RuntimeError(
            f"Downloaded dataset does not contain expected train/ val/ test/ under {dest_root}."
        )


def get_dataloaders(
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = 256,
):
    local_base = Path(_required_env("KWX_DATA_BASE"))
    bucket_name = _required_env("KWX_BUCKET")

    root = local_base / dataset_name

    if not _has_local_structure(root):
        _download_dataset_from_gcs(bucket_name=bucket_name,
                                   dataset_name=dataset_name,
                                   local_base=local_base)

    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_dir = root / "train"
    val_dir   = root / "val"
    test_dir  = root / "test"

    ds_train = datasets.ImageFolder(str(train_dir), transform=tf_train)
    ds_val   = datasets.ImageFolder(str(val_dir),   transform=tf_eval)
    ds_test  = datasets.ImageFolder(str(test_dir),  transform=tf_eval)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dl_train, dl_val, dl_test, ds_train.classes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Kwiddex dataloader")
    ap.add_argument("dataset_name", help="Name under the bucket and local base (e.g., receipt_forgery)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    dtr, dval, dte, classes = get_dataloaders(
        dataset_name=args.dataset_name,
        batch_size=args.batch,
        num_workers=args.workers,
        img_size=args.size,
    )

    print(f"Classes: {classes}")
    print(f"Train: {len(dtr.dataset)} | Val: {len(dval.dataset)} | Test: {len(dte.dataset)}")

