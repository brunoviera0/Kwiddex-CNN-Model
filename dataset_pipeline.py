import argparse
import os
import random
import shutil
import tempfile
import zipfile
import tarfile
import json
import requests
from tqdm import tqdm
from google.cloud import storage

PROJECT_ID  = "sentiment-analysis-379200"
BUCKET_NAME = "kwiddex-datasets"
SPLITS      = {"train": 0.7, "val": 0.2, "test": 0.1}

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

def download_file(url, dest_path):
    if "storage.googleapis.com/kwiddex-datasets" in url:
        blob_name = url.split("kwiddex-datasets/")[-1]
        blob = bucket.blob(blob_name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)
        print(f"Downloaded from GCS bucket: {blob_name}")
    else:
        #HTTPS download
        r = requests.get(url, stream=True)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
        print(f"Downloaded via HTTPS: {url}")

def extract_archive(archive_path, extract_to):
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_to)
    elif archive_path.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as t:
            t.extractall(extract_to)
    else:
        raise ValueError("Unsupported archive type.")
    print("Extracted dataset to:", extract_to)

def split_dataset(base_dir):
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    result = {s: {c: [] for c in classes} for s in SPLITS.keys()}

    for cls in classes:
        src = os.path.join(base_dir, cls)
        files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * SPLITS["train"])
        n_val   = int(n * SPLITS["val"])
        splits = {
            "train": files[:n_train],
            "val":   files[n_train:n_train+n_val],
            "test":  files[n_train+n_val:],
        }
        for split, subset in splits.items():
            result[split][cls] = [os.path.join(src, f) for f in subset]
    return result

def upload_to_gcs(split_map, dataset_name):
    for split, classes in split_map.items():
        for cls, files in classes.items():
            for f in tqdm(files, desc=f"Uploading {split}/{cls}", unit="file"):
                dest_path = f"{dataset_name}/{split}/{cls}/{os.path.basename(f)}"
                blob = bucket.blob(dest_path)
                blob.upload_from_filename(f)
    print("Uploaded all dataset files to GCS.")

def write_manifest(split_map, dataset_name):
    manifest = {}
    for split, classes in split_map.items():
        manifest[split] = {cls: len(files) for cls, files in classes.items()}
    manifest_blob = bucket.blob(f"{dataset_name}/manifest.json")
    manifest_blob.upload_from_string(json.dumps(manifest, indent=2), content_type="application/json")
    print("Uploaded manifest.json:")
    print(json.dumps(manifest, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Kwiddex Dataset Pipeline")
    parser.add_argument("--url", required=True, help="URL of dataset zip/tar file")
    parser.add_argument("--dataset", required=True, help="Dataset name (folder name in bucket)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "dataset.zip")
        download_file(args.url, archive_path)

        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        extract_archive(archive_path, extract_dir)

        #detect first subdir containing images
        root_dirs = [os.path.join(extract_dir, d) for d in os.listdir(extract_dir)]
        root = next((d for d in root_dirs if os.path.isdir(d)), extract_dir)

        split_map = split_dataset(root)
        upload_to_gcs(split_map, args.dataset)
        write_manifest(split_map, args.dataset)

    print(f"\n Completed dataset pipeline for {args.dataset}")

if __name__ == "__main__":
    main()

