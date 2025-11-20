import argparse
import json
import os
import random
import sys
import tempfile
import zipfile
import tarfile
from typing import Dict, List
import shutil
import requests
from google.cloud import storage
from tqdm import tqdm


SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}

def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is required.\n"
            f'Example:\n'
            f'  export KWX_BUCKET="kwiddex-datasets"\n'
        )
    return value


def download_archive(url: str, dest_path: str) -> None:
    # 1) Local file path
    if os.path.isfile(url):
        print(f"Copying local file: {url} → {dest_path}")
        shutil.copyfile(url, dest_path)
        return

    # 2) file:// URI
    if url.lower().startswith("file://"):
        local_path = url[7:]  # strip file://
        print(f"Copying file URI: {local_path} → {dest_path}")
        shutil.copyfile(local_path, dest_path)
        return

    # 3) gs:// URI
    if url.lower().startswith("gs://"):
        print(f"Downloading from GCS: {url} → {dest_path}")
        parts = url[5:].split("/", 1)
        bucket_name, blob_name = parts[0], parts[1]
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(dest_path)
        return

    # 4) HTTP/HTTPS URL
    print(f"Downloading archive from HTTP: {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Downloading"
    ) as pbar:
        for chunk in r.iter_content(1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            pbar.update(len(chunk))
    print(f"Saved archive to: {dest_path}")


def extract_archive(archive_path: str, extract_to: str) -> None:
    print(f"Extracting archive to: {extract_to}")
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_to)
    elif archive_path.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as t:
            t.extractall(extract_to)
    else:
        raise ValueError("Unsupported archive type (expected .zip or .tar[.gz]).")


def find_root_dir(extract_dir: str) -> str:
    entries = [
        os.path.join(extract_dir, d)
        for d in os.listdir(extract_dir)
        if os.path.isdir(os.path.join(extract_dir, d))
    ]

    if len(entries) == 1:
        root = entries[0]
    else:
        root = extract_dir

    print(f"Detected dataset root: {root}")
    return root


def list_candidate_class_dirs(root: str) -> List[str]:
    ignore_keywords = [
        "groundtruth", "ground truth", "mask", "masks",
        "annotation", "annotations", "label", "labels"
    ]

    dirs = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        lower = name.lower()
        if any(k in lower for k in ignore_keywords):
            print(f"Skipping non-class dir: {name}")
            continue
        dirs.append(name)

    return dirs


def suggest_mapping(class_names: List[str]) -> Dict[str, str]:
    fake_keywords = ["fake", "forg", "tp", "tamper", "counterfeit", "spoof"]
    real_keywords = ["real", "auth", "genuine", "au", "original"]

    mapping: Dict[str, str] = {}
    used_targets = set()

    for name in class_names:
        lower = name.lower()
        target = None

        if any(k in lower for k in fake_keywords):
            target = "fake"
        elif any(k in lower for k in real_keywords):
            target = "real"

        if target and target not in used_targets:
            mapping[name] = target
            used_targets.add(target)
        else:
            mapping[name] = None

    return mapping


def resolve_mapping_interactively(class_names: List[str]) -> Dict[str, str]:
    #Step 1: suggestions
    suggestions = suggest_mapping(class_names)
    suggested_targets = {v for v in suggestions.values() if v is not None}

    use_suggestions = False
    if suggested_targets == {"fake", "real"} and all(
        suggestions[n] is not None for n in class_names
    ):
        print("\nSuggested class mapping based on folder names:")
        for n in class_names:
            print(f"  {n} -> {suggestions[n]}")
        while True:
            ans = input("Accept this mapping? [y/n]: ").strip().lower()
            if ans in ("y", "yes"):
                use_suggestions = True
                break
            if ans in ("n", "no"):
                break

    if use_suggestions:
        return suggestions

    #Step 2: manual mapping
    print("\nManual class mapping:")
    while True:
        mapping: Dict[str, str] = {}
        for name in class_names:
            while True:
                ans = input(f'Map "{name}" to (r)eal or (f)ake? [r/f]: ').strip().lower()
                if ans == "r":
                    mapping[name] = "real"
                    break
                if ans == "f":
                    mapping[name] = "fake"
                    break
                print("Please enter 'r' for real or 'f' for fake.")

        targets = set(mapping.values())
        if targets == {"fake", "real"}:
            return mapping

        print("You must map exactly one class to real and one to fake. Let's try again.\n")


def build_split_map(root: str, class_mapping: Dict[str, str]):
    random.seed(42)  # reproducible splits

    split_map = {
        "train": {"fake": [], "real": []},
        "val": {"fake": [], "real": []},
        "test": {"fake": [], "real": []},
    }

    for orig_name, target in class_mapping.items():
        src_dir = os.path.join(root, orig_name)
        files = [
            f for f in os.listdir(src_dir)
            if os.path.isfile(os.path.join(src_dir, f))
        ]
        if not files:
            print(f"WARNING: no files found in class folder: {src_dir}")
            continue

        random.shuffle(files)
        n = len(files)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        #whatever remains goes to test
        n_test = n - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        print(
            f'Class "{orig_name}" (-> {target}): total={n}, '
            f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )

        for fname in train_files:
            split_map["train"][target].append(os.path.join(src_dir, fname))
        for fname in val_files:
            split_map["val"][target].append(os.path.join(src_dir, fname))
        for fname in test_files:
            split_map["test"][target].append(os.path.join(src_dir, fname))

    return split_map


def upload_split_to_gcs(bucket, dataset_name: str, split_map) -> None:
    for split, classes in split_map.items():
        for cls, files in classes.items():
            for path in tqdm(files, desc=f"Uploading {split}/{cls}", unit="file"):
                dest = f"{dataset_name}/{split}/{cls}/{os.path.basename(path)}"
                blob = bucket.blob(dest)
                blob.upload_from_filename(path)


def write_manifest_and_mapping(bucket, dataset_name: str,
                               split_map,
                               class_mapping: Dict[str, str]) -> None:
    manifest = {}
    total_counts = {"fake": 0, "real": 0}

    for split, classes in split_map.items():
        manifest[split] = {}
        for cls, files in classes.items():
            count = len(files)
            manifest[split][cls] = count
            total_counts[cls] += count

    manifest["total_counts"] = total_counts

    # class_mapping.json
    mapping_payload = {
        "original_classes": list(class_mapping.keys()),
        "mapped_classes": class_mapping,          # e.g. {"Au": "real", "Tp": "fake"}
        "final_order": ["fake", "real"]           # label 0, label 1
    }

    m_blob = bucket.blob(f"{dataset_name}/manifest.json")
    m_blob.upload_from_string(json.dumps(manifest, indent=2),
                              content_type="application/json")

    c_blob = bucket.blob(f"{dataset_name}/class_mapping.json")
    c_blob.upload_from_string(json.dumps(mapping_payload, indent=2),
                              content_type="application/json")

    print("\nUploaded manifest.json and class_mapping.json")
    print(json.dumps(manifest, indent=2))
    print(json.dumps(mapping_payload, indent=2))



def main():
    parser = argparse.ArgumentParser(
        description="Kwiddex real/fake dataset pipeline"
    )
    parser.add_argument(
        "--url", required=True,
        help="HTTPS URL to dataset archive (e.g. GCS public URL)"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Logical dataset name (e.g. casia2_forgery)"
    )
    args = parser.parse_args()

    bucket_name = _required_env("KWX_BUCKET")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "dataset.zip")
        extract_dir = os.path.join(tmpdir, "extracted")

        os.makedirs(extract_dir, exist_ok=True)

        # 1. Download + extract
        download_archive(args.url, archive_path)
        extract_archive(archive_path, extract_dir)

        # 2. Find dataset root
        root = find_root_dir(extract_dir)

        # 3. Find candidate class dirs
        class_names = list_candidate_class_dirs(root)
        print(f"\nCandidate class folders under root: {class_names}")

        if len(class_names) != 2:
            print(
                "ERROR: Dataset is incompatible with Kwiddex real/fake pipeline.\n"
                f"Reason: Expected exactly 2 class folders, found {len(class_names)}.\n"
                "Please check the archive structure and choose a different dataset\n"
                "or adjust how it is packaged."
            )
            sys.exit(1)

        # 4. Resolve mapping (semi-interactive)
        class_names_sorted = sorted(class_names)
        class_mapping = resolve_mapping_interactively(class_names_sorted)
        print("\nFinal class mapping:")
        for orig, target in class_mapping.items():
            print(f"  {orig} -> {target}")

        # 5. Build split map (no physical renaming needed; we standardize in GCS paths)
        split_map = build_split_map(root, class_mapping)

        # 6. Upload images + metadata to GCS
        upload_split_to_gcs(bucket, args.dataset, split_map)
        write_manifest_and_mapping(bucket, args.dataset, split_map, class_mapping)

    print(f"\nCompleted real/fake dataset pipeline for '{args.dataset}'")


if __name__ == "__main__":
    main()
