#!/bin/bash

# ======== DATASET VARIABLES (fill these in) ========
DATA_URL="https://your-dataset-download-link.zip"     #DOWNLOAD URL
BUCKET_NAME="kwiddex-datasets"                        # your GCS bucket
DATA_NAME="document_authenticity"                     #FOLDER NAME FOR DATA
SPLIT_TRAIN=0.7
SPLIT_VAL=0.2
SPLIT_TEST=0.1

set -e  #exit on error

echo "📦 Downloading dataset..."
mkdir -p ~/datasets
cd ~/datasets
wget -O "${DATA_NAME}.zip" "$DATA_URL"

echo "Extracting..."
unzip -q "${DATA_NAME}.zip" -d "${DATA_NAME}_raw"

echo "Preparing split directories..."
mkdir -p "${DATA_NAME}/train" "${DATA_NAME}/val" "${DATA_NAME}/test"

echo "Splitting dataset..."
python3 <<'PYCODE'
import os, random, shutil

raw_dir = "${DATA_NAME}_raw"
out_dir = "${DATA_NAME}"
split_train = float("${SPLIT_TRAIN}")
split_val   = float("${SPLIT_VAL}")
split_test  = float("${SPLIT_TEST}")

#detect class subfolders (e.g. real/fake)
classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

for cls in classes:
    src = os.path.join(raw_dir, cls)
    files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    random.shuffle(files)
    n = len(files)
    n_train = int(n * split_train)
    n_val   = int(n * split_val)
    splits = {
        "train": files[:n_train],
        "val":   files[n_train:n_train+n_val],
        "test":  files[n_train+n_val:],
    }
    for split, subset in splits.items():
        dst = os.path.join(out_dir, split, cls)
        os.makedirs(dst, exist_ok=True)
        for f in subset:
            shutil.move(os.path.join(src, f), os.path.join(dst, f))
PYCODE

echo "Uploading to Google Cloud bucket..."
gsutil -m cp -r "${DATA_NAME}/" "gs://${BUCKET_NAME}/${DATA_NAME}/"

echo "Dataset uploaded to gs://${BUCKET_NAME}/${DATA_NAME}/"
