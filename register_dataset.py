import os, sys
from pathlib import Path
from google.cloud import storage

PROJECT = "sentiment-analysis-379200"
BUCKET  = "kwiddex-datasets"
DEST_PREFIX = "raw"  # uploaded to gs://BUCKET/raw/<file>.zip

try:
    import kaggle
except Exception:
    print("Install Kaggle first: pip3 install --user kaggle")
    sys.exit(1)

def find_archive(tmp: Path) -> Path:
    for p in tmp.iterdir():
        if p.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"} or p.name.endswith(".tar.gz"):
            return p
    raise FileNotFoundError("No archive (.zip/.tar/.gz) found in /tmp after download.")

def main():
    ds = input("Kaggle dataset (e.g. sunilthite/text-document-classification-dataset): ").strip()
    if not ds:
        print("Dataset ref required.")
        return

    #ensure Kaggle creds exist
    kg = Path("~/.kaggle/kaggle.json").expanduser()
    if not kg.exists():
        print("Missing ~/.kaggle/kaggle.json (Kaggle API key).")
        return

    tmp = Path("/tmp")
    print(f"Downloading {ds} from Kaggle to /tmp (no unzip)…")
    kaggle.api.dataset_download_files(ds, path=str(tmp), unzip=False, quiet=False)

    arc = find_archive(tmp)
    print(f"Found archive: {arc.name}")

    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    dest = f"{DEST_PREFIX}/{arc.name}".replace("//", "/")
    print(f"↑ Uploading to gs://{BUCKET}/{dest} …")
    blob = bucket.blob(dest)
    blob.upload_from_filename(str(arc))

    public_url = f"https://storage.googleapis.com/{BUCKET}/{dest}"
    print("\nPublic URL:")
    print(public_url)
    print("\nUse with your existing pipeline:")
    print(f'python3 dataset_pipeline.py --url "{public_url}" --dataset <your_dataset_name>')

if __name__ == "__main__":
    main()
