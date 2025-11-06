Steps to Run Code:

1. SSH into VM instance within the Google Cloud Compute Engine
2. Clone Repo and CD into project folder "git clone https://github.com/brunoviera0/Kwiddex-CNN-Model.git", and then "cd Kwiddex-CNN-Model"
3. Install dependencies: "pip install -r requirements.txt"
4. Configue cloud storage: "gcloud auth list" & "gcloud config set project sentiment-analysis-379200"
5. Run the dataset pipeline using the following structure (once per dataset):

"python3 dataset_pipeline.py \
  --url "URL/to/dataset.zip" \
  --dataset "dataset_name"

What happens:
-Downloads ZIP/TAR.
-Extracts and finds class subfolders (real/fake)
-Splits data (70/20/10)
-Uploads train/val/test to google cloud bucket
-Creates a manifest.json summary (record of how the dataset was split)

6. Sync dataset back to VM:

"mkdir -p data/full_dataset"
"gsutil -m rsync -r gs://kwiddex-datasets/"dataset_name" data/full_dataset"

7. Train the CNN model: "python3 cnn_model.py"

8. Repeat steps with new URL and new dataset name

