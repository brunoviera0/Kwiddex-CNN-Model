Steps to Run Code:

1. SSH into VM instance within the Google Cloud Compute Engine

2. Clone Repo and CD into project folder "git clone https://github.com/brunoviera0/Kwiddex-CNN-Model.git", and then "cd Kwiddex-CNN-Model"

3. Install dependencies: "pip install -r requirements.txt"

4. Configue cloud storage: "gcloud auth list" & "gcloud config set project sentiment-analysis-379200"

5. Run the dataset pipeline using the following structure (once per dataset):
  5. Set enviornment variables: export KWX_DATA_BASE="data/full_dataset"
export KWX_BUCKET="kwiddex-datasets"
export KWX_DATASET="name_of_dataset"

6. python3 register_dataset.py

7. Code will ask you for "user/datasetname" end portion of kaggle link.

8. python3 dataset_pipeline.py \
  --url "URL/to/dataset/in/GCS/bucket/dataset.zip" \
  --dataset "name_of_dataset"

-Downloads ZIP/TAR.
-Extracts and finds class subfolders (real/fake)
-Splits data (70/20/10)
-Uploads train/val/test to google cloud bucket
-Creates a manifest.json summary (record of how the dataset was split)

9. Train the CNN model: "python3 cnn_model.py"

-Loads dataset using environment variables
-Trains ResNet-18
-Saves the best model (best_real_fake_resnet18.pt)
-Prints train/val/test accuracy

10. Repeat steps with new dataset

