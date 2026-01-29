This project classifies documents as "real" or "fake" using a fine-tuned CNN.
It includes pipelines for dataset processing, model training, evaluation, and
a REST API for inference.

========================================
STEPS TO RUN CODE

1. SSH into VM instance within Google Cloud Compute Engine
2. Clone repo and CD into project folder
  git clone https://github.com/brunoviera0/Kwiddex-CNN-Model.git
  cd Kwiddex-CNN-Model
3. Install dependencies
  pip install -r requirements.txt
4. Configure Google Cloud
  gcloud auth application-default login
  gcloud config set project sentiment-analysis-379200
5. Set environment variables
  export KWX_DATA_BASE="data/full_dataset"
  export KWX_BUCKET="kwiddex-datasets"
  export KWX_DATASET="name_of_dataset"

6. Register a dataset from Kaggle
  python3 register_dataset.py

  Code will ask for "user/datasetname" from the Kaggle URL
  Example: if Kaggle URL is kaggle.com/datasets/uconn/fake-documents
  then enter: uconn/fake-documents
  Uploads the dataset ZIP to GCS and prints the URL

7. Run the dataset pipeline
  For real/fake binary datasets:
    python3 real_fake_dataset_pipeline.py 
    --url "https://storage.googleapis.com/kwiddex-datasets/raw/dataset.zip" 
    --dataset "name_of_dataset"

  For generic multi-class datasets:
    python3 dataset_pipeline.py 
    --url "https://storage.googleapis.com/kwiddex-datasets/raw/dataset.zip" 
    --dataset "name_of_dataset"

What this does:

Downloads ZIP/TAR from the URL
Extracts and finds class subfolders
Prompts you to map folders to "real" or "fake" (for real_fake pipeline)
Splits data 70/20/10 (train/val/test)
Uploads train/val/test to Google Cloud bucket
Creates manifest.json (record of how dataset was split)


8. Train the CNN model
  python3 cnn_model.py

What this does:

Loads dataset using environment variables
Phase 1: Trains classifier head only (5 epochs)
Phase 2: Fine-tunes full network (10 epochs)
Saves the best model as best_real_fake_resnet18.pt
Prints train/val/test accuracy


9. Two-stage training

Stage 1: Train on document classification dataset:

export KWX_DATASET="document_types_dataset"
unset PREV_BACKBONE_PATH

python3 cnn_part1.py


Stage 2: Fine-tune on real/fake dataset:
export KWX_DATASET="forgery_dataset"
export DOC_BACKBONE_PATH="doc_backbone_document_types_dataset.pt"

python3 cnn_part2.py


10. Evaluate trained model

Edit evaluate_model.py to set:

checkpoint = "your_model.pt"
test_dir = "data/full_dataset/your_dataset/test"

Then run:
python3 evaluate_model.py

Prints classification report (precision, recall, F1)
Prints confusion matrix


11. Run the inference API
python3 predict.py

API Endpoints:

POST /predict        Single prediction (upload image or PDF)
POST /monte_carlo    Prediction with uncertainty estimation
GET  /health         Health check
GET  /document/{id}  Get previous prediction result

Test with curl: curl -X POST "http://localhost:8000/predict" 
-F "file=@test_document.jpg"

Results are stored in google cloud datastore table
