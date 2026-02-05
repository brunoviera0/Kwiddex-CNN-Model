IMPORTANT: Never commit the private key.
Add private key to .gitignore:
echo "keys/kwiddex_private.pem" >> .gitignore

ENVIRONMENT VARIABLES

KWX_BUCKET            GCS bucket name (kwiddex-datasets)
KWX_DATA_BASE         Local cache directory (data/full_dataset)
KWX_DATASET           Dataset name/folder in bucket
PREV_BACKBONE_PATH    Previous model for cnn_part1.py
DOC_BACKBONE_PATH     Document backbone for cnn_part2.py




STEPS TO RUN CODE

1. SSH into VM instance within Google Cloud Compute Engine

2. Clone repo and CD into project folder

		git clone https://github.com/brunoviera0/Kwiddex-CNN-Model.git
		cd Kwiddex-CNN-Model

4. Install dependencies

		pip install -r requirements.txt

5. Configure Google Cloud
   
		gcloud auth application-default login
		gcloud config set project sentiment-analysis-379200

6. Set environment variables

		export KWX_DATA_BASE="data/full_dataset"
		export KWX_BUCKET="kwiddex-datasets"
		export KWX_DATASET="name_of_dataset"
	
7. Register a dataset from Kaggle

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



11. Document Certification

Setup (one time only):

	python3 certification.py setup

	Creates keys/kwiddex_private.pem (Secret)
	Creates keys/kwiddex_public.pem (safe to share)

Test the system:

	python3 certification.py test

Certify a document/image:

	python3 certification.py certify input.pdf output_certified.pdf
	python3 certification.py certify input.jpg output_certified.pdf

Verify a certified document:

	python3 certification.py verify certified.pdf

How it works:

SHA-256 hash of document creates unique fingerprint

Certificate contains hash, confidence score, timestamp, human ID

RSA private key signs the certificate

Certificate embedded in PDF metadata

Can verify using the public key


12. User Creation

Create a user:

	python3 auth.py create username password

Login:

	python3 auth.py login username password

List users:

	python3 auth.py list

What is stored:

User ID

Username

Password hash (SHA-256)



13. Run the inference API

		python3 predict.py

API Endpoints:

POST /predict        Single prediction (upload image or PDF)

POST /monte_carlo    Prediction with uncertainty estimation

GET  /health         Health check

GET  /document/{id}  Get previous prediction result

POST /certify          Certify a verified document

POST /verify-certificate   Verify a certified PDF

GET  /certificate/{id}     Look up certificate using ID


Test with curl: 

	curl -X POST "http://localhost:8000/predict" 
	-F "file=@test_document.jpg"

Results are stored in google cloud datastore table



14. Demo

Run the pipeline demo:

	python3 demo_pipeline.py

Shows:

1. User account creation

2. User login

3. Document upload and model scoring

4. Document certification with digital signature

5. Certificate verification

6. Image certification

7. Certificate revocation
