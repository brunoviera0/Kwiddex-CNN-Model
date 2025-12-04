from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import datastore, storage
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import io
import pdf2image
from datetime import datetime
import numpy as np
import uuid
import imgaug.augmenters as iaa

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#gcp
GCP_PROJECT = "sentiment-analysis-379200"
BUCKET_NAME = "kwiddex-datasets"
DOCUMENTS_FOLDER = "documents"
MODEL_LOCAL_PATH = "best_real_fake_resnet18.pt"

#clients
datastore_client = datastore.Client(project=GCP_PROJECT)
storage_client = storage.Client(project=GCP_PROJECT)
bucket = storage_client.bucket(BUCKET_NAME)



#load model
def load_model():
    model = resnet18(weights=None)
    
    #match saved model (sequential with dropout)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, 2)
    )
    
    #saved weights
    state_dict = torch.load(MODEL_LOCAL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

model = load_model()


#image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    confidence_interval: dict
    timestamp: str
    result_id: str
    document_id: str
    gcs_path: str

class MonteCarloResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    confidence_interval: dict
    monte_carlo_stats: dict
    timestamp: str
    result_id: str
    document_id: str
    gcs_path: str



def upload_to_gcs(file_content: bytes, original_filename: str, content_type: str) -> tuple:
    #unique document ID
    document_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    #GCS path with timestamp and UUID
    file_extension = original_filename.split('.')[-1] if '.' in original_filename else 'unknown'
    blob_name = f"{DOCUMENTS_FOLDER}/{timestamp}_{document_id}.{file_extension}"
    
    #Upload to GCS
    blob = bucket.blob(blob_name)
    blob.upload_from_string(file_content, content_type=content_type)
    
    gcs_path = f"gs://{BUCKET_NAME}/{blob_name}"
    
    return document_id, gcs_path



def preprocess_image(image: Image.Image):
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)





def compute_confidence_interval(probabilities: np.ndarray, confidence_level=0.95) -> dict:
    confidence = float(np.max(probabilities))
    predicted_class = int(np.argmax(probabilities))
    
    #confidence interval based on softmax probabilities
    z_score = 1.96 if confidence_level == 0.95 else 2.576
    std_estimate = np.sqrt(confidence * (1 - confidence))
    margin = z_score * std_estimate
    
    return {
        "mean": confidence,
        "lower_bound": max(0.0, confidence - margin),
        "upper_bound": min(1.0, confidence + margin),
        "confidence_level": confidence_level
    }


def save_to_datastore(prediction: int, label: str, confidence: float, ci: dict, 
                      filename: str, document_id: str, gcs_path: str, monte_carlo_stats: dict = None) -> str:
    key = datastore_client.key("PredictionResult")
    entity = datastore.Entity(key)
    entity_data = {
        "document_id": document_id,
        "gcs_path": gcs_path,
        "original_filename": filename,
        "prediction": prediction,
        "prediction_label": label,
        "confidence": confidence,
        "confidence_interval": ci,
        "timestamp": datetime.utcnow(),
        "processed": True,
        "method": "monte_carlo" if monte_carlo_stats else "standard"
    }
    if monte_carlo_stats:
        entity_data["monte_carlo_stats"] = monte_carlo_stats
    entity.update(entity_data)
    datastore_client.put(entity)
    return str(entity.key.id)




def apply_augmentations(image: Image.Image, num_augmentations: int = 30) -> list:
    img_np = np.array(image)
    augmenter = iaa.SomeOf((1, 3), [
        iaa.Rotate((-10, 10)),
        iaa.Multiply((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        iaa.Affine(scale=(0.95, 1.05)),
        iaa.JpegCompression(compression=(70, 99))
    ], random_order=True)
    augmented_images = []
    for _ in range(num_augmentations):
        aug_img = augmenter(image=img_np)
        augmented_images.append(Image.fromarray(aug_img))
    return augmented_images




def monte_carlo_inference(image: Image.Image, num_samples: int = 30) -> dict:
    augmented_images = apply_augmentations(image, num_samples)
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for aug_img in augmented_images:
            processed = preprocess_image(aug_img)
            outputs = model(processed)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()[0]
            all_predictions.append(int(np.argmax(probs_np)))
            all_probabilities.append(probs_np)
    
    all_probabilities = np.array(all_probabilities)
    mean_probs = np.mean(all_probabilities, axis=0)
    std_probs = np.std(all_probabilities, axis=0)
    final_prediction = int(np.argmax(mean_probs))
    final_confidence = float(mean_probs[final_prediction])
    agreement_rate = float(np.sum(np.array(all_predictions) == final_prediction) / len(all_predictions))
    percentile_lower = np.percentile(all_probabilities[:, final_prediction], 2.5)
    percentile_upper = np.percentile(all_probabilities[:, final_prediction], 97.5)
    
    

    return {
        "prediction": final_prediction,
        "confidence": final_confidence,
        "confidence_interval": {
            "mean": final_confidence,
            "lower_bound": float(percentile_lower),
            "upper_bound": float(percentile_upper),
            "confidence_level": 0.95
        },
        "monte_carlo_stats": {
            "num_samples": num_samples,
            "agreement_rate": agreement_rate,
            "std_dev": float(std_probs[final_prediction]),
            "class_probabilities": {
                "fake": float(mean_probs[0]),
                "real": float(mean_probs[1])
            }
        }
    }




@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        #upload to GCS
        document_id, gcs_path = upload_to_gcs(content, file.filename, file.content_type)
        
        #process the document (pdf or image)
        if file.content_type == "application/pdf":
            #convert first page of PDF to image
            images = pdf2image.convert_from_bytes(content)
            image = images[0]
        else:
            #open as image
            image = Image.open(io.BytesIO(content))
        
        processed_image = preprocess_image(image)
        
        #run inference
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probabilities_np = probabilities.cpu().numpy()[0]
        
        #prediction and confidence
        predicted_class = int(np.argmax(probabilities_np))
        confidence = float(np.max(probabilities_np))
        
        #map to label (0=fake, 1=real)
        label = "real" if predicted_class == 1 else "fake"
        
        #confidence interval
        ci = compute_confidence_interval(probabilities_np)
        
        #save to Datastore with GCS reference
        result_id = save_to_datastore(
            prediction=predicted_class,
            label=label,
            confidence=confidence,
            ci=ci,
            filename=file.filename,
            document_id=document_id,
            gcs_path=gcs_path
        )
        
        return PredictionResponse(
            prediction=predicted_class,
            prediction_label=label,
            confidence=confidence,
            confidence_interval=ci,
            timestamp=datetime.utcnow().isoformat(),
            result_id=result_id,
            document_id=document_id,
            gcs_path=gcs_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")



@app.post("/monte_carlo", response_model=MonteCarloResponse)
async def predict_monte_carlo(file: UploadFile = File(...), num_samples: int = 30):
    try:
        content = await file.read()
        document_id, gcs_path = upload_to_gcs(content, file.filename, file.content_type)
        
        if file.content_type == "application/pdf":
            images = pdf2image.convert_from_bytes(content)
            image = images[0]
        else:
            image = Image.open(io.BytesIO(content))
        
        mc_result = monte_carlo_inference(image, num_samples)
        predicted_class = mc_result["prediction"]
        confidence = mc_result["confidence"]
        ci = mc_result["confidence_interval"]
        mc_stats = mc_result["monte_carlo_stats"]
        label = "real" if predicted_class == 1 else "fake"
        
        result_id = save_to_datastore(
            prediction=predicted_class,
            label=label,
            confidence=confidence,
            ci=ci,
            filename=file.filename,
            document_id=document_id,
            gcs_path=gcs_path,
            monte_carlo_stats=mc_stats
        )
        
        return MonteCarloResponse(
            prediction=predicted_class,
            prediction_label=label,
            confidence=confidence,
            confidence_interval=ci,
            monte_carlo_stats=mc_stats,
            timestamp=datetime.utcnow().isoformat(),
            result_id=result_id,
            document_id=document_id,
            gcs_path=gcs_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")




@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}



@app.get("/document/{document_id}")
async def get_document_result(document_id: str):
    try:
        query = datastore_client.query(kind="PredictionResult")
        query.add_filter("document_id", "=", document_id)
        results = list(query.fetch(limit=1))
        
        if not results:
            raise HTTPException(status_code=404, detail="Document not found")
        
        result = results[0]
        return {
            "document_id": result.get("document_id"),
            "gcs_path": result.get("gcs_path"),
            "original_filename": result.get("original_filename"),
            "prediction": result.get("prediction"),
            "prediction_label": result.get("prediction_label"),
            "confidence": result.get("confidence"),
            "confidence_interval": result.get("confidence_interval"),
            "timestamp": result.get("timestamp").isoformat() if result.get("timestamp") else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
