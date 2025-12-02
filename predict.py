from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import datastore, storage
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import pdf2image
from datetime import datetime
import numpy as np
from torchvision.models import resnet18

app = FastAPI()

#CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#GCP Configuration
GCP_PROJECT = "sentiment-analysis-379200"
BUCKET_NAME = "kwiddex-datasets"
MODEL_LOCAL_PATH = "best_real_fake_resnet18.pt"

#initialize clients
datastore_client = datastore.Client(project=GCP_PROJECT)
storage_client = storage.Client(project=GCP_PROJECT)

# Load model from local file
def load_model():
    # Recreate the ResNet18 model architecture
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    
    # Modify final layer to match your saved model (sequential with dropout)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, 2)
    )
    
    # Load the saved weights
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

def save_to_datastore(prediction: int, label: str, confidence: float, ci: dict, filename: str) -> str:
    key = datastore_client.key("PredictionResult")
    entity = datastore.Entity(key)
    
    entity.update({
        "prediction": prediction,
        "prediction_label": label,
        "confidence": confidence,
        "confidence_interval": ci,
        "filename": filename,
        "timestamp": datetime.utcnow()
    })
    
    datastore_client.put(entity)
    return str(entity.key.id)

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        #read file content
        content = await file.read()
        
        #handle PDF or image
        if file.content_type == "application/pdf":
            #convert first page of PDF to image
            images = pdf2image.convert_from_bytes(content)
            image = images[0]
        else:
            #open as image
            image = Image.open(io.BytesIO(content))
        
        #preprocess image
        processed_image = preprocess_image(image)
        
        #run inference
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probabilities_np = probabilities.cpu().numpy()[0]
        
        #prediction and confidence
        predicted_class = int(np.argmax(probabilities_np))
        confidence = float(np.max(probabilities_np))
        
        #map to label
        label = "real" if predicted_class == 1 else "fake"
        
        #compute confidence interval
        ci = compute_confidence_interval(probabilities_np)
        
        #save to datastore
        result_id = save_to_datastore(
            prediction=predicted_class,
            label=label,
            confidence=confidence,
            ci=ci,
            filename=file.filename
        )
        
        return PredictionResponse(
            prediction=predicted_class,
            prediction_label=label,
            confidence=confidence,
            confidence_interval=ci,
            timestamp=datetime.utcnow().isoformat(),
            result_id=result_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
