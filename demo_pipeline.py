import io
import time
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
from auth import create_user, authenticate
import pdf2image
from certification import certify_document, verify_pdf, PRIVATE_KEY_PATH, setup_keys
import certificate_store

MODEL_PATH = "best_real_fake_resnet18.pt"

def load_model():
    model = resnet18(weights=None)
    
    #match saved model structure (sequential with dropout)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, 2)
    )
    
    #load weights
    if not Path(MODEL_PATH).exists():
        print(f"  WARNING: Model file '{MODEL_PATH}' not found.")
        print(f"           Demo will use simulated scores.")
        return None
    
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


#image preprocessing (match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_document(model, image: Image.Image):
    if model is None:
        #fallback to simulated score if model not available
        return "real", 0.94
    
    #preprocess
    image = image.convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    #run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probs = probabilities.cpu().numpy()[0]
    
    #get prediction (0 = fake, 1 = real)
    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs))
    label = "real" if predicted_class == 1 else "fake"
    
    return label, confidence


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step_num, text):
    print(f"\n[Step {step_num}] {text}")
    print("-" * 40)


def wait():
    time.sleep(1)


def create_test_document():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, 750, "SAMPLE DOCUMENT")
    c.setFont("Helvetica", 12)
    c.drawString(100, 680, f"Date: {datetime.now().strftime('%B %d, %Y')}")
    c.drawString(100, 640, "This document exists")
    c.save()
    return buffer.getvalue()


def create_test_image():
    img = Image.new('RGB', (800, 600), color=(255, 255, 255))
    for y in range(50, 550, 30):
        for x in range(50, 750, 2):
            img.putpixel((x, y), (0, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


def run_demo():
    print_header("Demo")
    print("\nThis demo shows the complete flow of the Kwiddex system:")
    print("  1. User account creation")
    print("  2. User login")
    print("  3. Document upload and scoring")
    print("  4. Document certification (digital signature)")
    print("  5. Certificate verification")
    print("  6. Certificate revocation")
    
    wait()
    

    print_step(1, "User Creation")
    
    test_username = f"demo_user_{datetime.now().strftime('%H%M%S')}"
    test_password = "demo_password_123"
    
    print(f"Creating user account...")
    print(f"  Username: {test_username}")
    print(f"  Password: {'*' * len(test_password)}")
    
    success, user_id = create_user(test_username, test_password)
    
    if success:
        print(f"\n  User created successfully!")
        print(f"  User ID: {user_id}")
    else:
        print(f"\n Error: {user_id}")
        return
    
    wait()
    

    print_step(2, "User Login")
    
    print(f"Authenticating user...")
    
    success, logged_in_user_id = authenticate(test_username, test_password)
    
    if success:
        print(f"\n  Login successful!")
        print(f"  Logged in as: {logged_in_user_id}")
    else:
        print(f"\n  Login failed")
        return
    
    wait()
    

    print_step(3, "Document upload and model scoring")
    
    print("Loading CNN model...")
    model = load_model()
    
    if model is not None:
        print(f"  Model loaded from {MODEL_PATH}")
    else:
        print(f"  Model not found, using simulated scores")
    
    print("\nCreating sample document...")
    document_pdf = create_test_document()
    print(f"  Created test PDF ({len(document_pdf)} bytes)")
    
    
    
    #Convert PDF to image for model
    print("\nConverting PDF to image for model input...")
    images = pdf2image.convert_from_bytes(document_pdf)
    document_image = images[0]
    print(f"  Converted to image ({document_image.size[0]}x{document_image.size[1]})")
    
    print("\nRunning document through CNN model...")
    prediction, confidence_score = predict_document(model, document_image)
    
    print(f"\n  Model prediction: {prediction.upper()}")
    print(f"  Confidence score: {confidence_score:.1%}")
    print(f"\n  The model is {confidence_score:.1%} confident this document is {prediction}.")
    
    wait()
    



    print_step(4, "Document Certification")
    
    
    #ensure keys exist
    if not PRIVATE_KEY_PATH.exists():
        print("Setting up signing keys...")
        setup_keys()
    
    print(f"Certifying document...")
    print(f"  Reviewer: {logged_in_user_id}")
    print(f"  Confidence: {confidence_score:.1%}")
    
    certified_pdf, certificate = certify_document(
        content=document_pdf,
        confidence_score=confidence_score,
        reviewer_id=logged_in_user_id,
        client_reference="DEMO-2026-001",
        notes="Demo certification"
    )
    
    print(f"\n  Document certified successfully!")
    print(f"  Certificate ID: {certificate['certificate_id']}")
    print(f"  Issued at: {certificate['issued_at']}")
    print(f"  Document hash: {certificate['document_hash'][:16]}...")
    print(f"  Reviewer ID embedded: {certificate['reviewer_id']}")
    
    #store certificate in Datastore
    print(f"\n  Storing certificate in Datastore...")
    try:
        entity_id = certificate_store.store_certificate(
            certificate_id=certificate["certificate_id"],
            document_hash=certificate["document_hash"],
            confidence_score=confidence_score,
            reviewer_id=logged_in_user_id,
            client_reference="DEMO-2026-001",
            original_filename="demo_document.pdf",
            notes="Demo certification"
        )
        print(f"  Stored in Datastore (entity ID: {entity_id})")
    except Exception as e:
        print(f"  WARNING: Could not store in Datastore: {e}")
        print(f"  (This is expected if running without GCP credentials)")
    
    
    #save certified document
    output_path = Path("demo_certified_document.pdf")
    output_path.write_bytes(certified_pdf)
    print(f"\n  Saved certified PDF to: {output_path}")
    
    wait()
   

  
    print_step(5, "Certificate Verification")
    
    print("Verifying certified document...")
    
    result = verify_pdf(certified_pdf)
    
    print(f"\n  Verification result:")
    print(f"    Valid: {'YES' if result['valid'] else 'NO'}")
    print(f"    Has certificate: {'YES' if result['has_certificate'] else 'NO'}")
    print(f"    Signature valid: {'YES' if result['signature_valid'] else 'NO'}")
    print(f"    Certificate active: {'YES' if result['certificate_active'] else 'NO'}")
    print(f"\n  Message: {result['message']}")
    
    if result['valid']:
        print(f"\n  This document is verified as certified by Kwiddex!")
        print(f"  It has not been modified since certification.")
    
    #look up certificate from Datastore
    print(f"\n  Looking up certificate in Datastore...")
    try:
        db_record = certificate_store.lookup_certificate(certificate['certificate_id'])
        if db_record:
            print(f"    Found in Datastore!")
            print(f"    Status: {db_record['status']}")
            print(f"    Stored at: {db_record['issued_at']}")
            print(f"    Reviewer: {db_record['reviewer_id']}")
        else:
            print(f"    Not found in Datastore (offline mode)")
    except Exception as e:
        print(f"    Could not query Datastore: {e}")
    
    wait()
    
 

    print_step(6, "Image Certification")
    
    print("Creating sample image (simulating phone photo of document)...")
    document_image_bytes = create_test_image()
    print(f"  Created test JPEG ({len(document_image_bytes)} bytes)")
    
    #score the image with the model
    print("\nRunning image through CNN model...")
    test_image = Image.open(io.BytesIO(document_image_bytes))
    image_prediction, image_confidence = predict_document(model, test_image)
    print(f"  Model prediction: {image_prediction.upper()}")
    print(f"  Confidence score: {image_confidence:.1%}")
    
    print("\nCertifying image (auto-converts to PDF)...")
    
    certified_image_pdf, image_cert = certify_document(
        content=document_image_bytes,
        confidence_score=image_confidence,
        reviewer_id=logged_in_user_id,
        client_reference="DEMO-IMG-001"
    )
    
    print(f"\n  Image certified")
    print(f"  Certificate ID: {image_cert['certificate_id']}")
    
    #store image certificate in Datastore
    try:
        certificate_store.store_certificate(
            certificate_id=image_cert["certificate_id"],
            document_hash=image_cert["document_hash"],
            confidence_score=image_confidence,
            reviewer_id=logged_in_user_id,
            client_reference="DEMO-IMG-001",
            original_filename="demo_image.jpg"
        )
        print(f"  Stored in Datastore")
    except Exception as e:
        print(f"  WARNING: Could not store in Datastore: {e}")
    
    #verify
    result = verify_pdf(certified_image_pdf)
    print(f"  Verification: {'VALID' if result['valid'] else 'INVALID'}")
    
    wait()
    


    print_step(7, "Certificate Revocation")
    
    print("Revoking certificate in Datastore...")
    print(f"  Certificate ID: {certificate['certificate_id']}")
    
    try:
        revoke_result = certificate_store.revoke_certificate(
            certificate['certificate_id'],
            reason="Demo revocation test"
        )
        
        if revoke_result["success"]:
            print(f"\n  Certificate revoked!")
            print(f"  Revoked at: {revoke_result['revoked_at']}")
            print(f"  Reason: {revoke_result['reason']}")
        else:
            print(f"\n  Revocation result: {revoke_result['message']}")
        
        #confirm revocation by looking it up
        print(f"\n  Confirming revocation status...")
        status = certificate_store.check_revocation_status(certificate['certificate_id'])
        print(f"  Datastore status: {status}")
        
        #reverify the same certified PDF, should now fail if API checks Datastore
        print(f"\n  Note: The PDF's embedded signature is still cryptographically valid.")
        print(f"  However, the /verify-certificate API endpoint now checks Datastore")
        print(f"  and will report this certificate as revoked.")
        
    except Exception as e:
        print(f"\n  Could not revoke in Datastore: {e}")
        print(f"  (This is expected if running without GCP credentials)")
    
    wait()
    
   
    print_header("DEMO COMPLETE")
    


if __name__ == "__main__":
    run_demo()
