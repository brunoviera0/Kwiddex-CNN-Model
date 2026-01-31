import hashlib
import json
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import uuid

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


#config
KEYS_DIR = Path("keys")
PRIVATE_KEY_PATH = KEYS_DIR / "kwiddex_private.pem"
PUBLIC_KEY_PATH = KEYS_DIR / "kwiddex_public.pem"

#PDF metadata keys for storing certificate
CERT_KEY = "/KwiddexCertificate"
SIG_KEY = "/KwiddexSignature"
VERSION_KEY = "/KwiddexVersion"


#RSA key pair: private key signs documents, public key verifies them.
#generate once during setup, keep private key SECRET.
def generate_key_pair(key_size: int = 2048) -> Tuple[bytes, bytes]:
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
        backend=default_backend()
    )
    
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return private_pem, public_pem


def setup_keys() -> None:
    KEYS_DIR.mkdir(exist_ok=True)
    
    if PRIVATE_KEY_PATH.exists():
        print(f"Keys already exist at {KEYS_DIR}/")
        print("Delete them manually to regenerate.")
        return
    
    print("Generating new RSA key pair")
    private_pem, public_pem = generate_key_pair()
    
    PRIVATE_KEY_PATH.write_bytes(private_pem)
    PUBLIC_KEY_PATH.write_bytes(public_pem)
    
    print(f"Private key saved to: {PRIVATE_KEY_PATH}")
    print(f"Public key saved to: {PUBLIC_KEY_PATH}")
    print("\nIMPORTANT: Keep the private key SECRET and back it up.")


def load_private_key():
    if not PRIVATE_KEY_PATH.exists():
        raise FileNotFoundError(
            f"Private key not found. Run 'python certification.py setup' first."
        )
    private_pem = PRIVATE_KEY_PATH.read_bytes()
    return serialization.load_pem_private_key(
        private_pem, password=None, backend=default_backend()
    )


def load_public_key():
    if not PUBLIC_KEY_PATH.exists():
        raise FileNotFoundError(
            f"Public key not found. Run 'python certification.py setup' first."
        )
    public_pem = PUBLIC_KEY_PATH.read_bytes()
    return serialization.load_pem_public_key(public_pem, backend=default_backend())

#hashing and signing
def hash_document(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def create_certificate_data(
    document_hash: str,
    confidence_score: float,
    reviewer_id: Optional[str] = None,
    client_reference: Optional[str] = None,
    notes: Optional[str] = None
) -> dict:

    return { #what the certification contains, can be modified
        "certificate_id": f"KWX-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}",
        "version": "1.0",
        "issuer": "Kwiddex Document Authentication System",
        "issued_at": datetime.utcnow().isoformat() + "Z",
        "document_hash": document_hash,
        "authenticity_confidence": confidence_score,
        "reviewer_id": reviewer_id,
        "client_reference": client_reference,
        "notes": notes,
        "status": "active"
    }


def sign_certificate(certificate_data: dict) -> str:
    private_key = load_private_key()
    cert_json = json.dumps(certificate_data, sort_keys=True).encode('utf-8')
    
    signature = private_key.sign(
        cert_json,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    
    return base64.b64encode(signature).decode('utf-8')


#verify if signature is valid
def verify_signature(certificate_data: dict, signature_b64: str) -> bool:
    try:
        public_key = load_public_key()
        cert_json = json.dumps(certificate_data, sort_keys=True).encode('utf-8')
        signature = base64.b64decode(signature_b64)
        
        public_key.verify(
            signature,
            cert_json,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    
    except InvalidSignature:
        return False
    
    except Exception as e:
        print(f"Verification error: {e}")
        return False


#PDF embedding and extraction
def embed_certificate_in_pdf(pdf_content: bytes, certificate: dict, signature: str) -> bytes:
    reader = PdfReader(io.BytesIO(pdf_content))
    writer = PdfWriter()
    
    for page in reader.pages:
        writer.add_page(page)
    
    cert_json = json.dumps(certificate, sort_keys=True)
    
    writer.add_metadata({
        CERT_KEY: cert_json,
        SIG_KEY: signature,
        VERSION_KEY: "1.0",
        "/Subject": f"Kwiddex Certified - {certificate.get('certificate_id', 'Unknown')}",
    })
    
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()



def extract_certificate_from_pdf(pdf_content: bytes) -> Optional[Tuple[dict, str]]:
    try:
        reader = PdfReader(io.BytesIO(pdf_content))
        metadata = reader.metadata
        
        if metadata is None:
            return None
        
        cert_json = metadata.get(CERT_KEY)
        signature = metadata.get(SIG_KEY)
        
        if cert_json is None or signature is None:
            return None
        
        certificate = json.loads(cert_json)
        return certificate, signature
    
    except Exception as e:
        print(f"Error extracting certificate: {e}")
        return None


def create_certificate_page(certificate: dict) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    #header
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 1 * inch, "CERTIFICATE OF AUTHENTICITY")
    
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, height - 1.4 * inch, "Issued by Kwiddex Document Authentication System")
    
    #line
    c.setStrokeColorRGB(0.2, 0.2, 0.7)
    c.setLineWidth(2)
    c.line(1 * inch, height - 1.6 * inch, width - 1 * inch, height - 1.6 * inch)
    
    #details
    y = height - 2.2 * inch
    line_height = 0.35 * inch
    
    details = [
        ("Certificate ID:", certificate.get("certificate_id", "N/A")),
        ("Issued:", certificate.get("issued_at", "N/A")),
        ("Document Hash:", certificate.get("document_hash", "N/A")[:32] + "..."),
        ("Confidence:", f"{certificate.get('authenticity_confidence', 0):.1%}"),
        ("Reviewer:", certificate.get("reviewer_id", "N/A")),
        ("Reference:", certificate.get("client_reference", "N/A")),
        ("Status:", certificate.get("status", "N/A").upper()),
    ]
    
    c.setFont("Helvetica-Bold", 11)
    for label, value in details:
        c.drawString(1.2 * inch, y, label)
        c.setFont("Helvetica", 11)
        c.drawString(3 * inch, y, str(value) if value else "N/A")
        c.setFont("Helvetica-Bold", 11)
        y -= line_height
    
    #footer
    y -= 0.5 * inch
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(width / 2, y, "This certificate is digitally signed and embedded in this PDF.")
    
    c.save()
    return buffer.getvalue()





def certify_pdf(
    pdf_content: bytes,
    confidence_score: float,
    reviewer_id: Optional[str] = None,
    client_reference: Optional[str] = None,
    notes: Optional[str] = None,
    add_visible_page: bool = True
) -> Tuple[bytes, dict]:
    """
    Certify a PDF document.

        pdf_content: Original PDF as bytes
        confidence_score: CNN model confidence (0.0 to 1.0)
        reviewer_id: Who approved the certification
        client_reference: Client's case/reference number
        notes: Additional notes
        add_visible_page: Append a human-readable certificate page
    
    Returns:
        Tuple of (certified_pdf_bytes, certificate_dict)
    """
    #hash the original document
    doc_hash = hash_document(pdf_content)
    
    #create certificate
    certificate = create_certificate_data(
        document_hash=doc_hash,
        confidence_score=confidence_score,
        reviewer_id=reviewer_id,
        client_reference=client_reference,
        notes=notes
    )
    
    #sign it
    signature = sign_certificate(certificate)
    
    #embed in PDF
    certified_pdf = embed_certificate_in_pdf(pdf_content, certificate, signature)
    
    #optionally add visible certificate page
    if add_visible_page:
        cert_page = create_certificate_page(certificate)
        
        writer = PdfWriter()
        reader = PdfReader(io.BytesIO(certified_pdf))
        
        for page in reader.pages:
            writer.add_page(page)
        
        cert_reader = PdfReader(io.BytesIO(cert_page))
        writer.add_page(cert_reader.pages[0])
        
        if reader.metadata:
            writer.add_metadata(dict(reader.metadata))
        
        output = io.BytesIO()
        writer.write(output)
        certified_pdf = output.getvalue()
    
    return certified_pdf, certificate


def verify_pdf(pdf_content: bytes) -> dict:
    #try to extract certificate
    extracted = extract_certificate_from_pdf(pdf_content)
    
    if extracted is None:
        return {
            "valid": False,
            "has_certificate": False,
            "message": "No Kwiddex certificate found in this PDF.",
            "certificate": None
        }
    
    certificate, signature = extracted
    
    #verify signature
    signature_valid = verify_signature(certificate, signature)
    
    #check status
    certificate_active = certificate.get("status") == "active"
    
    #result
    valid = signature_valid and certificate_active
    
    #build message
    if valid:
        cert_id = certificate.get("certificate_id", "Unknown")
        issued = certificate.get("issued_at", "Unknown")
        confidence = certificate.get("authenticity_confidence", 0)
        message = f"Valid. Certificate {cert_id} issued {issued}. Confidence: {confidence:.1%}"
    elif not signature_valid:
        message = "Invalid signature. Certificate may have been tampered with."
    elif not certificate_active:
        message = "Certificate has been revoked."
    else:
        message = "Verification failed."
    
    return {
        "valid": valid,
        "has_certificate": True,
        "signature_valid": signature_valid,
        "certificate_active": certificate_active,
        "message": message,
        "certificate": certificate
    }


def is_certified(pdf_content: bytes) -> bool: #check if PDF has a certificate or not
    return extract_certificate_from_pdf(pdf_content) is not None



if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Kwiddex Certification System")
        print("Usage:")
        print("  python certification.py setup    - Generate signing keys (run once)")
        print("  python certification.py test     - Test the system")
        print("  python certification.py certify <input.pdf> <output.pdf>")
        print("  python certification.py verify <certified.pdf>")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "setup":
        setup_keys()
    
    elif command == "test":
        print("Running certification test...\n")
        
        if not PRIVATE_KEY_PATH.exists():
            print("Keys not found. Run 'python certification.py setup' first.")
            sys.exit(1)
        
        #create test PDF
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.drawString(100, 700, "Test document for Kwiddex certification.")
        c.drawString(100, 680, f"Created: {datetime.now()}")
        c.save()
        test_pdf = buffer.getvalue()
        
        print(f"1. Created test PDF ({len(test_pdf)} bytes)")
        
        #certify
        certified, cert = certify_pdf(
            test_pdf, 
            confidence_score=0.94,
            reviewer_id="test@kwiddex.com",
            client_reference="TEST-001"
        )
        print(f"2. Certified ({len(certified)} bytes)")
        print(f"   Certificate ID: {cert['certificate_id']}")
        
        #save
        Path("test_certified.pdf").write_bytes(certified)
        print("3. Saved to test_certified.pdf")
        
        #verify
        result = verify_pdf(certified)
        print(f"4. Verification: {'PASSED' if result['valid'] else 'FAILED'}")
        print(f"   {result['message']}")
        
        print("\nTest complete!")
    
    elif command == "certify" and len(sys.argv) >= 4:
        input_path = sys.argv[2]
        output_path = sys.argv[3]
        
        with open(input_path, "rb") as f:
            pdf = f.read()
        
        certified, cert = certify_pdf(pdf, confidence_score=0.95, reviewer_id="cli")
        
        with open(output_path, "wb") as f:
            f.write(certified)
        
        print(f"Certified: {cert['certificate_id']}")
        print(f"Saved to: {output_path}")
    
    elif command == "verify" and len(sys.argv) >= 3:
        input_path = sys.argv[2]
        
        with open(input_path, "rb") as f:
            pdf = f.read()
        
        result = verify_pdf(pdf)
        print(f"Result: {'VALID' if result['valid'] else 'INVALID'}")
        print(f"Message: {result['message']}")
    
    else:
        print("Invalid command. Run without arguments for help.")
