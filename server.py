"""
MedBook – TrOCR FastAPI Server
Run: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import io
import re

# ── Model path ──────────────────────────────────────────────
MODEL_PATH = "E:/AI_project/trocr_model"
# ────────────────────────────────────────────────────────────

app = FastAPI(title="MedBook OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ Loading TrOCR model...")
processor = TrOCRProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model loaded on {device}")

# ── Drug list ────────────────────────────────────────────────
DRUG_NAMES = [
    "paracetamol","amoxicillin","ibuprofen","omeprazole","metformin",
    "aspirin","cetirizine","atorvastatin","metronidazole","doxycycline",
    "vitamin d","calcium","iron","folic acid","zinc","pantoprazole",
    "clarithromycin","levothyroxine","amlodipine","omega-3",
]

def parse_prescription(text: str):
    lower = text.lower()
    name  = next((d.title() for d in DRUG_NAMES if d in lower), "Unknown")

    dose_m = re.search(r"(\d+\.?\d*)\s*(mg|mcg|ml|g|iu|%)", text, re.I)
    dose   = dose_m.group(0) if dose_m else "N/A"

    freq_map = [
        (r"once\s+daily|1\s*x\s*daily|\bod\b",   "Once daily"),
        (r"twice\s+daily|2\s*x\s*daily|\bbd\b",   "Twice daily"),
        (r"3\s+times|tds|tid|3x",                  "3 times daily"),
        (r"every\s+8\s+hours|q8h",                 "Every 8 hours"),
        (r"every\s+12\s+hours|q12h",               "Every 12 hours"),
        (r"at\s+night|before\s+sleep|bedtime",      "At night"),
        (r"when\s+needed|as\s+needed|prn",          "When needed"),
    ]
    freq = next((label for pattern, label in freq_map if re.search(pattern, text, re.I)), "N/A")

    dur_m = re.search(r"(\d+)\s*(day|days|week|weeks|month|months)", text, re.I)
    dur   = dur_m.group(0) if dur_m else "N/A"

    return {"name": name, "dose": dose, "frequency": freq, "duration": dur}


@app.get("/")
def root():
    return {"status": "ok", "message": "MedBook OCR API is running 🚀"}


@app.post("/scan")
async def scan_prescription(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run TrOCR
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Parse
    medicine = parse_prescription(text)

    return {
        "text":      text,
        "medicines": [medicine],
    }