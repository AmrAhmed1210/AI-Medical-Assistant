import os
import io
import json
import re
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    print(f"DEBUG: API Key found (starts with: {GOOGLE_API_KEY[:5]}...)")
    genai.configure(api_key=GOOGLE_API_KEY)
    # Using gemini-flash-latest which is usually the best for free tier
    model = genai.GenerativeModel('gemini-flash-latest')
else:
    print("WARNING: GOOGLE_API_KEY not found in environment variables.")

app = FastAPI(title="Luxury Medical AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SurgeryInput(BaseModel):
    description: str

class PatientHistoryInput(BaseModel):
    vitals: Optional[List[dict]] = []
    surgeries: Optional[List[dict]] = []
    medications: Optional[List[dict]] = []
    allergies: Optional[List[dict]] = []
    chronic_diseases: Optional[List[dict]] = []

@app.get("/")
def root():
    return {"status": "ok", "message": "Luxury Medical AI API is active 🤖"}

@app.post("/summarize-surgery")
async def summarize_surgery(data: SurgeryInput):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")
    
    prompt = f"Summarize the following surgery details into a concise, professional medical note (max 2 sentences):\n{data.description}"
    
    try:
        response = model.generate_content(prompt)
        return {"summary": response.text.strip()}
    except Exception as e:
        print(f"DEBUG: Error in summarize_surgery: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-history")
async def analyze_history(data: PatientHistoryInput):
    print(f"DEBUG: Received history data for analysis")
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")
    
    context = f"""
    Analyze the following patient medical history and provide a professional diagnosis summary and health insights.
    
    Vitals: {json.dumps(data.vitals)}
    Surgeries: {json.dumps(data.surgeries)}
    Medications: {json.dumps(data.medications)}
    Allergies: {json.dumps(data.allergies)}
    Chronic Diseases: {json.dumps(data.chronic_diseases)}
    
    Provide the response in a structured format with:
    1. Overall Health Status
    2. Key Concerns
    3. Recommendations
    
    Keep it concise and professional.
    """
    
    try:
        response = model.generate_content(context)
        return {"analysis": response.text.strip()}
    except Exception as e:
        print(f"DEBUG: Error in analyze_history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), type: str = "prescription"):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")
    
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    if type == "prescription":
        prompt = "Extract medication names, dosages, frequencies, and durations from this prescription image. Return it as a JSON list of objects with keys: name, dose, frequency, duration."
    else:
        prompt = "Extract lab test results (test name, value, unit, reference range) from this image. Return it as a JSON list of objects."
    
    try:
        # For multimodal, we pass the image and the prompt
        response = model.generate_content([prompt, img])
        
        # Extract JSON from response (Gemini sometimes wraps it in markdown)
        text = response.text
        json_match = re.search(r'\[.*\]|\{.*\}', text, re.DOTALL)
        if json_match:
            extracted_data = json.loads(json_match.group())
            return {"data": extracted_data, "raw_text": text}
        else:
            return {"data": [], "raw_text": text}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize-medical-item")
async def summarize_item(data: dict):
    """Refines medical descriptions for surgeries, diseases, or allergies."""
    item_type = data.get("type", "medical item")
    description = data.get("description", "")
    
    prompt = f"As a medical assistant, refine this description of a {item_type}: '{description}'. Make it professional, concise, and correctly spelled in medical terms. Return only the refined text."
    
    try:
        response = model.generate_content(prompt)
        return {"refined_text": response.text.strip()}
    except Exception as e:
        print(f"DEBUG: Error in summarize_item: {e}")
        return {"refined_text": description} # Fallback to original

@app.post("/analyze-vitals")
async def analyze_vitals(data: dict):
    """Analyzes vital signs and gives immediate advice."""
    vitals = data.get("vitals", [])
    patient_info = data.get("patient_info", {}) # age, weight, etc.
    
    prompt = f"""
    Analyze these vitals: {vitals}. 
    Patient Info: {patient_info}.
    Provide a very short (1-2 sentences) medical advice or observation. 
    If values are dangerous, emphasize the need to see a doctor.
    Be encouraging but professional.
    """
    
    try:
        response = model.generate_content(prompt)
        return {"advice": response.text.strip()}
    except Exception as e:
        print(f"DEBUG: Error in analyze_vitals: {e}")
        return {"advice": "Keep monitoring your vitals and consult your doctor if you feel unwell."}

@app.post("/check-medication-safety")
async def check_medication(data: dict):
    """Checks for risks between a new medication and existing history."""
    new_med = data.get("medication", "")
    history = data.get("history", {})
    
    prompt = f"""
    Check if the medication '{new_med}' has any known major risks or contraindications with this patient history: {history}.
    If there's a risk, explain it briefly. If safe, say 'No major immediate risks found with your history'.
    Always end with 'Always consult your doctor before starting new medication'.
    """
    
    try:
        response = model.generate_content(prompt)
        return {"safety_report": response.text.strip()}
    except Exception as e:
        print(f"DEBUG: Error in check_medication: {e}")
        return {"safety_report": "Please consult your doctor to ensure this medication is safe for you."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)