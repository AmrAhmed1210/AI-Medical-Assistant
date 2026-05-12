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
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
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
    vitals: List[dict]
    surgeries: List[dict]
    medications: List[dict]
    allergies: List[dict]
    chronic_diseases: List[dict]

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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-history")
async def analyze_history(data: PatientHistoryInput):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)