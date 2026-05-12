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
    # Using gemini-flash-latest for stable and higher quota limits
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

def strip_markdown(text: str) -> str:
    """Remove markdown formatting characters from AI responses."""
    text = re.sub(r'#{1,6}\s*', '', text)  # Remove # headings
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)  # Remove *bold* and **bold**
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text, flags=re.DOTALL)  # Remove `code`
    text = re.sub(r'---+', '', text)  # Remove horizontal rules
    text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse multiple blank lines
    return text.strip()


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
    
    prompt = f"""
    Summarize this surgery: '{data.description}'.
    RETURN ONLY A VALID JSON OBJECT:
    {{
      "summary_en": "Concise English summary",
      "summary_ar": "ملخص مختصر بالعربية"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "summary_en": strip_markdown(result.get("summary_en", "Surgery recorded.")),
                "summary_ar": strip_markdown(result.get("summary_ar", "تم تسجيل العملية."))
            }
        return {"summary_en": strip_markdown(response.text), "summary_ar": strip_markdown(response.text)}
    except Exception as e:
        print(f"DEBUG: Error in summarize_surgery: {str(e)}")
        return {"summary_en": "Surgery recorded.", "summary_ar": "تم تسجيل العملية."}

@app.post("/analyze-history")
async def analyze_history(data: PatientHistoryInput):
    print(f"DEBUG: Received history data for analysis")
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")
    
    prompt = f"""
    Analyze the following patient medical history and provide a professional diagnosis summary and health insights.
    
    Vitals: {json.dumps(data.vitals)}
    Surgeries: {json.dumps(data.surgeries)}
    Medications: {json.dumps(data.medications)}
    Allergies: {json.dumps(data.allergies)}
    Chronic Diseases: {json.dumps(data.chronic_diseases)}
    
    RETURN ONLY A JSON OBJECT with these keys:
    {{
      "en": "Detailed analysis in English (plain text, no markdown)",
      "ar": "تحليل مفصل باللغة العربية (نص عادي، بدون مارك داون)"
    }}
    
    IMPORTANT: Do NOT use markdown (#, *, etc). Use plain text with line breaks.
    """
    
    prompt = f"""
    Analyze the following patient medical history:
    Vitals: {json.dumps(data.vitals)}
    Surgeries: {json.dumps(data.surgeries)}
    Medications: {json.dumps(data.medications)}
    Allergies: {json.dumps(data.allergies)}
    Chronic Diseases: {json.dumps(data.chronic_diseases)}
    
    You MUST provide two separate reports.
    
    REPORT 1: English ONLY. Professional medical tone. Plain text.
    REPORT 2: Arabic ONLY. Professional medical tone. Plain text.
    
    RETURN YOUR RESPONSE AS A VALID JSON OBJECT ONLY:
    {{
      "en": "FULL_ENGLISH_REPORT_HERE",
      "ar": "FULL_ARABIC_REPORT_HERE"
    }}
    
    CRITICAL: 
    - The 'en' field must NOT contain any Arabic characters.
    - The 'ar' field must NOT contain any English sentences (except medical terms if necessary).
    - Do NOT use markdown symbols like # or *.
    """
    
    try:
        response = model.generate_content(prompt)
        # Robust JSON extraction
        content = response.text.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "analysis_en": strip_markdown(result.get("en", "No English report generated.")),
                "analysis_ar": strip_markdown(result.get("ar", "لم يتم إنشاء تقرير بالعربية."))
            }
        
        # Fallback split logic if AI fails JSON but returns both
        text = strip_markdown(response.text)
        return {"analysis_en": text, "analysis_ar": text}
        
    except Exception as e:
        print(f"DEBUG: Error in analyze_history: {str(e)}")
        return {
            "analysis_en": "Your health profile is being monitored. Please consult your doctor.",
            "analysis_ar": "يتم مراقبة حالتك الصحية. يرجى استشارة طبيبك."
        }



@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), type: str = "prescription"):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")
    
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    if type == "prescription":
        prompt = "Carefully extract every detail from this medical prescription. I need medication names, dosages (e.g. 500mg), frequencies (e.g. twice daily), and durations. Provide a professional bilingual summary (English/Arabic) and the raw data in JSON format."
    else:
        prompt = "Analyze this lab result or medical scan. Extract all test names, values, units, and reference ranges. Provide a clear bilingual interpretation (English/Arabic) explaining if the results are within normal range."
    
    try:
        # For multimodal, we pass the image and the prompt
        response = model.generate_content([prompt, img])
        
        # Extract JSON from response (Gemini sometimes wraps it in markdown)
        text = response.text
        json_match = re.search(r'\[.*\]|\{.*\}', text, re.DOTALL)
        
        # Also ask Gemini for a very brief summary (for title) in both languages
        summary_resp = model.generate_content(f"Based on this extracted text, give me a very brief (3-4 words) title for this document in both English and Arabic (e.g. Blood Test / تحليل دم): {text}")
        summary = summary_resp.text.strip()

        if json_match:
            extracted_data = json.loads(json_match.group())
            # Split summary if it contains '/'
            s_en, s_ar = "Document", "مستند"
            if " / " in summary:
                parts = summary.split(" / ")
                s_en = parts[0].strip()
                s_ar = parts[1].strip()
            
            return {
                "data": extracted_data, 
                "raw_text": strip_markdown(text), 
                "summary_en": strip_markdown(s_en),
                "summary_ar": strip_markdown(s_ar)
            }
        else:
            return {"data": [], "raw_text": strip_markdown(text), "summary_en": "Document", "summary_ar": "مستند"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize-medical-item")
async def summarize_item(data: dict):
    """Refines medical descriptions for surgeries, diseases, or allergies."""
    item_type = data.get("type", "medical item")
    description = data.get("description", "")
    
    prompt = f"""
    Refine this {item_type}: '{description}'.
    RETURN ONLY A VALID JSON OBJECT:
    {{
      "summary_en": "Professional English version",
      "summary_ar": "نسخة احترافية بالعربية"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "summary_en": strip_markdown(result.get("summary_en", description)),
                "summary_ar": strip_markdown(result.get("summary_ar", description))
            }
        return {"summary_en": description, "summary_ar": description}
    except Exception as e:
        print(f"DEBUG: Error in summarize_item: {e}")
        return {"summary_en": description, "summary_ar": description}

@app.post("/analyze-vitals")
async def analyze_vitals(data: dict):
    """Analyzes vital signs and gives immediate advice."""
    vitals = data.get("vitals", [])
    patient_info = data.get("patient_info", {}) # age, weight, etc.
    
    prompt = f"""
    Analyze these vitals: {vitals}. Patient Info: {patient_info}.
    (BP rules: 12/8 is 120/80. Normal 120/80. High > 140/90. Emergency > 180/110).
    
    You MUST provide two separate advice reports.
    
    REPORT 1: English ONLY.
    REPORT 2: Arabic ONLY.
    
    RETURN ONLY A VALID JSON OBJECT:
    {{
      "en": "English advice here",
      "ar": "النصيحة بالعربية هنا"
    }}
    
    CRITICAL: No language mixing. No markdown symbols (#, *).
    """
    
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "advice_en": strip_markdown(result.get("en", "Stay safe.")),
                "advice_ar": strip_markdown(result.get("ar", "حافظ على صحتك."))
            }
        return {"advice_en": strip_markdown(response.text), "advice_ar": strip_markdown(response.text)}
    except Exception as e:
        print(f"DEBUG: Error in analyze_vitals: {e}")
        return {"advice_en": "Keep monitoring.", "advice_ar": "استمر في المتابعة."}

@app.post("/check-medication-safety")
async def check_medication(data: dict):
    """Checks for risks between a new medication and existing history."""
    new_med = data.get("medication", "")
    history = data.get("history", {})
    
    prompt = f"""
    Check if '{new_med}' is safe for this history: {history}.
    
    REPORT 1: English ONLY.
    REPORT 2: Arabic ONLY.
    
    RETURN ONLY A VALID JSON OBJECT:
    {{
      "en": "English safety report",
      "ar": "تقرير السلامة بالعربية"
    }}
    
    CRITICAL: No language mixing. No markdown symbols (#, *).
    """
    
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "safety_en": strip_markdown(result.get("en", "Check with doctor.")),
                "safety_ar": strip_markdown(result.get("ar", "استشر الطبيب."))
            }
        return {"safety_en": strip_markdown(response.text), "safety_ar": strip_markdown(response.text)}
    except Exception as e:
        print(f"DEBUG: Error in check_medication: {e}")
        return {"safety_en": "Consult doctor.", "safety_ar": "استشر الطبيب."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)