import os
import io
import json
import re
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from typing import List, Optional, Any
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

# Internal security key for ASP.NET to FastAPI communication
INTERNAL_SECRET_KEY = "LuxuryMedicalAiSecretKey2026"

def verify_internal_token(x_internal_token: str = Header(None)):
    if not x_internal_token or x_internal_token != INTERNAL_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing internal token")

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
    vitals: Optional[List[Any]] = []
    surgeries: Optional[List[Any]] = []
    medications: Optional[List[Any]] = []
    allergies: Optional[List[Any]] = []
    chronic_diseases: Optional[List[Any]] = []
    documents_analysis: Optional[List[Any]] = []
    recommended_doctors: Optional[List[Any]] = []
    recommended_specialty: Optional[str] = None

class PreVisitInput(BaseModel):
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    chronic_diseases: Optional[List[Any]] = []
    medications: Optional[List[Any]] = []
    allergies: Optional[List[Any]] = []
    vitals: Optional[List[Any]] = []

class PersonalizedTipInput(BaseModel):
    patient_id: str
    chronic_diseases: Optional[List[str]] = []

class MessageDto(BaseModel):
    role: str
    content: str
    
class AskRequest(BaseModel):
    question: str
    history: Optional[List[MessageDto]] = None

@app.get("/")
def root():
    return {"status": "ok", "message": "Luxury Medical AI API is active 🤖"}

@app.post("/summarize-surgery", dependencies=[Depends(verify_internal_token)])
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

@app.post("/ask", dependencies=[Depends(verify_internal_token)])
async def ask_endpoint(data: AskRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")
    
    chat_history = []
    if data.history:
        for msg in data.history:
            # map roles to 'user' and 'model'
            role = "user" if msg.role.lower() == "user" else "model"
            chat_history.append({"role": role, "parts": [msg.content]})
            
    try:
        chat = model.start_chat(history=chat_history)
        prompt = f"""
        You are an advanced, professional medical AI assistant.
        The user is asking: {data.question}
        
        CRITICAL RULES:
        1. STRICTLY MEDICAL: You are a medical assistant. You are ONLY allowed to answer medical, health, and wellness related questions. If the user asks about programming (like Python), tech, math, general knowledge, or anything outside the medical field, you MUST politely apologize and state that you are a specialized medical assistant and can only help with health-related matters.
        2. Always respond in the SAME language the user is speaking.
        3. Keep it professional, empathetic, and highly informative.
        4. DO NOT use markdown symbols like # or *. Return beautiful plain text with elegant spacing and emojis where appropriate.
        5. ACTIVE INQUIRY: If the user complains of pain, tiredness, or any symptoms, DO NOT just give a final diagnosis. Instead, ALWAYS end your response by asking 1 or 2 relevant follow-up questions to gather more details. Keep investigating until you have a clear picture.
        6. Once you have a clear picture, provide safe advice and ALWAYS conclude with a recommendation to see a doctor.
        7. NEVER say or imply that you are a doctor. Say you are an AI medical assistant.
        8. If the user's message indicates they are perfectly healthy with no problems, do NOT recommend any doctors or specialties, just encourage a healthy lifestyle.
        9. If the prompt includes platform doctor recommendations context AND the user has a medical issue, use ONLY those doctors when suggesting doctors, prioritize matching specialty and highest rating/review count, and do not invent doctors.
        """
        response = chat.send_message(prompt)
        return {
            "query": data.question,
            "reply": strip_markdown(response.text),
            "model_used": "gemini-flash-latest",
            "is_medical": True,
            "found_in_database": False,
            "low_confidence": False,
            "language": "ar" if any("\u0600" <= c <= "\u06FF" for c in data.question) else "en",
            "disclaimer": "This is an AI response and should not replace professional medical advice."
        }
    except Exception as e:
        print(f"DEBUG: Error in /ask: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-history", dependencies=[Depends(verify_internal_token)])
async def analyze_history(data: PatientHistoryInput):
    print(f"DEBUG: Received history data for analysis. Documents analysis: {data.documents_analysis}")
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")
    
    prompt = f"""
    Analyze the following patient medical history and provide a professional diagnosis summary and health insights.
    
    Vitals: {json.dumps(data.vitals)}
    Surgeries: {json.dumps(data.surgeries)}
    Medications: {json.dumps(data.medications)}
    Allergies: {json.dumps(data.allergies)}
    Chronic Diseases: {json.dumps(data.chronic_diseases)}
    AI Analyzed Documents: {json.dumps(data.documents_analysis)}
    
    CRITICAL INSTRUCTION: You MUST read the 'AI Analyzed Documents' provided. If there are any lab results, prescriptions, or medical scans inside 'AI Analyzed Documents', you MUST explicitly mention them in your final report and incorporate any abnormalities, risks, or new medications into your overall diagnosis.
    
    ACTIONABLE ADVICE INSTRUCTION: If you detect any abnormal vitals or health issues, you MUST provide simple, safe, non-medical home remedies or quick first-aid tips (e.g., "If your blood pressure is low, try drinking some water or having a light salty snack"). Always follow these tips with the phrase: "However, please consult a doctor for professional medical advice" (or its Arabic equivalent).
    
    EXPANDED LIFESTYLE & WELLNESS ADVICE: You MUST provide 3 to 4 detailed, highly practical, and safe wellness and preventative lifestyle tips tailored to the patient's medical history (e.g. dietary recommendations, foods to avoid, exercise guidance, sleep, stress reduction, and hydration tips based on their chronic diseases or vitals). Make this advice extremely rich, rich in detail, and reassuring.
    

    RETURN ONLY A JSON OBJECT with these keys:
    {{
      "en": "Detailed professional analysis in English, cleanly structured using these exact section headers (with emojis, plain text, no markdown, separate sections with newlines):\n🩺 General Summary:\n...\n⚠️ Health Risks & Warnings:\n...\n💊 Medications & Interactions:\n...\n💡 Home Remedies & Non-Medical Advice:\n...",
      "ar": "تحليل طبي مفصل ومنظم للغاية باللغة العربية، مقسم إلى الأقسام التالية بشكل منسق وجذاب (بدون مارك داون، مع ترك أسطر فارغة بين الأقسام):\n🩺 الملخص الصحي العام:\n...\n⚠️ التحذيرات والمخاطر الطبية:\n...\n💊 الأدوية والتداخلات الدوائية:\n...\n💡 النصائح والإسعافات المنزلية السريعة (مثل تناول شيء مالح للضغط المنخفض، مع جملة الاستشارة الطبية):\n..."
    }}
    
    IMPORTANT: Do NOT use markdown (#, *, etc). Use plain text with emojis and line breaks to create a clean, elegant layout.
    """
    
    prompt = f"""
    Analyze the following patient medical history:
    Vitals: {json.dumps(data.vitals)}
    Surgeries: {json.dumps(data.surgeries)}
    Medications: {json.dumps(data.medications)}
    Allergies: {json.dumps(data.allergies)}
    Chronic Diseases: {json.dumps(data.chronic_diseases)}
    
    You MUST provide two separate reports.
    
    IMPORTANT INSTRUCTION: You must explicitly analyze this history for any HEALTH RISKS, DRUG INTERACTIONS, or DANGEROUS VITALS. If you detect any danger, allergy conflict, or high risk based on their vitals and chronic diseases, YOU MUST START THE REPORT WITH AN EXPLICIT WARNING (e.g. "WARNING: ...").
    
    REPORT 1: English ONLY. Professional medical tone. Plain text. Include risk analysis.
    REPORT 2: Arabic ONLY. Professional medical tone. Plain text. Include risk analysis (تحذير: ...).
    
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

    prompt = f"""
    Analyze this patient medical history and return two plain-text reports as valid JSON only.

    Patient data:
    Vitals: {json.dumps(data.vitals)}
    Surgeries: {json.dumps(data.surgeries)}
    Medications: {json.dumps(data.medications)}
    Allergies: {json.dumps(data.allergies)}
    Chronic Diseases: {json.dumps(data.chronic_diseases)}
    AI Analyzed Documents: {json.dumps(data.documents_analysis)}
    Recommended Specialty: {data.recommended_specialty or "best available match"}
    Platform Doctors Sorted By Rating/Reviews: {json.dumps(data.recommended_doctors)}

    Instructions:
    - Identify health risks, dangerous vitals, allergy conflicts, and medication interactions.
    - If there is a serious risk, start that report with an explicit warning.
    - Mention relevant uploaded document findings when present.
    - Give safe, simple home-care tips only when appropriate, then recommend consulting a doctor.
    - CRITICAL: If the patient is healthy and has no medical problems, abnormalities, or concerning symptoms, do NOT recommend any specialty or doctors. Instead, state that their health profile looks good and encourage maintaining a healthy lifestyle.
    - If platform doctors are provided AND the patient actually has a medical issue, include a doctor recommendation section using only those doctors. Recommend the highest-rated relevant doctors by name, specialty, rating/review count, and why they fit the patient's likely specialty need.
    - If a Recommended Specialty is provided but Platform Doctors is empty AND the patient has a medical issue, you MUST explicitly state in the report that there are no doctors available currently on our platform for this specialty, and politely advise the patient to seek an external consultation with a <Recommended Specialty> doctor. Do NOT hallucinate any doctor names.
    - Do not claim you are a doctor. You are an AI medical assistant and your advice does not replace professional care.
    - Do not use markdown symbols like # or *.

    Return this JSON shape only:
    {{
      "en": "English report with sections separated by double newlines (\\n\\n). Sections: General Summary, Warnings, Medications & Interactions, Safe Advice, Recommended Doctors (if any).",
      "ar": "تقرير عربي بأقسام مفصولة بأسطر جديدة (\\n\\n): الملخص العام، التحذيرات، الأدوية والتداخلات، نصائح آمنة، الأطباء المقترحون من الموقع (إن وجد).",
      "needsDoctor": true // set to false ONLY if the patient is completely healthy and has no medical problems.
    }}

    Language rules:
    - The en field must be English only.
    - The ar field must be Arabic only except doctor names and necessary medical terms.
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
                "analysis_ar": strip_markdown(result.get("ar", "لم يتم إنشاء تقرير بالعربية.")),
                "needsDoctor": result.get("needsDoctor", True)
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



@app.post("/analyze-image", dependencies=[Depends(verify_internal_token)])
async def analyze_image(
    file: UploadFile = File(...), 
    type: str = Form("prescription"),
    patient_context: Optional[str] = Form(None)
):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")
    
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    context_str = f"\n\nPATIENT MEDICAL HISTORY CONTEXT:\n{patient_context}\n" if patient_context else ""
    
    format_rules = "CRITICAL: DO NOT use markdown tables or markdown formatting (#, *, etc.) under any circumstances. Format the output strictly as clear, plain text using line breaks and beautiful section headers with emojis. Use clean spacing between sections to ensure a highly organized layout."
    actionable_advice = """
    ACTIONABLE ADVICE: If you detect any health issues, abnormal results, or dangers, provide simple, safe, non-medical home remedies or quick first-aid tips (e.g., 'If your blood sugar is low, try having something sugary'). Always follow these tips with the phrase 'However, please consult a doctor for professional medical advice' (or its Arabic equivalent).
    
    EXPANDED LIFESTYLE & WELLNESS ADVICE: Provide 3 to 4 extremely detailed, highly practical advice on patient wellness, recovery, and daily lifestyle adjustments related to these results (e.g. hydration, recommended foods, foods to avoid, physical activity precautions, and symptom monitoring tips). Keep this advice extremely rich, comprehensive, and actionable.
    """
    
    if type == "prescription":
        prompt = f"""
        Carefully extract every detail from this medical prescription. I need medication names, dosages, frequencies, and durations.{context_str} 
        IMPORTANT: You must also analyze the prescription for any potential risks, drug interactions, or critical warnings, ESPECIALLY in relation to the PATIENT MEDICAL HISTORY CONTEXT (if provided). Check for allergic reactions or conflicts with their current chronic diseases and medications. If there are any dangers, start your response with an explicit warning (e.g. تحذير هام:). {actionable_advice} 
        
        Provide a professional detailed Arabic summary, strictly organized into these sections:
        🩺 الملخص العام للروشتة:
        [اكتب هنا ملخص الأدوية المستخرجة بشكل مرتب]
        
        ⚠️ التحذيرات والتفاعلات الدوائية:
        [اكتب هنا أي تعارضات مع حالته المرضية أو حساسيته إن وجدت]
        
        💡 الإرشادات والنصائح المنزلية:
        [اكتب هنا نصائح إضافية للاستخدام الآمن للأدوية]
        
        {format_rules}
        """
    else:
        prompt = f"""
        Analyze this lab result or medical scan. Extract all test names, values, units, and reference ranges.{context_str} 
        IMPORTANT: You must analyze the results for any abnormalities, health risks, or dangerous levels, ESPECIALLY in relation to the PATIENT MEDICAL HISTORY CONTEXT (if provided). If there are any dangers, start your response with an explicit warning (e.g. تحذير هام:). {actionable_advice} 
        
        Provide a clear Arabic interpretation explaining if the results are within normal range or if there is any danger, strictly organized into these sections:
        🩺 نتائج التحاليل المستخرجة:
        [اكتب هنا قائمة التحاليل والقيم والنسب بشكل منظم]
        
        ⚠️ المخاطر والمؤشرات غير الطبيعية:
        [اكتب هنا أي قراءة مرتفعة أو منخفضة تشكل خطورة وتفسيرها]
        
        💡 الإسعافات والنصائح المنزلية السريعة:
        [اكتب هنا نصائح آمنة وبسيطة حتى يستشير الطبيب]
        
        {format_rules}
        """
    
    try:
        # For multimodal, we pass the image and the prompt
        response = model.generate_content([prompt, img])
        
        # Get the text from the response
        text = response.text
        
        # We also ask for technical details (extracted JSON)
        tech_prompt = f"Based on this extracted text, provide the raw medical data (test names, values, or medication list) as a plain text summary or JSON: {text}"
        tech_response = model.generate_content(tech_prompt)
        tech_details = tech_response.text.strip()
        
        return {
            "status": "success", 
            "analysis_ar": strip_markdown(text), 
            "technical_details": strip_markdown(tech_details),
            "model_used": "gemini-flash-latest",
            "disclaimer": "تحذير: هذا التحليل بواسطة الذكاء الاصطناعي ولا يغني عن استشارة الطبيب."
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize-medical-item", dependencies=[Depends(verify_internal_token)])
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

@app.post("/analyze-vitals", dependencies=[Depends(verify_internal_token)])
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

@app.post("/check-medication-safety", dependencies=[Depends(verify_internal_token)])
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

@app.post("/doctor-ai-assist", dependencies=[Depends(verify_internal_token)])
async def doctor_ai_assist(data: dict):
    """Provides AI help for doctors while filling visit records."""
    chief_complaint = data.get("chief_complaint", "")
    history = data.get("history_of_illness", "")
    vitals = data.get("vitals", {})
    background = data.get("background", {})
    
    print(f"DEBUG: Doctor AI Assist request for: {chief_complaint[:20]}...")

    prompt = f"""
    You are a professional medical assistant helping a doctor. 
    
    Patient Context:
    - Chief Complaint: {chief_complaint}
    - History of Present Illness: {history}
    - Vitals: {json.dumps(vitals)}
    - Medical Background: {json.dumps(background)}
    
    Based on this complete profile, suggest:
    1. A list of possible differential diagnoses (Assessment).
    2. A suggested treatment plan (Plan).
    
    CRITICAL: Take the Medical Background (allergies, chronic diseases) into account for the treatment plan!
    
    RETURN ONLY A VALID JSON OBJECT:
    {{
      "assessment_en": "...",
      "assessment_ar": "...",
      "plan_en": "...",
      "plan_ar": "..."
    }}
    
    Keep it professional, evidence-based, and concise. Use plain text only.
    """
    
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "assessment_en": strip_markdown(result.get("assessment_en", "")),
                "assessment_ar": strip_markdown(result.get("assessment_ar", "")),
                "plan_en": strip_markdown(result.get("plan_en", "")),
                "plan_ar": strip_markdown(result.get("plan_ar", ""))
            }
        return {"error": "Could not parse AI response"}
    except Exception as e:
        print(f"DEBUG: Error in doctor_ai_assist: {e}")
        return {"error": str(e)}

class MedicalProfileInput(BaseModel):
    text: str

@app.post("/parse-medical-profile", dependencies=[Depends(verify_internal_token)])
async def parse_medical_profile(data: MedicalProfileInput):
    """Parse natural language medical text into structured data (chronic diseases, medications, allergies)."""
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")

    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    prompt = f"""
    You are a medical data extraction AI. Analyze the following patient description and extract all medical information mentioned.
    The patient may write in Arabic, English, or a mix of both. You MUST understand both languages perfectly.
    
    Patient says: "{data.text.strip()}"
    
    Extract and return ONLY a valid JSON object with these arrays:
    {{
      "chronic_diseases": [
        {{
          "diseaseName": "Professional English name of the disease",
          "diseaseNameAr": "اسم المرض بالعربية",
          "diseaseType": "Category (e.g., Endocrine, Cardiovascular, Respiratory, Neurological, Musculoskeletal, Gastrointestinal, Other)",
          "severity": "Mild | Moderate | Severe",
          "diagnosedDate": "YYYY-MM-DD if mentioned, otherwise null",
          "notes": "Any additional details mentioned"
        }}
      ],
      "medications": [
        {{
          "medicationName": "Brand name if mentioned, otherwise generic name",
          "genericName": "Generic/scientific name",
          "dosage": "e.g., 500mg",
          "form": "Tablet | Capsule | Syrup | Injection | Cream | Inhaler | Drops | Other",
          "frequency": "e.g., Twice daily, Once daily, As needed",
          "instructions": "Any special instructions mentioned (e.g., before meals, with water)",
          "doseTimes": "e.g., 08:00, 20:00 (if the user mentioned times, try to format them in 24h, else null)",
          "isChronic": true or false based on context
        }}
      ],
      "allergies": [
        {{
          "allergenName": "Name of the allergen in English",
          "allergenNameAr": "اسم المادة المسببة للحساسية بالعربية",
          "allergyType": "Drug | Food | Environmental | Insect | Latex | Other",
          "severity": "Mild | Moderate | Severe",
          "reactionDescription": "Description of the allergic reaction if mentioned"
        }}
      ],
      "summary_ar": "ملخص جميل ومنظم بالعربية لما تم استخراجه من بيانات المريض. اكتب بأسلوب ودود ومطمئن. استخدم إيموجي مناسبة. إذا لاحظت أي تداخل دوائي خطير أو أعراض مقلقة، يجب أن تضع تحذيراً واضحاً هنا (مثال: ⚠️ تحذير: ...). اذكر كل عنصر تم استخراجه بوضوح.",
      "summary_en": "A friendly, well-organized English summary of what was extracted. Use emojis. If you notice any dangerous drug interactions or alarming symptoms, put a clear warning here. Mention each extracted item clearly.",
      "follow_up_ar": "سؤال متابعة بالعربية. اسأل عن التفاصيل الناقصة (مثال: إذا ذكر المريض دواء ولم يذكر الجرعة أو المواعيد، اسأله عنها: 'امتى بتاخد الدواء ده وتحديداً كم مللي؟'). وإذا كانت البيانات مكتملة، اسأل إذا كان يود إضافة شيء آخر كعمليات جراحية أو حساسيات.",
      "follow_up_en": "A follow-up question in English asking for missing details (like dose times) or if they want to add anything else."
    }}
    
    RULES:
    1. If the patient mentions a disease in Arabic slang (e.g., "سكر" = Diabetes, "ضغط" = Hypertension), translate it professionally.
    2. If a medication is mentioned in Arabic (e.g., "جلوكوفاج"), use the proper brand name (Glucophage).
    3. If duration is mentioned (e.g., "من 5 سنين"), calculate the approximate diagnosed date.
    4. If the user mentions a medication without dose/times, YOU MUST ask about them in the follow_up_ar.
    5. If no items are found for a category, return an empty array [].
    6. Be thorough — extract EVERY medical detail mentioned.
    7. RETURN ONLY THE JSON OBJECT. No markdown, no extra text.
    """

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()

        # Robust JSON extraction
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Ensure all required keys exist
            result.setdefault("chronic_diseases", [])
            result.setdefault("medications", [])
            result.setdefault("allergies", [])
            result.setdefault("summary_ar", "تم تحليل البيانات.")
            result.setdefault("summary_en", "Data parsed successfully.")
            result.setdefault("follow_up_ar", "هل تحب تضيف حاجة تانية؟")
            result.setdefault("follow_up_en", "Would you like to add anything else?")
            return result

        return {
            "chronic_diseases": [], "medications": [], "allergies": [],
            "summary_ar": "لم أتمكن من استخراج بيانات طبية. حاول وصف حالتك بشكل أوضح.",
            "summary_en": "Could not extract medical data. Please describe your condition more clearly.",
            "follow_up_ar": "ممكن تقولي ايه الأمراض أو الأدوية اللي عندك؟",
            "follow_up_en": "Can you tell me about your diseases or medications?"
        }
    except Exception as e:
        print(f"DEBUG: Error in parse_medical_profile: {e}")
        raise HTTPException(status_code=500, detail=f"AI parsing error: {str(e)}")

@app.post("/pre-visit-summary", dependencies=[Depends(verify_internal_token)])
async def pre_visit_summary(data: PreVisitInput):
    """Generates a 30-second AI summary for the doctor before the patient enters."""
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")
        
    prompt = f"""
    You are an expert AI clinical assistant. Your task is to read the patient's data and generate a highly concise "Pre-visit Summary" (الملخص الاستباقي) for the doctor to read in 30 seconds before the patient enters the clinic.
    
    Patient ID: {data.patient_id}, {data.age} years old, {data.gender}
    Chief Complaint / Reason for visit: {data.chief_complaint}
    Chronic Diseases: {json.dumps(data.chronic_diseases)}
    Medications: {json.dumps(data.medications)}
    Allergies: {json.dumps(data.allergies)}
    Recent Vitals: {json.dumps(data.vitals)}
    
    RETURN ONLY A VALID JSON OBJECT with these keys:
    {{
      "summary_en": "A highly concise, bulleted summary in English. Include: 1) Main reason for visit, 2) Relevant history/vitals, 3) AI Alerts (e.g. drug interactions, abnormal vitals). Use plain text and emojis, NO MARKDOWN.",
      "summary_ar": "ملخص مركز جداً بالعربية في نقاط. يحتوي على: 1) سبب الزيارة الأساسي، 2) التاريخ المرضي/العلامات الحيوية ذات الصلة، 3) تنبيهات الذكاء الاصطناعي (مثل تعارض أدوية أو قراءات غير طبيعية). استخدم نص عادي وإيموجي، بدون مارك داون."
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "summary_en": strip_markdown(result.get("summary_en", "Patient is ready.")),
                "summary_ar": strip_markdown(result.get("summary_ar", "المريض جاهز للزيارة."))
            }
        return {"summary_en": "Patient is ready.", "summary_ar": "المريض جاهز للزيارة."}
    except Exception as e:
        print(f"DEBUG: Error in pre_visit_summary: {e}")
        return {"summary_en": "Error generating summary.", "summary_ar": "حدث خطأ أثناء توليد الملخص."}

@app.post("/personalized-tip", dependencies=[Depends(verify_internal_token)])
async def personalized_tip(data: PersonalizedTipInput):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")
        
    diseases_context = ", ".join(data.chronic_diseases) if data.chronic_diseases else "No chronic diseases"
    
    prompt = f"""
    You are a professional health and wellness coach. Generate ONE highly practical, safe, and motivating daily health tip (max 2 sentences) customized for a patient with the following chronic diseases: {diseases_context}.
    If there are no chronic diseases, provide a general excellent wellness tip (e.g. hydration, posture, sleep).
    If they have diabetes, mention something relevant like checking feet or balancing carbs safely. If hypertension, mention salt intake or stress reduction. Make it warm and encouraging.
    
    RETURN ONLY A VALID JSON OBJECT:
    {{
      "tip_en": "English tip here",
      "tip_ar": "نصيحة طبية ودودة بالعربية هنا"
    }}
    """
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "tip_en": strip_markdown(result.get("tip_en", "Stay hydrated and take short walks throughout your day.")),
                "tip_ar": strip_markdown(result.get("tip_ar", "حافظ على شرب الماء وامشِ قليلاً خلال يومك."))
            }
        return {"tip_en": "Stay healthy and active.", "tip_ar": "حافظ على صحتك ونشاطك."}
    except Exception as e:
        print(f"DEBUG: Error in personalized_tip: {e}")
        return {"tip_en": "Stay hydrated.", "tip_ar": "حافظ على شرب الماء."}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
