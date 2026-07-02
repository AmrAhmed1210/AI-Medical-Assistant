import os
import io
import json
import re
import time
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, field_validator
from typing import List, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
if GOOGLE_API_KEY:
    print(f"DEBUG: API Key found (starts with: {GOOGLE_API_KEY[:5]}...)")
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    print(f"DEBUG: Gemini model configured: {GEMINI_MODEL}")
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

# --- Retry wrapper for Gemini API calls ---
import asyncio

async def generate_with_retry(prompt_or_parts, max_retries=3, initial_wait=15):
    """Calls model.generate_content with automatic retry on 429 rate limit errors."""
    for attempt in range(max_retries + 1):
        try:
            # Run the synchronous API call in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, model.generate_content, prompt_or_parts)
            return response
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries:
                # Extract retry delay from error message if available
                wait_time = initial_wait * (2 ** attempt)  # exponential backoff: 15s, 30s, 60s
                # Try to parse the suggested retry time from the error
                retry_match = re.search(r'retry in (\d+(?:\.\d+)?)', error_str, re.IGNORECASE)
                if retry_match:
                    wait_time = max(float(retry_match.group(1)) + 2, wait_time)  # add 2s buffer
                print(f"DEBUG: Rate limited (429). Waiting {wait_time:.0f}s before retry {attempt + 1}/{max_retries}...")
                await asyncio.sleep(wait_time)
            else:
                raise  # Re-raise non-429 errors or if max retries exhausted

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
    recent_visits: Optional[List[Any]] = []
    recommended_doctors: Optional[List[Any]] = []
    recommended_specialty: Optional[str] = None

    # NOTE: these "before" validators are a defensive fix for the /analyze-history
    # 422 - they coerce whatever shape the .NET side sends (null, a single
    # object instead of an array, etc.) into something this model accepts,
    # instead of FastAPI rejecting the whole request before it reaches the
    # endpoint. If a 422 still happens after this, paste the literal 422
    # "detail" array so the remaining mismatch can be pinned exactly.
    @field_validator(
        "vitals", "surgeries", "medications", "allergies",
        "chronic_diseases", "documents_analysis", "recent_visits",
        "recommended_doctors",
        mode="before",
    )
    @classmethod
    def _coerce_to_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]

    @field_validator("recommended_specialty", mode="before")
    @classmethod
    def _coerce_specialty_to_str(cls, v):
        if v is None or isinstance(v, str):
            return v
        return str(v)

class PreVisitInput(BaseModel):
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    chronic_diseases: Optional[List[Any]] = []
    medications: Optional[List[Any]] = []
    allergies: Optional[List[Any]] = []
    vitals: Optional[List[Any]] = []

    @field_validator(
        "chronic_diseases", "medications", "allergies", "vitals", mode="before"
    )
    @classmethod
    def _coerce_to_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]

class PersonalizedTipInput(BaseModel):
    patient_id: str
    chronic_diseases: Optional[List[str]] = []

    @field_validator("chronic_diseases", mode="before")
    @classmethod
    def _coerce_to_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]

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
        response = await generate_with_retry(prompt)
        content = response.text.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(), strict=False)
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

    # ── Extract original user question from the combined contextPrompt ──
    # The mobile app injects English context blocks like:
    #   [Platform doctor recommendations context ...]
    #   [System Context: ...]
    #   [Conversation safety rule: ...]
    # We split these out so language detection works on the ORIGINAL question only.
    raw_question = data.question
    original_question = raw_question
    internal_context_parts = []

    # Split on context markers injected by the app
    context_split = re.split(r'\n\n\[', raw_question, maxsplit=1)
    if len(context_split) > 1:
        original_question = context_split[0].strip()
        internal_context_parts.append("[" + context_split[1])

    # Detect language from the ORIGINAL question only (not the injected English context)
    arabic_chars = sum(1 for c in original_question if '\u0600' <= c <= '\u06FF')
    total_chars = max(len(original_question.strip()), 1)
    is_arabic = (arabic_chars / total_chars) > 0.2  # Lowered threshold from 0.3 to 0.2 for better detection
    detected_lang = "ar" if is_arabic else "en"

    # Build internal context string
    context_block = "\n".join(internal_context_parts).strip()

    try:
        chat = model.start_chat(history=chat_history)

        # ── Language enforcement ──
        if is_arabic:
            lang_instruction = (
                "🔴 تعليمات اللغة — إلزامية:\n"
                "يجب أن يكون ردك بالكامل باللغة العربية.\n"
                "المريض يتحدث العربية. أجب بالعربية فقط.\n"
                "لا ترد بالإنجليزية أبداً إلا للمصطلحات الطبية أو أسماء الأدوية.\n"
                "السياق الداخلي المرفق أدناه مكتوب بالإنجليزية لأغراض تقنية — تجاهل لغته وأجب بالعربية."
            )
        else:
            lang_instruction = (
                "🔴 LANGUAGE INSTRUCTION — MANDATORY:\n"
                "You MUST respond entirely in English.\n"
                "The patient is speaking English. Respond in English only."
            )

        # ── Build the structured prompt ──
        prompt = f"""
{lang_instruction}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🩺 ROLE: You are an advanced, professional medical AI assistant.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 PATIENT'S QUESTION:
{original_question}

"""
        # Add internal context if present (doctor recommendations, patient medical profile, etc.)
        if context_block:
            prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 INTERNAL CONTEXT (use this data to personalize your response):
{context_block}

IMPORTANT: The above context contains the patient's medical history, chronic diseases, medications, allergies, and/or doctor recommendations from the platform database.
You MUST use this information to give personalized, relevant medical advice.
If the patient asks about symptoms, CHECK their chronic diseases and medications for possible connections.
If doctor recommendations are provided AND the patient has a medical issue, suggest those specific doctors.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        prompt += f"""
CRITICAL RULES:
1. STRICTLY MEDICAL: You are ONLY allowed to answer medical, health, and wellness questions. If the user asks about non-medical topics (programming, tech, math, etc.), politely apologize and say you are a specialized medical assistant.
2. {'أجب باللغة العربية فقط. لا تستخدم الإنجليزية إلا للمصطلحات الطبية وأسماء الأدوية.' if is_arabic else 'Respond in English only.'}
3. Keep it professional, empathetic, and highly informative.
4. DO NOT use markdown symbols like # or *. Return beautiful plain text with elegant spacing and emojis where appropriate.
5. ACTIVE INQUIRY: If the user complains of pain, tiredness, or any symptoms, DO NOT give a final diagnosis. ALWAYS end your response by asking 1-2 relevant follow-up questions to gather more details.
6. Once you have a clear picture, provide safe advice and ALWAYS recommend seeing a doctor.
7. NEVER say or imply that you are a doctor. Say you are an AI medical assistant.
8. If the user is perfectly healthy with no problems, do NOT recommend doctors, just encourage a healthy lifestyle.
9. If platform doctor recommendations are in the context AND the user has a medical issue, use ONLY those doctors, prioritize by specialty match and rating. Do not invent doctor names.
10. If the patient's medical profile is in the context, USE IT to personalize your response. Mention relevant chronic diseases, medications, or allergies when applicable.
"""
        response = chat.send_message(prompt)
        return {
            "query": data.question,
            "reply": strip_markdown(response.text),
            "model_used": GEMINI_MODEL,
            "is_medical": True,
            "found_in_database": False,
            "low_confidence": False,
            "language": detected_lang,
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
    Analyze this patient medical history and return two plain-text reports as valid JSON only.

    Patient data:
    Vitals: {json.dumps(data.vitals)}
    Surgeries: {json.dumps(data.surgeries)}
    Medications: {json.dumps(data.medications)}
    Allergies: {json.dumps(data.allergies)}
    Chronic Diseases: {json.dumps(data.chronic_diseases)}
    AI Analyzed Documents: {json.dumps(data.documents_analysis)}
    Recent Visits (Last 8 Months): {json.dumps(data.recent_visits)}
    Recommended Specialty: {data.recommended_specialty or "best available match"}
    Platform Doctors Sorted By Rating/Reviews: {json.dumps(data.recommended_doctors)}

    Instructions:
    - CAREFULLY SUMMARIZE the patient's existing history, vitals, surgeries, medications, and recent visits from the last 8 months.
    - Include a dedicated section that summarizes the patient's visit history over the last 8 months. Mention dates, complaints, doctors, and outcomes when available.
    - Do NOT invent new diagnoses, do NOT hallucinate medical conditions, and do NOT act like a diagnosing doctor.
    - Identify ONLY clear health risks (like high blood pressure) or allergy conflicts based strictly on the provided data.
    - Mention relevant uploaded document findings when present.
    - Give safe, simple wellness tips (hydration, sleep) but avoid complex medical prescriptions.
    - CRITICAL: If the patient is completely healthy, just say their health profile looks good and encourage a healthy lifestyle. Do NOT recommend any specialty or doctors.
    - If platform doctors are provided AND the patient has a clear medical issue, you MUST recommend the highest-rated relevant doctors from the provided list. Briefly explain exactly why that doctor's specialty matches the patient's existing recorded condition. Do NOT hallucinate any doctor names.
    - Do NOT use emojis. Do NOT use markdown symbols. Format strictly as clear plain text.

    Return this JSON shape only:
    {{
      "en": "English report with sections separated by double newlines (\\n\\n). Sections: General Summary, Recent Visits Summary (Last 8 Months), Warnings, Medications & Interactions, Safe Advice, Recommended Doctors (if any). DO NOT USE EMOJIS.",
      "ar": "تقرير عربي بأقسام مفصولة بأسطر جديدة (\\n\\n): الملخص العام، ملخص الزيارات (آخر 8 أشهر)، التحذيرات، الأدوية والتداخلات، نصائح آمنة، الأطباء المقترحون من الموقع (إن وجد). لا تستخدم أي إيموجي على الإطلاق.",
      "needsDoctor": true // set to false ONLY if the patient is completely healthy and has no medical problems.
    }}

    Language rules:
    - The en field must be English only.
    - The ar field must be Arabic only except doctor names and necessary medical terms.
    """
    
    try:
        response = await generate_with_retry(prompt)
        # Robust JSON extraction
        content = response.text.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(), strict=False)
            analysis_en = strip_markdown(result.get("en", "No English report generated."))
            analysis_ar = strip_markdown(result.get("ar", "لم يتم إنشاء تقرير بالعربية."))
            if not analysis_ar.strip() or analysis_ar.strip() == analysis_en.strip():
                analysis_ar = "لم يتم إنشاء التقرير بالعربية بشكل كامل. يرجى إعادة التحليل."
            return {
                "analysis_en": analysis_en,
                "analysis_ar": analysis_ar,
                "needsDoctor": result.get("needsDoctor", True)
            }
        
        # Fallback split logic if AI fails JSON but returns both
        text = strip_markdown(response.text)
        return {
            "analysis_en": text,
            "analysis_ar": "تعذّر إنشاء التقرير العربي تلقائياً. يرجى إعادة المحاولة."
        }
        
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
        response = await generate_with_retry([prompt, img])
        
        # Get the text from the response
        text = response.text
        
        # We also ask for technical details (extracted JSON)
        tech_prompt = f"Based on this extracted text, provide the raw medical data (test names, values, or medication list) as a plain text summary or JSON: {text}"
        tech_response = await generate_with_retry(tech_prompt)
        tech_details = tech_response.text.strip()
        
        return {
            "status": "success", 
            "analysis_ar": strip_markdown(text), 
            "technical_details": strip_markdown(tech_details),
            "model_used": GEMINI_MODEL,
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
        response = await generate_with_retry(prompt)
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

@app.post("/summarize-visit", dependencies=[Depends(verify_internal_token)])
async def summarize_visit(data: dict):
    """Generates a bilingual summary of a medical visit."""
    complaint = data.get("complaint", "None")
    diagnosis = data.get("diagnosis", "None")
    treatment = data.get("treatment", "None")
    
    prompt = f"""
    You are an expert AI medical assistant. Summarize this medical visit concisely for the patient.
    Complaint: {complaint}
    Diagnosis: {diagnosis}
    Treatment Plan: {treatment}
    
    RETURN ONLY A VALID JSON OBJECT exactly like this:
    {{
      "summary_en": "A concise, patient-friendly summary in English",
      "summary_ar": "ملخص موجز ومبسط للمريض باللغة العربية"
    }}
    """
    
    try:
        response = await generate_with_retry(prompt)
        content = response.text.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            en_text = strip_markdown(result.get("summary_en", "Summary not available."))
            ar_text = strip_markdown(result.get("summary_ar", "الملخص غير متاح."))
            return {"summary_en": en_text, "summary_ar": ar_text}
        return {"summary_en": "Summary not available.", "summary_ar": "الملخص غير متاح."}
    except Exception as e:
        print(f"DEBUG: Error in summarize_visit: {e}")
        return {"summary_en": "Error generating summary.", "summary_ar": "حدث خطأ أثناء إنشاء الملخص."}

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
        response = await generate_with_retry(prompt)
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
        response = await generate_with_retry(prompt)
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
    
    CRITICAL QUALITY RULE: If the Chief Complaint or History of Present Illness contains gibberish, letters, typos, or nonsensical terms with no real medical meaning (e.g. 'ni', 'بالل', 'بتاال', 'اتلنت', 'انم', 'asd', 'xyz'), do NOT generate speculative differential diagnoses (like Gastroenteritis, GERD, etc.) or standard treatment plans. Instead, return a clean, polite response asking the user to enter clear medical symptoms:
    - "assessment_en" should be "The entered complaint is unclear or contains typographical errors. Please provide clear clinical symptoms to proceed."
    - "assessment_ar" should be "الشكوى المدخلة غير واضحة أو تحتوي على أخطاء إملائية. يرجى إدخال أعراض سريرية واضحة للمتابعة."
    - "plan_en" should be "N/A"
    - "plan_ar" should be "غير متاح"
    
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
        response = await generate_with_retry(prompt)
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

    # Detect language from the input text
    arabic_chars = sum(1 for c in data.text if '\u0600' <= c <= '\u06FF')
    total_chars = max(len(data.text.strip()), 1)
    is_arabic = (arabic_chars / total_chars) > 0.2  # Lowered threshold from 0.3 to 0.2 for better detection
    detected_lang = "ar" if is_arabic else "en"

    # Language instruction
    if is_arabic:
        lang_instruction = (
            "🔴 تعليمات اللغة — إلزامية:\n"
            "المريض يتحدث العربية. يجب أن تكون الحقول العربية (summary_ar, follow_up_ar) هي الأساسية والأكثر تفصيلاً.\n"
            "الحقول الإنجليزية (summary_en, follow_up_en) يجب أن تكون موجودة ولكن يمكن أن تكون مختصرة.\n"
            "استخدم العربية في جميع التفسيرات والأسئلة المتابعة.\n"
        )
    else:
        lang_instruction = (
            "🔴 LANGUAGE INSTRUCTION — MANDATORY:\n"
            "The patient is speaking English. The English fields (summary_en, follow_up_en) must be primary and most detailed.\n"
            "The Arabic fields (summary_ar, follow_up_ar) should exist but can be brief.\n"
            "Use English for all explanations and follow-up questions.\n"
        )

    prompt = f"""
    {lang_instruction}

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
          "medicationNameAr": "اسم الدواء بالعربية",
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
        response = await generate_with_retry(prompt)
        content = response.text.strip()

        # Robust JSON extraction
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(), strict=False)
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
    
    CRITICAL QUALITY RULE: If the Chief Complaint contains gibberish, typos, or nonsense words with no clear medical meaning (e.g., 'ni', 'بالل', 'بتاال', 'اتلنت', 'انم'), do NOT hypothesize about it or suggest irrelevant specialties/warnings. Instead, return:
    - "summary_en": "Reason for visit is unclear or contains typos. Please verify symptoms."
    - "summary_ar": "سبب الزيارة غير واضح أو يحتوي على أخطاء إملائية. يرجى التحقق من الأعراض."
    
    RETURN ONLY A VALID JSON OBJECT with these keys:
    {{
      "summary_en": "A highly concise, bulleted summary in English. Include: 1) Main reason for visit, 2) Relevant history/vitals, 3) AI Alerts (e.g. drug interactions, abnormal vitals). Use clean plain text only. Do NOT use any emojis or markdown symbols under any circumstances.",
      "summary_ar": "ملخص مركز جداً بالعربية في نقاط. يحتوي على: 1) سبب الزيارة الأساسي، 2) التاريخ المرضي/العلامات الحيوية ذات الصلة، 3) تنبيهات الذكاء الاصطناعي (مثل تعارض أدوية أو قراءات غير طبيعية). استخدم نصاً عادياً ونظيفاً فقط. لا تستخدم أي إيموجي (رموز تعبيرية) أو مارك داون تحت أي ظرف من الظروف."
    }}
    """
    
    try:
        response = await generate_with_retry(prompt)
        content = response.text.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(), strict=False)
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
        response = await generate_with_retry(prompt)
        content = response.text.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(), strict=False)
            return {
                "tip_en": strip_markdown(result.get("tip_en", "Stay hydrated and take short walks throughout your day.")),
                "tip_ar": strip_markdown(result.get("tip_ar", "حافظ على شرب الماء وامشِ قليلاً خلال يومك."))
            }
        return {"tip_en": "Stay healthy and active.", "tip_ar": "حافظ على صحتك ونشاطك."}
    except Exception as e:
        print(f"DEBUG: Error in personalized_tip: {e}")
        return {"tip_en": "Stay hydrated.", "tip_ar": "حافظ على شرب الماء."}



@app.post("/summarize-booking-reason", dependencies=[Depends(verify_internal_token)])
async def summarize_booking_reason(data: AskRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")
        
    chat_logs = [{"role": msg.role, "content": msg.content} for msg in (data.history or [])]
    
    prompt = f"""
    You are an expert medical AI assistant.
    Review the following chat history between a patient and the AI assistant during an appointment booking process:
    {json.dumps(chat_logs)}
    
    The patient was asked: "{data.question}"
    
    Extract and summarize the primary 'Reason for Visit' (Chief Complaint) and any relevant symptoms mentioned.
    Keep it extremely concise (1-3 sentences) and professional. Do NOT use markdown or emojis.
    If they didn't provide enough info, just say "General Consultation".
    
    CRITICAL: If the patient's answers contain only gibberish, typos, or nonsense words with no clear medical meaning (e.g., 'ni', 'بالل', 'بتاال', 'اتلنت', 'انم'), you MUST summarize it simply as "General Consultation". Do NOT try to interpret or explain the gibberish.
    """
    
    try:
        response = await generate_with_retry(prompt)
        summary = strip_markdown(response.text)
        return {"summary": summary}
    except Exception as e:
        print(f"DEBUG: Error in summarize_booking_reason: {e}")
        return {"summary": "General Consultation"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)