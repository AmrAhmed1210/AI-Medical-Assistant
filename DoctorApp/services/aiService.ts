import { API } from "../constants/api";
import { apiFetch } from "./http";

export const analyzePatientHistory = async (history: unknown) => {
  return apiFetch<Record<string, string>>(
    API.chat.analyzeHistory,
    {
      method: "POST",
      body: JSON.stringify(history),
    },
    true
  );
};

export const summarizeSurgery = async (description: string) => {
  return apiFetch<{ summary_en: string; summary_ar: string }>(
    `${API.chat.ask.replace("/ask", "/summarize-surgery")}`,
    {
      method: "POST",
      body: JSON.stringify({ description }),
    },
    true
  );
};

export const summarizeMedicalItem = async (type: string, description: string) => {
  try {
    return await apiFetch<{ summary_en: string; summary_ar: string }>(
      `${API.chat.ask.replace("/ask", "/summarize-medical-item")}`,
      {
        method: "POST",
        body: JSON.stringify({ type, description }),
      },
      true
    );
  } catch {
    return { summary_en: description, summary_ar: description };
  }
};

export const analyzeVitalsAdvice = async (vitals: unknown[], patientInfo: unknown) => {
  try {
    return await apiFetch<{ advice_en: string; advice_ar: string }>(
      `${API.chat.ask.replace("/ask", "/analyze-vitals")}`,
      {
        method: "POST",
        body: JSON.stringify({ vitals, patient_info: patientInfo }),
      },
      true
    );
  } catch {
    return { advice_en: "Keep monitoring.", advice_ar: "استمر في المتابعة." };
  }
};

export const checkMedicationSafety = async (medication: string, history: unknown) => {
  try {
    return await apiFetch<{ safety_en: string; safety_ar: string }>(
      `${API.chat.ask.replace("/ask", "/check-medication-safety")}`,
      {
        method: "POST",
        body: JSON.stringify({ medication, history }),
      },
      true
    );
  } catch {
    return { safety_en: "Consult doctor.", safety_ar: "استشر طبيبك." };
  }
};

export const analyzeMedicalImage = async (
  uri: string,
  _type: "prescription" | "lab" = "prescription",
  patientContext: string = ""
) => {
  const formData = new FormData();
  const filename = uri.split("/").pop() || "image.jpg";
  const match = /\.(\w+)$/.exec(filename);
  const mimeType = match ? `image/${match[1]}` : "image/jpeg";

  formData.append("file", { uri, name: filename, type: mimeType } as any);
  if (patientContext) {
    formData.append("patient_context", patientContext);
  }

  return apiFetch<unknown>(
    API.chat.analyzeImage,
    {
      method: "POST",
      body: formData,
      headers: { "Content-Type": "multipart/form-data" },
    },
    true
  );
};

export interface ParsedMedicalProfile {
  chronic_diseases: Array<{
    diseaseName: string;
    diseaseNameAr?: string;
    diseaseType: string;
    severity: string;
    diagnosedDate?: string | null;
    notes?: string;
  }>;
  medications: Array<{
    medicationName: string;
    genericName?: string;
    dosage: string;
    form: string;
    frequency: string;
    instructions?: string;
    doseTimes?: string | null;
    isChronic: boolean;
  }>;
  allergies: Array<{
    allergenName: string;
    allergenNameAr?: string;
    allergyType: string;
    severity: string;
    reactionDescription?: string;
  }>;
  summary_ar: string;
  summary_en: string;
  follow_up_ar: string;
  follow_up_en: string;
}

export const parseMedicalProfile = async (
  text: string,
  saveToProfile: boolean = false
): Promise<ParsedMedicalProfile> => {
  try {
    console.log("[parseMedicalProfile] Calling API...");
    const result = await apiFetch<ParsedMedicalProfile>(
      API.chat.parseMedicalProfile,
      {
        method: "POST",
        body: JSON.stringify({ text, saveToProfile }),
      },
      true
    );
    console.log("[parseMedicalProfile] API success, items:", 
      (result.chronic_diseases?.length ?? 0) + (result.medications?.length ?? 0) + (result.allergies?.length ?? 0));
    return result;
  } catch (err: any) {
    console.warn("[parseMedicalProfile] API failed, using local fallback:", err?.message);
    return parseMedicalProfileLocally(text);
  }
};

const DISEASE_KEYWORDS: [string, string, string][] = [
  ["diabetes", "Diabetes", "سكر"],
  ["sugar", "Diabetes", "سكر"],
  ["سكر", "Diabetes", "سكر"],
  ["سكري", "Diabetes", "سكري"],
  ["hypertension", "Hypertension", "ضغط عالي"],
  ["blood pressure", "Hypertension", "ضغط عالي"],
  ["pressure", "Hypertension", "ضغط عالي"],
  ["ضغط", "Hypertension", "ضغط عالي"],
  ["asthma", "Asthma", "ربو"],
  ["ربو", "Asthma", "ربو"],
  ["حساسية صدر", "Asthma", "حساسية صدرية"],
  ["heart", "Heart Disease", "قلب"],
  ["قلب", "Heart Disease", "مرض قلبي"],
  ["cholesterol", "High Cholesterol", "كوليسترول"],
  ["كوليسترول", "High Cholesterol", "كوليسترول عالي"],
  ["دهون", "High Cholesterol", "دهون عالية"],
  ["thyroid", "Thyroid Disorder", "غدة درقية"],
  ["غدة", "Thyroid Disorder", "غدة درقية"],
  ["درقية", "Thyroid Disorder", "غدة درقية"],
  ["كلى", "Kidney Disease", "مرض كلوي"],
  ["kidney", "Kidney Disease", "مرض كلوي"],
  ["كبد", "Liver Disease", "مرض كبدي"],
  ["liver", "Liver Disease", "مرض كبدي"],
  ["انيميا", "Anemia", "أنيميا"],
  ["anemia", "Anemia", "أنيميا"],
  ["روماتيزم", "Rheumatism", "روماتيزم"],
  ["arthritis", "Arthritis", "التهاب مفاصل"],
  ["مفاصل", "Arthritis", "التهاب مفاصل"],
  ["depression", "Depression", "اكتئاب"],
  ["اكتئاب", "Depression", "اكتئاب"],
  ["epilepsy", "Epilepsy", "صرع"],
  ["صرع", "Epilepsy", "صرع"],
];

const ALLERGY_KEYWORDS: [string, string, string][] = [
  ["penicillin", "Penicillin", "بنسلين"],
  ["بنسلين", "Penicillin", "بنسلين"],
  ["aspirin", "Aspirin", "أسبرين"],
  ["اسبرين", "Aspirin", "أسبرين"],
  ["sulfa", "Sulfa drugs", "سلفا"],
  ["سلفا", "Sulfa drugs", "سلفا"],
  ["ibuprofen", "Ibuprofen", "ايبوبروفين"],
  ["بروفين", "Ibuprofen", "بروفين"],
  ["حساسية", "General Allergy", "حساسية"],
  ["allergy", "General Allergy", "حساسية"],
  ["بيض", "Eggs", "بيض"],
  ["لبن", "Milk/Dairy", "لبن"],
  ["فول سوداني", "Peanuts", "فول سوداني"],
  ["peanut", "Peanuts", "فول سوداني"],
  ["gluten", "Gluten", "جلوتين"],
  ["جلوتين", "Gluten", "جلوتين"],
];

const MED_KEYWORDS: [string, string][] = [
  ["جلوكوفاج", "Glucophage (Metformin)"],
  ["glucophage", "Glucophage (Metformin)"],
  ["metformin", "Metformin"],
  ["ميتفورمين", "Metformin"],
  ["كونكور", "Concor (Bisoprolol)"],
  ["concor", "Concor (Bisoprolol)"],
  ["اميلور", "Amilor"],
  ["لازكس", "Lasix (Furosemide)"],
  ["lasix", "Lasix (Furosemide)"],
  ["انسولين", "Insulin"],
  ["insulin", "Insulin"],
  ["اسبرين", "Aspirin"],
  ["aspirin", "Aspirin"],
  ["بنادول", "Panadol (Paracetamol)"],
  ["panadol", "Panadol (Paracetamol)"],
  ["ليبيتور", "Lipitor (Atorvastatin)"],
  ["lipitor", "Lipitor (Atorvastatin)"],
  ["اوميبرازول", "Omeprazole"],
  ["omeprazole", "Omeprazole"],
  ["ثيروكسين", "Thyroxine"],
  ["eltroxin", "Eltroxin (Thyroxine)"],
  ["التروكسين", "Eltroxin (Thyroxine)"],
];

const AR_DISEASE_KEYWORDS: [string, string, string][] = [
  ["سكر", "Diabetes", "سكر"],
  ["سكري", "Diabetes", "سكري"],
  ["ضغط", "Hypertension", "ضغط عالي"],
  ["ربو", "Asthma", "ربو"],
  ["حساسية صدر", "Asthma", "حساسية صدرية"],
  ["قلب", "Heart Disease", "مرض قلبي"],
  ["كوليسترول", "High Cholesterol", "كوليسترول عالي"],
  ["دهون", "High Cholesterol", "دهون عالية"],
  ["غدة", "Thyroid Disorder", "غدة درقية"],
  ["درقية", "Thyroid Disorder", "غدة درقية"],
  ["كلى", "Kidney Disease", "مرض كلوي"],
  ["كبد", "Liver Disease", "مرض كبدي"],
  ["انيميا", "Anemia", "أنيميا"],
  ["أنيميا", "Anemia", "أنيميا"],
  ["روماتيزم", "Rheumatism", "روماتيزم"],
  ["مفاصل", "Arthritis", "التهاب مفاصل"],
  ["اكتئاب", "Depression", "اكتئاب"],
  ["صرع", "Epilepsy", "صرع"],
];

const AR_ALLERGY_KEYWORDS: [string, string, string][] = [
  ["بنسلين", "Penicillin", "بنسلين"],
  ["اسبرين", "Aspirin", "أسبرين"],
  ["أسبرين", "Aspirin", "أسبرين"],
  ["سلفا", "Sulfa drugs", "سلفا"],
  ["بروفين", "Ibuprofen", "بروفين"],
  ["ايبوبروفين", "Ibuprofen", "ايبوبروفين"],
  ["حساسية", "General Allergy", "حساسية"],
  ["بيض", "Eggs", "بيض"],
  ["لبن", "Milk/Dairy", "لبن"],
  ["فول سوداني", "Peanuts", "فول سوداني"],
  ["جلوتين", "Gluten", "جلوتين"],
];

const AR_MED_KEYWORDS: [string, string][] = [
  ["جلوكوفاج", "Glucophage (Metformin)"],
  ["ميتفورمين", "Metformin"],
  ["كونكور", "Concor (Bisoprolol)"],
  ["اميلور", "Amilor"],
  ["لازكس", "Lasix (Furosemide)"],
  ["انسولين", "Insulin"],
  ["اسبرين", "Aspirin"],
  ["أسبرين", "Aspirin"],
  ["بنادول", "Panadol (Paracetamol)"],
  ["ليبيتور", "Lipitor (Atorvastatin)"],
  ["اوميبرازول", "Omeprazole"],
  ["ثيروكسين", "Thyroxine"],
  ["التروكسين", "Eltroxin (Thyroxine)"],
];

const uniqueByName = <T extends Record<string, unknown>>(items: T[], key: keyof T) => {
  const seen = new Set<string>();
  return items.filter((item) => {
    const value = String(item[key] || "").trim().toLowerCase();
    if (!value || seen.has(value)) return false;
    seen.add(value);
    return true;
  });
};

function parseMedicalProfileLocally(text: string): ParsedMedicalProfile {
  const source = text.trim();
  const lower = source.toLowerCase();

  const chronic_diseases = uniqueByName(DISEASE_KEYWORDS
    .filter(([keyword]) => lower.includes(keyword.toLowerCase()))
    .map(([, name, nameAr]) => ({
      diseaseName: name,
      diseaseNameAr: nameAr,
      diseaseType: "Chronic",
      severity: "Moderate",
      diagnosedDate: null,
      notes: source,
    })), "diseaseName");

  const allergies = uniqueByName(ALLERGY_KEYWORDS
    .filter(([keyword]) => lower.includes(keyword.toLowerCase()))
    .map(([, name, nameAr]) => ({
      allergenName: name,
      allergenNameAr: nameAr,
      allergyType: "Medication",
      severity: "Moderate",
      reactionDescription: source,
    })), "allergenName");

  // Match medications from known list
  const medications = uniqueByName(MED_KEYWORDS
    .filter(([keyword]) => lower.includes(keyword.toLowerCase()))
    .map(([, name]) => ({
      medicationName: name,
      dosage: "",
      form: "Tablet",
      frequency: "Once daily",
      instructions: source,
      doseTimes: null,
      isChronic: true,
    })), "medicationName");

  // Fallback: try to extract unknown medication from common patterns
  if (medications.length === 0) {
    const medPatterns = [
      /(?:باخد|بأخد|اخد|آخد|بتاخد|take|taking)\s+([^\u060C،,.]+)/i,
      /(?:دواء|علاج|حبوب|medication)\s+([^\u060C،,.]+)/i,
    ];
    for (const pattern of medPatterns) {
      const match = source.match(pattern);
      if (match?.[1]?.trim()) {
        medications.push({
          medicationName: match[1].trim(),
          dosage: "",
          form: "Tablet",
          frequency: "Once daily",
          instructions: source,
          doseTimes: null,
          isChronic: true,
        });
        break;
      }
    }
  }

  const total = chronic_diseases.length + allergies.length + medications.length;
  const hasArabic = /[\u0600-\u06FF]/.test(source);

  const summary = total > 0
    ? (hasArabic
      ? `✅ تم استخراج ${total} عنصر طبي من كلامك. راجع البيانات واضغط "حفظ الكل" لتسجيلها.`
      : `✅ Extracted ${total} medical item(s). Review the data and tap "Save All" to record them.`)
    : (hasArabic
      ? "لم أتمكن من استخراج بيانات طبية واضحة. جرّب تقول حاجة زي \"عندي سكر\" أو \"باخد جلوكوفاج\"."
      : "I could not extract a clear medical item yet. Try saying \"I have diabetes\" or \"I take Glucophage\".");

  const follow_up = total > 0
    ? (hasArabic
      ? "هل في أدوية تانية أو حساسية عايز تضيفها؟ لو خلصت اضغط حفظ الكل."
      : "Any other medications or allergies to add? If done, tap Save All.")
    : (hasArabic
      ? "اكتب أمراضك أو أدويتك بأي شكل وأنا هفهمك."
      : "Describe your conditions or medications in any way.");

  return {
    chronic_diseases,
    medications,
    allergies,
    summary_ar: hasArabic ? summary : "",
    summary_en: hasArabic ? "" : summary,
    follow_up_ar: hasArabic ? follow_up : "",
    follow_up_en: hasArabic ? "" : follow_up,
  };
}
