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
    return await apiFetch<ParsedMedicalProfile>(
      API.chat.parseMedicalProfile,
      {
        method: "POST",
        body: JSON.stringify({ text, saveToProfile }),
      },
      true
    );
  } catch {
    return parseMedicalProfileLocally(text);
  }
};

const DISEASE_KEYWORDS = [
  ["diabetes", "Diabetes"],
  ["sugar", "Diabetes"],
  ["سكر", "Diabetes"],
  ["hypertension", "Hypertension"],
  ["blood pressure", "Hypertension"],
  ["pressure", "Hypertension"],
  ["ضغط", "Hypertension"],
  ["asthma", "Asthma"],
  ["ربو", "Asthma"],
  ["heart", "Heart disease"],
  ["قلب", "Heart disease"],
];

const ALLERGY_KEYWORDS = [
  ["penicillin", "Penicillin"],
  ["بنسلين", "Penicillin"],
  ["aspirin", "Aspirin"],
  ["اسبرين", "Aspirin"],
  ["allergy", "Allergy"],
  ["حساسية", "Allergy"],
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
    .map(([, name]) => ({
      diseaseName: name,
      diseaseNameAr: name,
      diseaseType: "Chronic",
      severity: "Moderate",
      diagnosedDate: null,
      notes: source,
    })), "diseaseName");

  const allergies = uniqueByName(ALLERGY_KEYWORDS
    .filter(([keyword]) => lower.includes(keyword.toLowerCase()))
    .map(([, name]) => ({
      allergenName: name,
      allergenNameAr: name,
      allergyType: "Medication",
      severity: "Moderate",
      reactionDescription: source,
    })), "allergenName");

  const medicationMatch = source.match(/(?:take|taking|باخد|اخد|دواء|علاج)\s+([^،,.]+)/i);
  const medications = medicationMatch?.[1]?.trim()
    ? [{
        medicationName: medicationMatch[1].trim(),
        dosage: "",
        form: "Tablet",
        frequency: "Once daily",
        instructions: source,
        doseTimes: null,
        isChronic: true,
      }]
    : [];

  const total = chronic_diseases.length + allergies.length + medications.length;
  const summary = total > 0
    ? `I found ${total} medical item(s). You can add more details or save.`
    : "I could not extract a clear medical item yet. Please mention a condition, medication, or allergy.";

  return {
    chronic_diseases,
    medications,
    allergies,
    summary_ar: summary,
    summary_en: summary,
    follow_up_ar: "Add any missing medications, allergies, or chronic conditions, then press Save.",
    follow_up_en: "Add any missing medications, allergies, or chronic conditions, then press Save.",
  };
}
