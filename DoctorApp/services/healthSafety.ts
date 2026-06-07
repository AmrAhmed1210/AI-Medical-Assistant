export type VitalSeverity = "normal" | "low" | "high" | "invalid";

export interface VitalAssessment {
  isNormal: boolean;
  severity: VitalSeverity;
  title: string;
  message: string;
  rangeText: string;
}

export const VITAL_RANGES: Record<string, { min: number; max: number; unit: string; label: string }> = {
  "Blood Sugar": { min: 70, max: 140, unit: "mg/dL", label: "Blood sugar" },
  "Heart Rate": { min: 60, max: 100, unit: "bpm", label: "Heart rate" },
  "Temperature": { min: 36.1, max: 37.5, unit: "C", label: "Temperature" },
  "SpO2": { min: 95, max: 100, unit: "%", label: "Oxygen saturation" },
  "Respiratory Rate": { min: 12, max: 20, unit: "breaths/min", label: "Respiratory rate" },
};

export const BP_RANGE = {
  systolic: { min: 90, max: 140 },
  diastolic: { min: 60, max: 90 },
  unit: "mmHg",
};

export function getVitalUnit(type: string): string {
  if (type === "Blood Pressure") return BP_RANGE.unit;
  return VITAL_RANGES[type]?.unit ?? "standard";
}

export function getVitalRangeText(type: string): string {
  if (type === "Blood Pressure") {
    return `${BP_RANGE.systolic.min}-${BP_RANGE.systolic.max}/${BP_RANGE.diastolic.min}-${BP_RANGE.diastolic.max} ${BP_RANGE.unit}`;
  }
  const range = VITAL_RANGES[type];
  return range ? `${range.min}-${range.max} ${range.unit}` : "";
}

export function assessVitalReading(type: string, value: number, value2?: number): VitalAssessment {
  const rangeText = getVitalRangeText(type);
  if (!Number.isFinite(value) || value <= 0 || (type === "Blood Pressure" && (value2 == null || !Number.isFinite(value2) || value2 <= 0))) {
    return {
      isNormal: false,
      severity: "invalid",
      title: "Invalid reading",
      message: "This reading is incomplete or not medically valid. Please re-enter it correctly.",
      rangeText,
    };
  }

  if (type === "Blood Pressure") {
    const low = value < BP_RANGE.systolic.min || (value2 ?? 0) < BP_RANGE.diastolic.min;
    const high = value > BP_RANGE.systolic.max || (value2 ?? 0) > BP_RANGE.diastolic.max;
    if (low || high) {
      return {
        isNormal: false,
        severity: low ? "low" : "high",
        title: low ? "Low blood pressure warning" : "High blood pressure warning",
        message: `Blood pressure ${value}/${value2} ${BP_RANGE.unit} is ${low ? "below" : "above"} the normal range (${rangeText}). If you feel dizzy, chest pain, shortness of breath, fainting, or severe headache, seek medical help urgently.`,
        rangeText,
      };
    }
  } else {
    const range = VITAL_RANGES[type];
    if (!range) {
      return { isNormal: true, severity: "normal", title: "Recorded", message: "Reading recorded.", rangeText };
    }
    if (value < range.min || value > range.max) {
      const low = value < range.min;
      return {
        isNormal: false,
        severity: low ? "low" : "high",
        title: `${low ? "Low" : "High"} ${range.label} warning`,
        message: `${range.label} ${value} ${range.unit} is ${low ? "below" : "above"} the normal range (${rangeText}). Please re-check the measurement and contact a doctor if symptoms are present or the result repeats.`,
        rangeText,
      };
    }
  }

  return {
    isNormal: true,
    severity: "normal",
    title: "Normal reading",
    message: `This reading is within the expected range (${rangeText}).`,
    rangeText,
  };
}

const normalize = (value: string) =>
  value.toLowerCase().replace(/[^a-z0-9\u0600-\u06ff]+/g, " ").trim();

const containsAny = (source: string, terms: string[]) => {
  const normalized = normalize(source);
  return terms.some((term) => normalized.includes(normalize(term)));
};

const MED_ALIASES: Record<string, string[]> = {
  aspirin: ["aspirin", "اسبرين", "اسبرين", "aspocid"],
  ibuprofen: ["ibuprofen", "brufen", "بروفين", "ايبوبروفين"],
  diclofenac: ["diclofenac", "voltaren", "فولتارين"],
  warfarin: ["warfarin", "وارفارين", "coumadin"],
  metformin: ["metformin", "glucophage", "جلوكوفاج", "ميتفورمين"],
  insulin: ["insulin", "انسولين"],
  bisoprolol: ["bisoprolol", "concor", "كونكور"],
  furosemide: ["furosemide", "lasix", "لازكس"],
  penicillin: ["penicillin", "amoxicillin", "augmentin", "بنسلين", "اوجمنتين"],
};

const findMedKey = (name: string) =>
  Object.entries(MED_ALIASES).find(([, aliases]) => containsAny(name, aliases))?.[0] ?? normalize(name);

export interface MedicationSafetyAssessment {
  hasWarning: boolean;
  severity: "safe" | "caution" | "danger";
  title: string;
  message: string;
  reasons: string[];
}

export function assessMedicationSafety(
  medicationName: string,
  context: { allergies?: string[]; medications?: string[]; chronicDiseases?: string[] }
): MedicationSafetyAssessment {
  const reasons: string[] = [];
  const newKey = findMedKey(medicationName);
  const allergyHit = (context.allergies ?? []).find((allergy) =>
    containsAny(medicationName, [allergy]) || containsAny(allergy, MED_ALIASES[newKey] ?? [medicationName])
  );
  if (allergyHit) reasons.push(`Possible allergy conflict with: ${allergyHit}.`);

  const currentKeys = (context.medications ?? []).map(findMedKey);
  const has = (key: string) => currentKeys.includes(key);
  if (newKey === "aspirin" && (has("ibuprofen") || has("diclofenac") || has("warfarin"))) {
    reasons.push("Aspirin may increase bleeding or stomach-risk with NSAIDs/warfarin.");
  }
  if ((newKey === "ibuprofen" || newKey === "diclofenac") && (has("aspirin") || has("warfarin"))) {
    reasons.push("NSAIDs may conflict with aspirin/warfarin and increase bleeding or stomach-risk.");
  }
  if (newKey === "warfarin" && (has("aspirin") || has("ibuprofen") || has("diclofenac"))) {
    reasons.push("Warfarin with aspirin/NSAIDs may raise bleeding risk.");
  }

  const chronicText = normalize((context.chronicDiseases ?? []).join(" "));
  if ((newKey === "ibuprofen" || newKey === "diclofenac") && containsAny(chronicText, ["kidney", "كلى", "ضغط", "hypertension"])) {
    reasons.push("NSAIDs can be risky with kidney disease or high blood pressure.");
  }
  if (newKey === "bisoprolol" && containsAny(chronicText, ["asthma", "ربو", "حساسية صدر"])) {
    reasons.push("Beta blockers may worsen asthma or breathing symptoms in some patients.");
  }
  if (newKey === "metformin" && containsAny(chronicText, ["kidney", "كلى"])) {
    reasons.push("Metformin needs medical review in kidney disease.");
  }

  if (reasons.length === 0) {
    return {
      hasWarning: false,
      severity: "safe",
      title: "No obvious conflict found",
      message: "No obvious allergy or medication conflict was found in the saved profile. Keep following your doctor's instructions.",
      reasons,
    };
  }

  const danger = Boolean(allergyHit) || reasons.length > 1;
  return {
    hasWarning: true,
    severity: danger ? "danger" : "caution",
    title: danger ? "Medication safety warning" : "Medication caution",
    message: `${danger ? "Warning" : "Caution"}: ${reasons.join(" ")} Save it only if your doctor prescribed it, and contact a clinician if you are unsure.`,
    reasons,
  };
}
