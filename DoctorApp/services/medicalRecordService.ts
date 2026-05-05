import { API } from "../constants/api";
import { apiFetch } from "./http";

// ============================================
// Types
// ============================================
export interface AllergyRecord {
  id: number;
  allergyType: string;
  allergenName: string;
  severity: string;
  reactionDescription?: string;
  firstObservedDate?: string;
  isActive: boolean;
}

export interface ChronicDisease {
  id: number;
  diseaseName: string;
  diseaseType: string;
  diagnosedDate?: string;
  severity: string;
  isActive: boolean;
  doctorNotes?: string;
  targetValues?: string;
  monitoringFrequency: string;
}

export interface Medication {
  id: number;
  medicationName: string;
  genericName?: string;
  dosage: string;
  form: string;
  frequency: string;
  startDate: string;
  endDate?: string;
  instructions?: string;
  isChronic: boolean;
  isActive: boolean;
}

export interface VitalReading {
  id: number;
  readingType: string;
  value: number;
  value2?: number;
  unit: string;
  isNormal: boolean;
  recordedAt: string;
  notes?: string;
}

// ============================================
// Get Allergies
// ============================================
export const getAllergies = async (patientId: number): Promise<AllergyRecord[]> => {
  const data = await apiFetch<any>(
    API.records.allergies(patientId),
    { method: "GET" },
    true
  );
  return Array.isArray(data) ? data : [];
};

// ============================================
// Get Chronic Diseases
// ============================================
export const getChronicDiseases = async (patientId: number): Promise<ChronicDisease[]> => {
  const data = await apiFetch<any>(
    API.records.chronicDiseases(patientId),
    { method: "GET" },
    true
  );
  const items = Array.isArray(data) ? data : [];
  return items.map((item: any) => ({
    id: Number(item.id ?? item.Id ?? 0),
    diseaseName: String(item.diseaseName ?? item.DiseaseName ?? ""),
    diseaseType: String(item.diseaseType ?? item.DiseaseType ?? ""),
    diagnosedDate: item.diagnosedDate ?? item.DiagnosedDate,
    severity: String(item.severity ?? item.Severity ?? ""),
    isActive: Boolean(item.isActive ?? item.IsActive),
    doctorNotes: item.doctorNotes ?? item.DoctorNotes,
    targetValues: item.targetValues ?? item.TargetValues,
    monitoringFrequency: String(item.monitoringFrequency ?? item.MonitoringFrequency ?? ""),
  }));
};

// ============================================
// Get Medications
// ============================================
export const getMedications = async (patientId: number): Promise<Medication[]> => {
  const data = await apiFetch<any>(
    API.records.medications(patientId),
    { method: "GET" },
    true
  );
  return Array.isArray(data) ? data : [];
};

// ============================================
// Get Vitals
// ============================================
export const getVitals = async (patientId: number): Promise<VitalReading[]> => {
  const data = await apiFetch<any>(
    API.records.vitals(patientId),
    { method: "GET" },
    true
  );
  return Array.isArray(data) ? data : [];
};
