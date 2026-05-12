import { Platform } from "react-native";
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

export interface SurgeryRecord {
  id: number;
  surgeryName: string;
  surgeryDate?: string;
  hospitalName?: string;
  doctorName?: string;
  complications?: string;
  notes?: string;
}

export interface PatientDocument {
  id: number;
  documentType: string;
  title: string;
  fileUrl: string;
  fileType: string;
  description?: string;
  documentDate: string;
  uploadedAt: string;
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

export const deleteMedication = async (medId: number): Promise<void> =>
  apiFetch<any>(API.records.medicationDelete(medId), { method: "DELETE" }, true);

// ============================================
// Allergy CRUD
// ============================================
export const createAllergy = async (patientId: number, payload: Omit<AllergyRecord, "id">): Promise<AllergyRecord> =>
  apiFetch<any>(API.records.allergies(patientId), { method: "POST", body: JSON.stringify(payload) }, true);

export const updateAllergy = async (allergyId: number, payload: Partial<AllergyRecord>): Promise<AllergyRecord> =>
  apiFetch<any>(API.records.allergyUpdate(allergyId), { method: "PATCH", body: JSON.stringify(payload) }, true);

export const deleteAllergy = async (allergyId: number): Promise<void> =>
  apiFetch<any>(API.records.allergyDelete(allergyId), { method: "DELETE" }, true);

// ============================================
// Surgery CRUD
// ============================================
export const getSurgeries = async (patientId: number): Promise<SurgeryRecord[]> => {
  const data = await apiFetch<any>(API.records.surgeries(patientId), { method: "GET" }, true);
  return Array.isArray(data) ? data : [];
};

export const createSurgery = async (patientId: number, payload: Omit<SurgeryRecord, "id">): Promise<SurgeryRecord> =>
  apiFetch<any>(API.records.surgeries(patientId), { method: "POST", body: JSON.stringify(payload) }, true);

export const updateSurgery = async (surgeryId: number, payload: Partial<SurgeryRecord>): Promise<SurgeryRecord> =>
  apiFetch<any>(API.records.surgeryUpdate(surgeryId), { method: "PATCH", body: JSON.stringify(payload) }, true);

export const deleteSurgery = async (surgeryId: number): Promise<void> =>
  apiFetch<any>(API.records.surgeryDelete(surgeryId), { method: "DELETE" }, true);

// ============================================
// Chronic Disease CRUD
// ============================================
export const createChronicDisease = async (patientId: number, payload: any): Promise<any> =>
  apiFetch<any>(API.records.chronicDiseases(patientId), { method: "POST", body: JSON.stringify(payload) }, true);

export const updateChronicDisease = async (chronicId: number, payload: any): Promise<any> =>
  apiFetch<any>(API.records.chronicUpdate(chronicId), { method: "PATCH", body: JSON.stringify(payload) }, true);

export const deleteChronicDisease = async (chronicId: number): Promise<void> =>
  apiFetch<any>(API.records.chronicDelete(chronicId), { method: "DELETE" }, true);

// ============================================
// Vital CRUD
// ============================================
export const createVital = async (patientId: number, payload: any): Promise<any> =>
  apiFetch<any>(API.records.vitals(patientId), { method: "POST", body: JSON.stringify(payload) }, true);

export const deleteVital = async (vitalId: number): Promise<void> =>
  apiFetch<any>(API.records.vitalDelete(vitalId), { method: "DELETE" }, true);

// ============================================
// Patient Documents (Scans / Labs)
// ============================================
export const getPatientDocuments = async (patientId: number, type?: string): Promise<PatientDocument[]> => {
  const url = type ? `${API.records.documents(patientId)}?type=${encodeURIComponent(type)}` : API.records.documents(patientId);
  const data = await apiFetch<any>(url, { method: "GET" }, true);
  return Array.isArray(data) ? data : [];
};

export const uploadPatientDocument = async (
  patientId: number,
  uri: string,
  documentType: string,
  title: string,
  description?: string
): Promise<PatientDocument> => {
  const formData = new FormData();

  const cleanUri = Platform.OS === 'android' ? uri : uri.replace('file://', '');
  const filename = uri.split("/").pop() || "document.jpg";
  const type = "image/jpeg"; // Most common, or extract from extension

  formData.append("file", { 
    uri: uri, 
    name: filename, 
    type: type 
  } as any);
  formData.append("documentType", documentType);
  formData.append("title", title);
  if (description) formData.append("description", description);

  console.log("[DocumentUpload] Uploading:", { filename, documentType, title });

  return apiFetch<any>(API.records.documentUpload(patientId), { method: "POST", body: formData }, true);
};

export const deletePatientDocument = async (docId: number): Promise<void> =>
  apiFetch<any>(API.records.documentDelete(docId), { method: "DELETE" }, true);

// ============================================
// AI Diagnosis
// ============================================
export const updateAiDiagnosis = async (patientId: number, diagnosisSummary: string): Promise<any> =>
  apiFetch<any>(
    `${API.patients.profile(patientId).replace('/profile', '/ai-diagnosis')}`,
    { 
      method: "PATCH", 
      body: JSON.stringify({ diagnosisSummary }) 
    }, 
    true
  );
