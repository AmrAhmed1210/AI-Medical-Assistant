import { API } from "../constants/api";
import { apiFetch } from "./http";

// ============================================
// Types
// ============================================
export interface PatientVisit {
  id: number;
  patientId: number;
  doctorId: number;
  appointmentId?: number;
  chiefComplaint: string;
  presentIllnessHistory?: string;
  examinationFindings?: string;
  assessment?: string;
  plan?: string;
  notes?: string;
  status: string;
  visitDate: string;
  createdAt: string;
  closedAt?: string;
}

export interface VisitSummary {
  id: number;
  patientName: string;
  patientAge: number;
  bloodType: string;
  allergies: AllergySummary[];
  visitDate: string;
  chiefComplaint: string;
  examinationFindings?: string;
  assessment?: string;
  plan?: string;
  vitalSigns: VitalSummary[];
  prescriptions: PrescriptionSummary[];
  symptoms: SymptomSummary[];
  notes?: string;
  followUpRequired?: boolean;
  followUpAfterDays?: number;
  followUpNotes?: string;
}

export interface AllergySummary {
  allergenName: string;
  severity: string;
  reaction: string;
}

export interface VitalSummary {
  type: string;
  value: number;
  value2?: number;
  unit: string;
  isAbnormal: boolean;
}

export interface PrescriptionSummary {
  medicationName: string;
  dosage: string;
  frequency: string;
  duration?: string;
  quantity?: number;
  instructions?: string;
  isChronic: boolean;
}

export interface SymptomSummary {
  name: string;
  severity: string;
  onset: string;
  location?: string;
  duration?: string;
  isChronic: boolean;
}

// ============================================
// Doctor DTO Types
// ============================================
export interface CreateVisitPayload {
  patientId: number;
  appointmentId?: number;
  chiefComplaint: string;
  presentIllnessHistory?: string;
}

export interface UpdateSymptomPayload {
  name: string;
  severity: string;
  onset: string;
  location?: string;
  duration?: string;
  isChronic: boolean;
}

export interface UpdatePrescriptionPayload {
  medicationName: string;
  dosage: string;
  frequency: string;
  duration: string;
  quantity?: number;
  instructions?: string;
  isChronic: boolean;
}

export interface UpdateVitalPayload {
  type: string;
  value: number;
  value2?: number;
  unit: string;
  isAbnormal: boolean;
}

export interface UpdateVisitPayload {
  chiefComplaint?: string;
  presentIllnessHistory?: string;
  examinationFindings?: string;
  assessment?: string;
  plan?: string;
  notes?: string;
  symptoms?: UpdateSymptomPayload[];
  prescriptions?: UpdatePrescriptionPayload[];
  vitalSigns?: UpdateVitalPayload[];
  followUpRequired?: boolean;
  followUpAfterDays?: number;
  followUpNotes?: string;
}

export interface PatientHistory {
  bloodType: string;
  allergies: AllergySummary[];
  chronicDiseases: Array<{ id: string; diseaseName: string; targetValues: string }>;
  medications: Array<{ id: string; medicationName: string; dosage: string; form: string }>;
  latestVitals: Record<string, string>;
  lastVisits: Array<{ id: string; visitDate: string; chiefComplaint: string }>;
}

// ============================================
// Get My Visits (Patient)
// ============================================
export const getMyVisits = async (): Promise<PatientVisit[]> => {
  const data = await apiFetch<any>(
    API.visits.my,
    { method: "GET" },
    true
  );
  return Array.isArray(data) ? data : [];
};

// ============================================
// Get Visit Summary
// ============================================
export const getVisitSummary = async (visitId: number): Promise<VisitSummary> => {
  return apiFetch<VisitSummary>(
    API.visits.summary(visitId),
    { method: "GET" },
    true
  );
};

// ============================================
// Doctor: Open New Visit
// ============================================
export const openVisit = async (payload: CreateVisitPayload): Promise<PatientVisit> => {
  return apiFetch<PatientVisit>(
    API.visits.open,
    { method: "POST", body: JSON.stringify(payload) },
    true
  );
};

// ============================================
// Doctor: Get Visit by ID
// ============================================
export const getVisitById = async (visitId: number): Promise<PatientVisit> => {
  return apiFetch<PatientVisit>(
    API.visits.getById(visitId),
    { method: "GET" },
    true
  );
};

// ============================================
// Doctor: Update Visit
// ============================================
export const updateVisit = async (visitId: number, payload: UpdateVisitPayload): Promise<PatientVisit> => {
  return apiFetch<PatientVisit>(
    API.visits.update(visitId),
    { method: "PATCH", body: JSON.stringify(payload) },
    true
  );
};

// ============================================
// Doctor: Close Visit
// ============================================
export const closeVisit = async (visitId: number): Promise<void> => {
  await apiFetch<unknown>(
    API.visits.close(visitId),
    { method: "PATCH" },
    true
  );
};

// ============================================
// Doctor: Get Today's Visits
// ============================================
export const getDoctorTodayVisits = async (): Promise<PatientVisit[]> => {
  const data = await apiFetch<any>(
    API.visits.doctorToday,
    { method: "GET" },
    true
  );
  return Array.isArray(data) ? data : [];
};

// ============================================
// Doctor: Get Patient History
// ============================================
export const getPatientHistory = async (patientId: number): Promise<PatientHistory> => {
  return apiFetch<PatientHistory>(
    API.visits.patientHistory(patientId),
    { method: "GET" },
    true
  );
};
