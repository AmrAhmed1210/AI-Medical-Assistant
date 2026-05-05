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
// Get My Visits
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
