import { API } from "../constants/api";
import { apiFetch } from "./http";

// ============================================
// Types
// ============================================
export interface MedicationTracker {
  id: number;
  patientId: number;
  prescribedByDoctorId?: number;
  chronicDiseaseMonitorId?: number;
  medicationName: string;
  genericName?: string;
  dosage: string;
  form: string;
  frequency: string;
  timesPerDay: number;
  doseTimes: string;
  daysOfWeek: string;
  startDate: string;
  endDate?: string;
  instructions?: string;
  pillsRemaining?: number;
  refillThreshold: number;
  isChronic: boolean;
  isActive: boolean;
  createdAt: string;
}

export interface MedicationScheduleItem {
  logId: number;
  medicationTrackerId: number;
  scheduledAt: string;
  medicationName: string;
  dosage: string;
  status: string; // pending / taken / missed / skipped
}

export interface CreateMedicationPayload {
  chronicDiseaseMonitorId?: number;
  medicationName: string;
  genericName?: string;
  dosage: string;
  form: string;
  frequency: string;
  timesPerDay: number;
  doseTimes: string;
  daysOfWeek: string;
  startDate: string;
  endDate?: string;
  instructions?: string;
  pillsRemaining?: number;
  refillThreshold: number;
  isChronic: boolean;
  isActive?: boolean;
}

// ============================================
// Get Active Medications
// ============================================
export const getPatientMedications = async (patientId: number): Promise<MedicationTracker[]> => {
  const data = await apiFetch<any>(
    API.records.medications(patientId),
    { method: "GET" },
    true
  );
  return Array.isArray(data) ? data : [];
};

// ============================================
// Get Today's Medication Schedule
// ============================================
export const getMedicationSchedule = async (patientId: number): Promise<MedicationScheduleItem[]> => {
  const data = await apiFetch<any>(
    API.records.medicationsSchedule(patientId),
    { method: "GET" },
    true
  );
  return Array.isArray(data) ? data : [];
};

// ============================================
// Create Medication (Patient self-add)
// ============================================
export const createPatientMedication = async (patientId: number, payload: CreateMedicationPayload): Promise<MedicationTracker> => {
  const data = await apiFetch<any>(
    API.records.medicationsSelf(patientId),
    { method: "POST", body: JSON.stringify(payload) },
    true
  );
  return data;
};

// ============================================
// Mark Medication as Taken
// ============================================
export const markMedicationTaken = async (logId: number): Promise<void> => {
  await apiFetch<any>(
    API.records.medicationLogTaken(logId),
    { method: "POST" },
    true
  );
};

// ============================================
// Update Medication
// ============================================
export const updateMedication = async (medicationId: number, payload: Partial<CreateMedicationPayload>): Promise<MedicationTracker> => {
  const data = await apiFetch<any>(
    API.records.medicationUpdate(medicationId),
    { method: "PATCH", body: JSON.stringify(payload) },
    true
  );
  return data;
};

// ============================================
// Delete (Deactivate) Medication
// ============================================
export const deleteMedication = async (medicationId: number): Promise<void> => {
  await apiFetch<any>(
    API.records.medicationDelete(medicationId),
    { method: "DELETE" },
    true
  );
};
