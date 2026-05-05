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
  medicationTrackerId: number;
  scheduledAt: string;
  medicationName: string;
  dosage: string;
  status: string; // pending / taken / missed / skipped
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
