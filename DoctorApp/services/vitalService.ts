import { API } from "../constants/api";
import { apiFetch } from "./http";

// ============================================
// Types
// ============================================
export interface VitalReading {
  id: number;
  patientId: number;
  chronicDiseaseMonitorId?: number;
  readingType: string;
  value: number;
  value2?: number;
  unit: string;
  sugarReadingContext?: string;
  isNormal: boolean;
  recordedBy: string;
  notes?: string;
  recordedAt: string;
}

export interface CreateVitalPayload {
  chronicDiseaseMonitorId?: number;
  readingType: string;
  value: number;
  value2?: number;
  unit: string;
  sugarReadingContext?: string;
  isNormal: boolean;
  notes?: string;
  recordedBy?: string;
}

export interface VitalTrendPoint {
  date: string;
  value: number;
  value2?: number;
}

// ============================================
// Get Patient Vitals
// ============================================
export const getPatientVitals = async (patientId: number, type?: string): Promise<VitalReading[]> => {
  const url = type
    ? `${API.records.vitals(patientId)}?type=${encodeURIComponent(type)}`
    : API.records.vitals(patientId);
  const data = await apiFetch<any>(url, { method: "GET" }, true);
  return Array.isArray(data) ? data : [];
};

// ============================================
// Get Latest Vital Reading
// ============================================
export const getLatestVital = async (patientId: number, type: string): Promise<VitalReading | null> => {
  const result = await apiFetch<VitalReading | undefined>(
    API.records.vitalsLatest(patientId, type),
    { method: "GET", allowedStatusCodes: [404] },
    true
  );
  return result ?? null;
};

// ============================================
// Get Vital Trend
// ============================================
export const getVitalTrend = async (patientId: number, type: string, days = 30): Promise<VitalTrendPoint[]> => {
  const data = await apiFetch<any>(
    API.records.vitalsTrend(patientId, type, days),
    { method: "GET" },
    true
  );
  return Array.isArray(data) ? data : [];
};

// ============================================
// Add Vital Reading (Patient can self-report)
// ============================================
export const addVitalReading = async (patientId: number, payload: CreateVitalPayload): Promise<VitalReading> => {
  const bodyPayload: Record<string, unknown> = {
    readingType: payload.readingType,
    value: payload.value,
    unit: payload.unit,
    isNormal: payload.isNormal,
  };
  if (payload.value2 != null) bodyPayload.value2 = payload.value2;
  if (payload.notes) bodyPayload.notes = payload.notes;
  if (payload.sugarReadingContext) bodyPayload.sugarReadingContext = payload.sugarReadingContext;
  if (payload.chronicDiseaseMonitorId != null) bodyPayload.chronicDiseaseMonitorId = payload.chronicDiseaseMonitorId;

  return apiFetch<VitalReading>(
    API.records.vitals(patientId),
    {
      method: "POST",
      body: JSON.stringify(bodyPayload),
    },
    true
  );
};

// ============================================
// Normal Ranges for Live Validation
// ============================================
export const NORMAL_RANGES: Record<string, { min: number; max: number; unit: string }> = {
  "Blood Pressure Systolic": { min: 90, max: 120, unit: "mmHg" },
  "Blood Pressure Diastolic": { min: 60, max: 80, unit: "mmHg" },
  "Blood Sugar": { min: 70, max: 100, unit: "mg/dL" },
  "Heart Rate": { min: 60, max: 100, unit: "bpm" },
  "Temperature": { min: 36.1, max: 37.2, unit: "°C" },
  "SpO2": { min: 95, max: 100, unit: "%" },
  "Respiratory Rate": { min: 12, max: 20, unit: "breaths/min" },
};

export function checkVitalNormal(type: string, value: number, value2?: number): boolean {
  const range = NORMAL_RANGES[type];
  if (!range) return true;
  if (value < range.min || value > range.max) return false;
  if (value2 != null) {
    if (value2 < range.min || value2 > range.max) return false;
  }
  return true;
}

export function getVitalRangeText(type: string): string {
  const r = NORMAL_RANGES[type];
  return r ? `${r.min}–${r.max} ${r.unit}` : "";
}
