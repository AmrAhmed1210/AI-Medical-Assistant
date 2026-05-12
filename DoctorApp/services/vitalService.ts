import { API, BASE_URL } from "../constants/api";
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
  "Blood Pressure Systolic": { min: 90, max: 140, unit: "mmHg" },
  "Blood Pressure Diastolic": { min: 60, max: 90, unit: "mmHg" },
  "Blood Sugar": { min: 70, max: 140, unit: "mg/dL" },
  "Heart Rate": { min: 60, max: 100, unit: "bpm" },
  "Temperature": { min: 36.1, max: 37.5, unit: "°C" },
  "SpO2": { min: 95, max: 100, unit: "%" },
  "Respiratory Rate": { min: 12, max: 20, unit: "breaths/min" },
};

// ============================================
// Update Vital Reading
// ============================================
export const updateVitalReading = async (vitalId: number, payload: Partial<CreateVitalPayload>): Promise<VitalReading> => {
  return apiFetch<VitalReading>(
    `${BASE_URL}/api/vitals/${vitalId}`,
    {
      method: "PUT",
      body: JSON.stringify(payload),
    },
    true
  );
};

// ============================================
// Delete Vital Reading
// ============================================
export const deleteVitalReading = async (vitalId: number): Promise<void> => {
  await apiFetch<void>(
    API.records.vitalDelete(vitalId),
    { method: "DELETE" },
    true
  );
};

export function checkVitalNormal(type: string, value: number, value2?: number): boolean {
  if (type === "Blood Pressure") {
    const sysRange = NORMAL_RANGES["Blood Pressure Systolic"];
    const diaRange = NORMAL_RANGES["Blood Pressure Diastolic"];
    if (value < sysRange.min || value > sysRange.max) return false;
    if (value2 != null && (value2 < diaRange.min || value2 > diaRange.max)) return false;
    return true;
  }
  const range = NORMAL_RANGES[type];
  if (!range) return true;
  return value >= range.min && value <= range.max;
}

export function getVitalRangeText(type: string): string {
  if (type === "Blood Pressure") {
    return "90-140 / 60-90 mmHg";
  }
  const r = NORMAL_RANGES[type];
  return r ? `${r.min}–${r.max} ${r.unit}` : "";
}

