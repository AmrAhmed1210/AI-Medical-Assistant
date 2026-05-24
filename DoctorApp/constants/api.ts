import Constants from "expo-constants";

const configuredBaseUrl = (
  (Constants.expoConfig?.extra as { apiBaseUrl?: string } | undefined)?.apiBaseUrl || ""
).trim() || undefined;

function getExpoHost(): string | null {
  const candidates = [
    (Constants as any).expoGoConfig?.debuggerHost,
    (Constants as any).manifest2?.extra?.expoClient?.hostUri,
    (Constants.expoConfig as any)?.hostUri,
  ];

  for (const candidate of candidates) {
    if (typeof candidate !== "string" || !candidate.trim()) continue;
    const [host] = candidate.split(":");
    if (host) return host;
  }

  return null;
}

const expoHost = getExpoHost();
const inferredBaseUrl = expoHost ? `http://${expoHost}:5194` : null;

// Expo host updates when you change Wi‑Fi; app.json apiBaseUrl is only a fallback.
export const BASE_URL = inferredBaseUrl || configuredBaseUrl || "http://localhost:5194";

if (__DEV__) {
  console.log("[API] Using backend:", BASE_URL);
}

// ============================================
// ENDPOINTS
// ============================================
export const API = {
  // Auth
  auth: {
    register: `${BASE_URL}/api/auth/register`,
    login:    `${BASE_URL}/api/auth/login`,
  },

  // Doctors
  doctors: {
    getAll:         `${BASE_URL}/api/doctors`,
    getById:        (id: number | string) => `${BASE_URL}/api/doctors/${id}`,
    getBySpecialty: (specialtyId: number) => `${BASE_URL}/api/doctors?specialtyId=${specialtyId}`,
    dashboard:      `${BASE_URL}/api/doctors/dashboard`,
    profile:        `${BASE_URL}/api/doctors/profile`,
    availability:   `${BASE_URL}/api/doctors/availability`,
    appointments:   `${BASE_URL}/api/doctors/appointments`,
    patients:       `${BASE_URL}/api/doctors/patients`,
    reports:        `${BASE_URL}/api/doctors/reports`,
    uploadPhoto:    `${BASE_URL}/api/doctors/photo`,
  },

  // Appointments
  appointments: {
    book:   `${BASE_URL}/api/appointments`,
    my:     `${BASE_URL}/api/appointments/my`,
    cancel: (id: number | string) => `${BASE_URL}/api/appointments/${id}`,
  },

  // Reviews
  reviews: {
    getByDoctor: (doctorId: number | string) => `${BASE_URL}/api/reviews/${doctorId}`,
    add:         `${BASE_URL}/api/reviews`,
    updateMine:  (doctorId: number | string) => `${BASE_URL}/api/reviews/${doctorId}/mine`,
    deleteMine:  (doctorId: number | string) => `${BASE_URL}/api/reviews/${doctorId}/mine`,
    updateById:  (reviewId: number | string) => `${BASE_URL}/api/reviews/${reviewId}`,
    deleteById:  (reviewId: number | string) => `${BASE_URL}/api/reviews/${reviewId}`,
  },

  // Profile
  profile: {
    get:    `${BASE_URL}/api/profile/me`,
    update: `${BASE_URL}/api/profile/me`,
    photo:  `${BASE_URL}/api/profile/photo`,
  },

  // Visits
  visits: {
    my:           `${BASE_URL}/api/visits/my`,
    open:         `${BASE_URL}/api/visits`,
    getById:      (id: number | string) => `${BASE_URL}/api/visits/${id}`,
    update:       (id: number | string) => `${BASE_URL}/api/visits/${id}`,
    close:        (id: number | string) => `${BASE_URL}/api/visits/${id}/close`,
    summary:      (id: number | string) => `${BASE_URL}/api/visits/${id}/summary`,
    doctorToday:  `${BASE_URL}/api/visits/doctor/today`,
    patientHistory: (id: number | string) => `${BASE_URL}/api/patients/${id}/history`,
  },

  // Patient
  patient: {
    me: `${BASE_URL}/api/patients/me`,
  },

  // Patients
  patients: {
    profile: (id: number | string) => `${BASE_URL}/api/patients/${id}/profile`,
  },

  // Patient Medical Records
  records: {
    allergies:       (patientId: number | string) => `${BASE_URL}/api/patients/${patientId}/allergies`,
    chronicDiseases: (patientId: number | string) => `${BASE_URL}/api/patients/${patientId}/chronic-diseases`,
    medications:     (patientId: number | string) => `${BASE_URL}/api/patients/${patientId}/medications`,
    medicationsSelf: (patientId: number | string) => `${BASE_URL}/api/patients/${patientId}/medications/self`,
    medicationsSchedule: (patientId: number | string) => `${BASE_URL}/api/patients/${patientId}/medications/schedule`,
    medicationUpdate: (medicationId: number | string) => `${BASE_URL}/api/medications/${medicationId}`,
    medicationDelete: (medicationId: number | string) => `${BASE_URL}/api/medications/${medicationId}`,
    medicationLogTaken: (logId: number | string) => `${BASE_URL}/api/medication-logs/${logId}/taken`,
    vitals:          (patientId: number | string) => `${BASE_URL}/api/patients/${patientId}/vitals`,
    vitalsLatest:    (patientId: number | string, type: string) => `${BASE_URL}/api/patients/${patientId}/vitals/latest?type=${encodeURIComponent(type)}`,
    vitalsTrend:     (patientId: number | string, type: string, days?: number) => `${BASE_URL}/api/patients/${patientId}/vitals/trend?type=${encodeURIComponent(type)}${days ? `&days=${days}` : ""}`,
    surgeries:       (patientId: number | string) => `${BASE_URL}/api/patients/${patientId}/surgeries`,
    familyHistory:   (patientId: number | string) => `${BASE_URL}/api/patients/${patientId}/family-history`,
    documents:       (patientId: number | string) => `${BASE_URL}/api/patients/${patientId}/documents`,
    documentUpload:  (patientId: number | string) => `${BASE_URL}/api/patients/${patientId}/documents/upload`,
    documentUpdate:    (docId: number | string) => `${BASE_URL}/api/patient-documents/${docId}`,
    documentDelete:  (docId: number | string) => `${BASE_URL}/api/patient-documents/${docId}`,
    allergyUpdate:   (allergyId: number | string) => `${BASE_URL}/api/allergies/${allergyId}`,
    allergyDelete:   (allergyId: number | string) => `${BASE_URL}/api/allergies/${allergyId}`,
    surgeryUpdate:   (surgeryId: number | string) => `${BASE_URL}/api/surgeries/${surgeryId}`,
    surgeryDelete:   (surgeryId: number | string) => `${BASE_URL}/api/surgeries/${surgeryId}`,
    chronicUpdate:   (chronicId: number | string) => `${BASE_URL}/api/chronic-diseases/${chronicId}`,
    chronicDelete:   (chronicId: number | string) => `${BASE_URL}/api/chronic-diseases/${chronicId}`,
    vitalUpdate:     (vitalId: number | string) => `${BASE_URL}/api/vitals/${vitalId}`,
    vitalDelete:     (vitalId: number | string) => `${BASE_URL}/api/vitals/${vitalId}`,
  },

  // AI Chat (doctor messaging uses /api/sessions via sessionService)
  chat: {
    ask: `${BASE_URL}/api/chat/ask`,
    analyzeHistory: `${BASE_URL}/api/chat/analyze-history`,
    analyzeImage: `${BASE_URL}/api/chat/analyze-image`,
    parseMedicalProfile: `${BASE_URL}/api/chat/parse-medical-profile`,
  },

  sessions: {
    list: `${BASE_URL}/api/sessions`,
    detail: (sessionId: number | string) => `${BASE_URL}/api/sessions/${sessionId}`,
  },
};
