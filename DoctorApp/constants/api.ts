import Constants from "expo-constants";

const configuredBaseUrl = (Constants.expoConfig?.extra as { apiBaseUrl?: string } | undefined)?.apiBaseUrl;

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

// Prefer the current Expo host in development so mobile builds do not keep using a stale LAN IP.
export const BASE_URL = "https://ai-medical-assistant-production-38a3.up.railway.app";

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
};
