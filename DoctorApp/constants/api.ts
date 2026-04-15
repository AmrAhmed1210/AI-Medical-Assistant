// ============================================
// BASE URL — غير الـ IP لو شغال على موبايل
// ============================================
export const BASE_URL = "http://192.168.43.216:5194";
// ============================================
// ENDPOINTS
// ============================================
export const API = {
  // Auth
  auth: {
    register: `${BASE_URL}/auth/register`,
    login:    `${BASE_URL}/auth/login`,
  },

  // Doctors
  doctors: {
    getAll:       `${BASE_URL}/doctors`,
    getById:      (id: number | string) => `${BASE_URL}/doctors/${id}`,
    getBySpecialty: (specialtyId: number) => `${BASE_URL}/doctors?specialtyId=${specialtyId}`,
  },

  // Appointments
  appointments: {
    book:   `${BASE_URL}/appointments`,
    my:     `${BASE_URL}/appointments/my`,
    cancel: (id: number | string) => `${BASE_URL}/appointments/${id}`,
  },

  // Reviews
  reviews: {
    getByDoctor: (doctorId: number | string) => `${BASE_URL}/reviews/${doctorId}`,
    add:         `${BASE_URL}/reviews`,
  },

  // Profile
  profile: {
    get:    `${BASE_URL}/profile/me`,
    update: `${BASE_URL}/profile/me`,
  },
};