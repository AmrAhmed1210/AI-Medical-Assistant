export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'https://ai-medical-assistant-production-38a3.up.railway.app'
export const SIGNALR_HUB_URL = import.meta.env.VITE_SIGNALR_HUB_URL || 'https://ai-medical-assistant-production-38a3.up.railway.app/hubs/notifications' 

export const APP_NAME = import.meta.env.VITE_APP_NAME || 'MedBook'

export const TOKEN_KEY = 'medbook_token'
export const REFRESH_TOKEN_KEY = 'medbook_refresh_token'
export const USER_KEY = 'medbook_user'

export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 10,
  PAGE_SIZE_OPTIONS: [10, 20, 50],
}

export const ROUTES = {
  LOGIN: '/login',
  // Admin
  ADMIN_DASHBOARD: '/admin/dashboard',
  ADMIN_USERS: '/admin/users',
  ADMIN_STATISTICS: '/admin/statistics',
  ADMIN_MODELS: '/admin/models',
  ADMIN_APPLICATIONS: '/admin/applications',
  ADMIN_SUPPORT: '/admin/support',
  // Doctor
  DOCTOR_DASHBOARD: '/doctor/dashboard',
  DOCTOR_PROFILE: '/doctor/profile',
  DOCTOR_SCHEDULE: '/doctor/schedule',
  DOCTOR_APPOINTMENTS: '/doctor/appointments',
  DOCTOR_PATIENTS: '/doctor/patients',
  DOCTOR_REPORTS: '/doctor/reports',
  DOCTOR_REVIEWS: '/doctor/reviews',
  DOCTOR_CHAT: '/doctor/chat',
}
