import axiosInstance from './axiosInstance'

export interface VisitDto {
  id: number
  patientId: number
  doctorId: number
  appointmentId?: number
  patientName: string
  patientAge: number
  chiefComplaint: string
  examinationFindings?: string
  assessment?: string
  plan?: string
  notes?: string
  visitDate: string
  status: 'open' | 'closed' | 'cancelled'
  summarySnapshot?: string
  followUpRequired?: boolean
  followUpDate?: string
  followUpTime?: string
  followUpNotes?: string
  createdAt: string
}

export interface VitalSignDto {
  type: string
  value: string
  value2?: string
  unit: string
  isAbnormal: boolean
}

export interface SymptomDto {
  name: string
  severity: string
  onset: string
  location?: string
  duration?: string
  isChronic: boolean
}

export interface PrescriptionDto {
  medicationName: string
  dosage: string
  frequency: string
  duration: string
  quantity: number
  instructions?: string
  isChronic: boolean
}

export interface CreateVisitRequest {
  patientId: number
  doctorId: number
  appointmentId?: number
  chiefComplaint: string
}

export interface UpdateVisitRequest {
  chiefComplaint?: string
  examinationFindings?: string
  assessment?: string
  plan?: string
  notes?: string
  symptoms?: SymptomDto[]
  prescriptions?: PrescriptionDto[]
  vitalSigns?: VitalSignDto[]
  followUpRequired?: boolean
  followUpDate?: string
  followUpTime?: string
  followUpNotes?: string
}

export interface VisitSummaryDto {
  id: number
  patientName: string
  patientAge: number
  bloodType: string
  allergies: { allergenName: string; severity: string; reaction: string }[]
  visitDate: string
  chiefComplaint: string
  examinationFindings: string
  assessment: string
  plan: string
  vitalSigns: VitalSignDto[]
  prescriptions: PrescriptionDto[]
  symptoms: SymptomDto[]
  notes?: string
  followUpRequired?: boolean
  followUpDate?: string
  followUpTime?: string
  followUpNotes?: string
  recentVisits?: LastVisitSummaryDto[]
  visitsTimelineSummaryEn?: string
  visitsTimelineSummaryAr?: string
}

export interface LastVisitSummaryDto {
  id: string
  visitDate: string
  chiefComplaint: string
  doctorName?: string
  doctorSpecialty?: string
  summary?: string
  summaryEn?: string
  summaryAr?: string
}

export const visitApi = {
  // Get today's visits for doctor
  getTodayVisits: () =>
    axiosInstance.get<VisitDto[]>('/api/visits/doctor/today').then((r) => r.data),

  // Create new visit
  createVisit: (data: CreateVisitRequest) =>
    axiosInstance.post<VisitDto>('/api/visits', data).then((r) => r.data),

  // Get visit by ID
  getVisit: (id: number) =>
    axiosInstance.get<VisitDto>(`/api/visits/${id}`).then((r) => r.data),

  // Update visit (draft)
  updateVisit: (id: number, data: UpdateVisitRequest) =>
    axiosInstance.patch<VisitDto>(`/api/visits/${id}`, data).then((r) => r.data),

  // Close visit
  closeVisit: (id: number) =>
    axiosInstance.patch<VisitDto>(`/api/visits/${id}/close`).then((r) => r.data),

  // Get visit summary
  getSummary: (id: number) =>
    axiosInstance.get<VisitSummaryDto>(`/api/visits/${id}/summary`).then((r) => r.data),

  // Download PDF
  downloadPdf: (id: number) =>
    axiosInstance.get(`/api/visits/${id}/summary-pdf`, { responseType: 'blob' }).then((r) => r.data),

  // Get patient history for workspace
  getPatientHistory: (patientId: number) =>
    axiosInstance.get(`/api/patients/${patientId}/history`).then((r) => r.data),
}
