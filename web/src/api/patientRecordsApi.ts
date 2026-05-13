import axiosInstance from './axiosInstance'

export interface AllergyRecord {
  id: number
  allergenName: string
  allergyType: string
  severity: string
  reactionDescription?: string
  firstObservedDate?: string
  isActive: boolean
}

export interface ChronicDiseaseRecord {
  id: number
  diseaseName: string
  diseaseType: string
  diagnosedDate?: string
  severity: string
  monitoringFrequency?: string
  doctorNotes?: string
  isActive: boolean
}

export interface SurgeryRecord {
  id: number
  surgeryName: string
  surgeryDate?: string
  hospitalName?: string
  doctorName?: string
  complications?: string
  notes?: string
}

export interface VitalRecord {
  id: number
  readingType: string
  value: number
  value2?: number
  unit: string
  isNormal: boolean
  recordedAt: string
  notes?: string
}

export interface PatientDocument {
  id: number
  documentType: string
  title: string
  fileUrl: string
  fileType: string
  description?: string
  documentDate?: string
  uploadedAt: string
}

export interface MedicationRecord {
  id: number
  patientId: number
  prescribedByDoctorId?: number
  chronicDiseaseMonitorId?: number
  medicationName: string
  genericName?: string
  dosage: string
  form: string
  frequency: string
  timesPerDay: number
  doseTimes: string
  daysOfWeek: string
  startDate: string
  endDate?: string
  instructions?: string
  pillsRemaining?: number
  refillThreshold: number
  isChronic: boolean
  isActive: boolean
  createdAt: string
}

export interface MedicalProfile {
  id: number
  bloodType?: string
  weightKg?: number
  heightCm?: number
  isSmoker: boolean
  smokingDetails?: string
  drinksAlcohol: boolean
  exerciseHabits?: string
  emergencyContactName?: string
  emergencyContactPhone?: string
  emergencyContactRelation?: string
  aiDiagnosisSummary?: string
  lastAiAnalysisAt?: string
}

export const patientRecordsApi = {
  getAllergies: (patientId: string | number) =>
    axiosInstance.get<AllergyRecord[]>(`/api/patients/${patientId}/allergies`).then((r) => r.data),

  createAllergy: (patientId: string | number, data: Omit<AllergyRecord, 'id'>) =>
    axiosInstance.post<AllergyRecord>(`/api/patients/${patientId}/allergies`, data).then((r) => r.data),

  deleteAllergy: (id: number) =>
    axiosInstance.delete(`/api/allergies/${id}`).then((r) => r.data),

  getChronicDiseases: (patientId: string | number) =>
    axiosInstance.get<ChronicDiseaseRecord[]>(`/api/patients/${patientId}/chronic-diseases`).then((r) => r.data),

  createChronicDisease: (patientId: string | number, data: any) =>
    axiosInstance.post(`/api/patients/${patientId}/chronic-diseases`, data).then((r) => r.data),

  deleteChronicDisease: (id: number) =>
    axiosInstance.delete(`/api/chronic-diseases/${id}`).then((r) => r.data),

  getSurgeries: (patientId: string | number) =>
    axiosInstance.get<SurgeryRecord[]>(`/api/patients/${patientId}/surgeries`).then((r) => r.data),

  createSurgery: (patientId: string | number, data: Omit<SurgeryRecord, 'id'>) =>
    axiosInstance.post<SurgeryRecord>(`/api/patients/${patientId}/surgeries`, data).then((r) => r.data),

  deleteSurgery: (id: number) =>
    axiosInstance.delete(`/api/surgeries/${id}`).then((r) => r.data),

  getVitals: (patientId: string | number) =>
    axiosInstance.get<VitalRecord[]>(`/api/patients/${patientId}/vitals`).then((r) => r.data),

  deleteVital: (id: number) =>
    axiosInstance.delete(`/api/vitals/${id}`).then((r) => r.data),

  getMedications: (patientId: string | number) =>
    axiosInstance.get<MedicationRecord[]>(`/api/patients/${patientId}/medications`).then((r) => r.data),

  createMedication: (patientId: string | number, data: Omit<MedicationRecord, 'id'>) =>
    axiosInstance.post<MedicationRecord>(`/api/patients/${patientId}/medications`, data).then((r) => r.data),

  deleteMedication: (id: number) =>
    axiosInstance.delete(`/api/medications/${id}`).then((r) => r.data),

  createVital: (patientId: string | number, data: Omit<VitalRecord, 'id'>) =>
    axiosInstance.post<VitalRecord>(`/api/patients/${patientId}/vitals`, data).then((r) => r.data),

  getDocuments: (patientId: string | number) =>
    axiosInstance.get<PatientDocument[]>(`/api/patients/${patientId}/documents`).then((r) => r.data),

  uploadDocument: (patientId: string | number, file: File, documentType: string, title: string, description?: string) => {
    const form = new FormData()
    form.append('file', file)
    form.append('documentType', documentType)
    form.append('title', title)
    if (description) form.append('description', description)
    return axiosInstance.post<PatientDocument>(`/api/patients/${patientId}/documents/upload`, form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data)
  },

  deleteDocument: (id: number) =>
    axiosInstance.delete(`/api/patient-documents/${id}`).then((r) => r.data),

  getMedicalProfile: (patientId: string | number) =>
    axiosInstance.get<MedicalProfile>(`/api/patients/${patientId}/medical-profile`).then((r) => r.data),

  updateMedicalProfile: (patientId: string | number, data: Partial<MedicalProfile>) =>
    axiosInstance.put(`/api/patients/${patientId}/medical-profile`, data).then((r) => r.data),
}
