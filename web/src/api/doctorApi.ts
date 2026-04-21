import axiosInstance from './axiosInstance'
import type {
  DoctorDashboardDto,
  DoctorDetailDto,
  AppointmentDto,
  PatientSummaryDto,
  AIReportDto,
  AvailabilityDto,
  ReviewDto,
} from '@/lib/types'

export const doctorApi = {
  // Get all doctors (public endpoint)
  getAllDoctors: (specialtyId?: number) =>
    axiosInstance.get<DoctorDetailDto[]>('/api/doctors', { params: { specialtyId } }).then((r) => r.data),

  // Get doctor by ID (public endpoint)
  getDoctorById: (id: string) =>
    axiosInstance.get<DoctorDetailDto>(`/api/doctors/${id}`).then((r) => r.data),

  // These endpoints would need to be implemented in the backend
  getDashboard: () =>
    axiosInstance.get<DoctorDashboardDto>('/api/doctors/dashboard').then((r) => r.data),

  getProfile: () =>
    axiosInstance.get<DoctorDetailDto>('/api/doctors/profile').then((r) => r.data),

  updateProfile: (data: Partial<DoctorDetailDto>) =>
    axiosInstance.put('/api/doctors/profile', data).then(() => undefined),

  uploadPhoto: (file: File) => {
    const form = new FormData()
    form.append('photo', file)
    return axiosInstance.post<{ photoUrl: string }>('/api/doctors/photo', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data)
  },

  getAppointments: (status?: string) =>
    axiosInstance.get<AppointmentDto[]>('/api/doctors/appointments', { params: { status } }).then((r) => r.data),

  getPatients: (search?: string) =>
    axiosInstance.get<PatientSummaryDto[]>('/api/doctors/patients', { params: { search } }).then((r) => r.data),

  getReports: (params?: { urgency?: string; patientId?: string; from?: string; to?: string }) =>
    axiosInstance.get<AIReportDto[]>('/api/doctors/reports', { params }).then((r) => r.data),

  getAvailability: () =>
    axiosInstance.get<AvailabilityDto[]>('/api/doctors/availability').then((r) => r.data),

  updateAvailability: (data: AvailabilityDto[]) =>
    axiosInstance.put('/api/doctors/availability', data).then(() => undefined),

  getReviews: () =>
    axiosInstance.get<ReviewDto[]>('/api/doctors/reviews').then((r) => r.data),

  updateScheduleVisibility: (isVisible: boolean) =>
    axiosInstance.put('/api/doctors/schedule-visibility', isVisible).then(() => undefined),
}
