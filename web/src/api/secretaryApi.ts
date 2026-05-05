import axiosInstance from './axiosInstance'
import type { SecretaryDto, CreateSecretaryRequest, DoctorDetailDto, PatientSummaryDto, AvailabilityDto } from '@/lib/types'

export const secretaryApi = {
  addSecretary: (data: CreateSecretaryRequest) =>
    axiosInstance.post<SecretaryDto>('/api/Secretary/add', data).then((r) => r.data),

  getMySecretaries: () =>
    axiosInstance.get<SecretaryDto[]>('/api/Secretary/my-secretaries').then((r) => r.data),

  deleteSecretary: (id: number) =>
    axiosInstance.delete(`/api/Secretary/${id}`).then((r) => r.data),

  // --- Secretary-specific ---
  getMyDoctor: () =>
    axiosInstance.get<DoctorDetailDto>('/api/Secretary/my-doctor').then((r) => r.data),

  searchMyDoctorPatients: (search?: string) =>
    axiosInstance.get<PatientSummaryDto[]>('/api/Secretary/my-doctor/patients', { params: { search } }).then((r) => r.data),

  getMyDoctorAvailability: () =>
    axiosInstance.get<AvailabilityDto[]>('/api/Secretary/my-doctor/availability').then((r) => r.data),

  updateMyDoctorAvailability: (data: AvailabilityDto[]) =>
    axiosInstance.put('/api/Secretary/my-doctor/availability', data).then((r) => r.data),

  updateMyDoctorScheduleVisibility: (isVisible: boolean) =>
    axiosInstance.put('/api/Secretary/my-doctor/schedule-visibility', isVisible).then((r) => r.data),

  createWalkInPatient: (data: { fullName: string; email: string; phoneNumber: string }) =>
    axiosInstance.post<{ id: number; fullName: string; email: string }>('/api/Secretary/create-walkin-patient', data).then((r) => r.data),
}
