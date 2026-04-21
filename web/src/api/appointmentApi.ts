import axiosInstance from './axiosInstance'
import type { AppointmentDto, BookAppointmentRequest } from '@/lib/types'

export const appointmentApi = {
  // Get appointments for current user (patient)
  getMyAppointments: () =>
    axiosInstance.get<AppointmentDto[]>('/api/appointments/my').then((r) => r.data),

  // Get appointment by ID
  getById: (id: string) =>
    axiosInstance.get<AppointmentDto>(`/api/appointments/${id}`).then((r) => r.data),

  // Create new appointment
  create: (data: BookAppointmentRequest) =>
    axiosInstance.post<AppointmentDto>('/api/appointments', data).then((r) => r.data),

  // Delete/cancel appointment
  delete: (id: string) =>
    axiosInstance.delete(`/api/appointments/${id}`).then((r) => r.data),

  // Note: The backend doesn't have confirm/complete endpoints yet
  // These are placeholder implementations that would need backend support
  confirm: (id: string) =>
    axiosInstance.put<AppointmentDto>(`/api/appointments/${id}/confirm`).then((r) => r.data),

  setPending: (id: string) =>
    axiosInstance.put<AppointmentDto>(`/api/appointments/${id}/pending`).then((r) => r.data),

  cancel: (id: string, reason?: string) =>
    axiosInstance.put(`/api/appointments/${id}/cancel`, { reason }).then((r) => r.data),

  complete: (id: string, notes?: string) =>
    axiosInstance.put<AppointmentDto>(`/api/appointments/${id}/complete`, { notes }).then((r) => r.data),
}
