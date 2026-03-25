import axiosInstance from './axiosInstance'
import type { AppointmentDto } from '@/lib/types'

export const appointmentApi = {
  confirm: (id: string) =>
    axiosInstance.put<AppointmentDto>(`/api/appointments/${id}/confirm`).then((r) => r.data),

  cancel: (id: string, reason?: string) =>
    axiosInstance.put(`/api/appointments/${id}/cancel`, { reason }).then((r) => r.data),

  complete: (id: string, notes?: string) =>
    axiosInstance.put<AppointmentDto>(`/api/appointments/${id}/complete`, { notes }).then((r) => r.data),
}
