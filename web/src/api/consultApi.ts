import axiosInstance from './axiosInstance'
import type { MessageDto, SessionDto, SessionDetailDto } from '@/lib/types'

export interface ConsultationPayload {
  patientId: number
  title: string
  description: string
  scheduledAt: string
}

export interface ConsultationDto {
  id: number
  doctorId: number
  doctorName: string
  patientId: number
  patientName: string
  title: string
  description: string
  scheduledAt: string
  status: string
  createdAt: string
}

export const consultApi = {
  getSessions: () =>
    axiosInstance.get<SessionDto[]>('/api/sessions').then((r) => r.data),

  getSession: (id: string) =>
    axiosInstance.get<SessionDetailDto>(`/api/sessions/${id}`).then((r) => r.data),

  sendMessage: (id: string | number, content: string, type: string = 'text', attachmentUrl?: string | null, fileName?: string | null) =>
    axiosInstance.post<MessageDto>(`/api/sessions/${id}/messages`, { 
      content, 
      messageType: type,
      attachmentUrl,
      fileName
    }).then((r) => r.data),

  deleteSession: (id: string) =>
    axiosInstance.delete(`/api/sessions/${id}`).then((r) => r.data),

  getSupportSessions: () =>
    axiosInstance.get<SessionDto[]>('/api/admin/support-sessions').then((r) => r.data),

  uploadFile: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return axiosInstance.post<{ url: string; fileName: string }>('/api/profile/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data)
  },

  // Consultation endpoints
  sendConsultation: (data: ConsultationPayload) =>
    axiosInstance.post<ConsultationDto>('/api/consultations', data).then((r) => r.data),

  getDoctorConsultations: () =>
    axiosInstance.get<ConsultationDto[]>('/api/consultations/doctor').then((r) => r.data),
}
