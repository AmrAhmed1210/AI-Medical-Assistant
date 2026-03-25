import axiosInstance from './axiosInstance'
import type { SessionDto, SessionDetailDto } from '@/lib/types'

export const consultApi = {
  getSessions: () =>
    axiosInstance.get<SessionDto[]>('/api/sessions').then((r) => r.data),

  getSession: (id: string) =>
    axiosInstance.get<SessionDetailDto>(`/api/sessions/${id}`).then((r) => r.data),

  deleteSession: (id: string) =>
    axiosInstance.delete(`/api/sessions/${id}`).then((r) => r.data),
}
