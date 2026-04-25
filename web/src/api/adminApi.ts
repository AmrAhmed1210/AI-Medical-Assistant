import axiosInstance from './axiosInstance'
import type { SystemStatsDto, UserDto, ModelVersionDto, CreateUserRequest } from '@/lib/types'

export const adminApi = {
  getStats: () =>
    axiosInstance.get<SystemStatsDto>('/api/admin/stats').then((r) => r.data),

  getUsers: (params?: { role?: string; page?: number; pageSize?: number; search?: string }) =>
    axiosInstance.get<{ items: UserDto[]; total: number }>('/api/admin/users', { params }).then((r) => r.data),

  toggleUser: (id: number) =>
    axiosInstance.put(`/api/admin/users/${id}/toggle`).then((r) => r.data),

  createUser: (data: CreateUserRequest) =>
    axiosInstance.post<UserDto>('/api/admin/users', data).then((r) => r.data),

  deleteUser: (id: number, role: string) =>
    axiosInstance.delete(`/api/admin/users/${id}`, { params: { role } }).then((r) => r.data),

  getModels: () =>
    axiosInstance.get<ModelVersionDto[]>('/api/admin/models').then((r) => r.data),

  reloadModel: (agentName: string) =>
    axiosInstance.post('/api/admin/reload-model', { agentName }).then((r) => r.data),

  getApplications: (status?: string) =>
    axiosInstance.get<any[]>('/api/admin/applications', { params: { status } }).then((r) => r.data),

  approveApplication: (id: number) =>
    axiosInstance.post(`/api/admin/applications/${id}/approve`).then((r) => r.data),

  rejectApplication: (id: number, reason?: string) =>
    axiosInstance.post(`/api/admin/applications/${id}/reject`, { reason }).then((r) => r.data),
}
