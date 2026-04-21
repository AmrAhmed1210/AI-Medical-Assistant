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

  deleteUser: (id: number) =>
    axiosInstance.delete(`/api/admin/users/${id}`).then((r) => r.data),

  getModels: () =>
    axiosInstance.get<ModelVersionDto[]>('/api/admin/models').then((r) => r.data),

  reloadModel: (agentName: string) =>
    axiosInstance.post('/api/admin/reload-model', { agentName }).then((r) => r.data),
}
