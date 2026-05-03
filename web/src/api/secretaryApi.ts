import axiosInstance from './axiosInstance'
import type { SecretaryDto, CreateSecretaryRequest } from '@/lib/types'

export const secretaryApi = {
  addSecretary: (data: CreateSecretaryRequest) =>
    axiosInstance.post<SecretaryDto>('/api/Secretary/add', data).then((r) => r.data),

  getMySecretaries: () =>
    axiosInstance.get<SecretaryDto[]>('/api/Secretary/my-secretaries').then((r) => r.data),

  deleteSecretary: (id: number) =>
    axiosInstance.delete(`/api/Secretary/${id}`).then((r) => r.data),
}
