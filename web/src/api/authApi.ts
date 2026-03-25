import axiosInstance from './axiosInstance'
import type { AuthResponseDto, LoginRequest, UserProfileDto } from '@/lib/types'

export const authApi = {
  login: (data: LoginRequest) =>
    axiosInstance.post<AuthResponseDto>('/api/auth/login', data).then((r) => r.data),

  logout: () =>
    axiosInstance.post('/api/auth/logout').then((r) => r.data),

  me: () =>
    axiosInstance.get<UserProfileDto>('/api/auth/me').then((r) => r.data),
}
