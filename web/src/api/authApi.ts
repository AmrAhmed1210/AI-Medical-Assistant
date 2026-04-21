import axiosInstance from './axiosInstance'
import type { LoginResponse, LoginRequest } from '@/lib/types'

export const authApi = {
  login: (data: LoginRequest) =>
    axiosInstance.post<LoginResponse>('/api/auth/login', data).then((r) => r.data),

  logout: () =>
    axiosInstance.post('/api/auth/logout').then((r) => r.data),

  me: () =>
    axiosInstance.get('/api/auth/me').then((r) => r.data),
}
