import { describe, it, expect, vi, beforeEach } from 'vitest'
import { authApi } from '@/api/authApi'

vi.mock('@/api/axiosInstance', () => ({
  default: {
    post: vi.fn(),
    get: vi.fn(),
  },
}))

import axiosInstance from '@/api/axiosInstance'

describe('authApi', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('login', () => {
    it('calls post with correct endpoint and returns data', async () => {
      const mockResponse = {
        data: {
          accessToken: 'token123',
          refreshToken: 'refresh123',
          expiresIn: 3600,
          user: { id: '1', fullName: 'أحمد', email: 'test@test.com', role: 'Doctor' },
        },
      }
      ;(axiosInstance.post as ReturnType<typeof vi.fn>).mockResolvedValue(mockResponse)

      const result = await authApi.login({ email: 'test@test.com', password: '123456' })

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/auth/login', {
        email: 'test@test.com',
        password: '123456',
      })
      expect(result).toEqual(mockResponse.data)
    })

    it('handles login error', async () => {
      ;(axiosInstance.post as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('Invalid credentials'))

      await expect(
        authApi.login({ email: 'wrong@test.com', password: 'wrong' })
      ).rejects.toThrow('Invalid credentials')
    })
  })

  describe('logout', () => {
    it('calls post to logout endpoint', async () => {
      const mockResponse = { data: { success: true } }
      ;(axiosInstance.post as ReturnType<typeof vi.fn>).mockResolvedValue(mockResponse)

      await authApi.logout()

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/auth/logout')
    })
  })

  describe('me', () => {
    it('calls get to /auth/me endpoint', async () => {
      const mockUser = { id: '1', fullName: 'أحمد', email: 'test@test.com', role: 'Doctor' }
      ;(axiosInstance.get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockUser })

      const result = await authApi.me()

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/auth/me')
      expect(result).toEqual(mockUser)
    })
  })
})