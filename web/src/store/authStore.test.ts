import { describe, it, expect, beforeEach } from 'vitest'
import { useAuthStore } from '@/store/authStore'
import type { UserDto, UserRole } from '@/lib/types'

const mockUser: UserDto = {
  id: 1,
  email: 'test@test.com',
  name: 'Ahmed Mohamed',
  role: 'Doctor' as UserRole,
  isActive: true,
  createdAt: '2024-01-01',
}

describe('authStore', () => {
  beforeEach(() => {
    const store = useAuthStore.getState()
    store.logout()
  })

  it('initial state should be empty', () => {
    const state = useAuthStore.getState()
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
    expect(state.isAuthenticated).toBe(false)
  })

  it('setAuth should update state correctly', () => {
    const store = useAuthStore.getState()
    store.setAuth(mockUser, 'token-123')

    const state = useAuthStore.getState()
    expect(state.user).toEqual(mockUser)
    expect(state.token).toBe('token-123')
    expect(state.role).toBe('Doctor')
    expect(state.isAuthenticated).toBe(true)
  })

  it('logout should clear all state', () => {
    const store = useAuthStore.getState()
    store.setAuth(mockUser, 'token-123')
    store.logout()

    const state = useAuthStore.getState()
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
    expect(state.role).toBeNull()
    expect(state.isAuthenticated).toBe(false)
  })

  it('setLoading should update loading state', () => {
    const store = useAuthStore.getState()
    store.setLoading(true)

    const state = useAuthStore.getState()
    expect(state.isLoading).toBe(true)
  })

  it('updateUser should partially update user', () => {
    const store = useAuthStore.getState()
    store.setAuth(mockUser, 'token-123')
    store.updateUser({ name: 'Ahmed Ali' })

    const state = useAuthStore.getState()
    expect(state.user?.name).toBe('Ahmed Ali')
    expect(state.user?.email).toBe(mockUser.email)
  })
})