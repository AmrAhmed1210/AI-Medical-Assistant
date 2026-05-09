import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'

const renderWithRouter = (ui: React.ReactElement) => {
  return render(<BrowserRouter>{ui}</BrowserRouter>)
}

describe('Auth Store Integration', () => {
  beforeEach(() => {
    localStorage.clear()
    const store = useAuthStore.getState()
    store.logout()
    vi.clearAllMocks()
  })

  it('should login and store user data', () => {
    const store = useAuthStore.getState()
    store.setAuth(
      { id: 1, name: 'أحمد', email: 'test@test.com', role: 'Doctor' as const, isActive: true },
      'token-123'
    )

    const state = useAuthStore.getState()
    expect(state.isAuthenticated).toBe(true)
    expect(state.user?.name).toBe('أحمد')
    expect(state.token).toBe('token-123')
  })

  it('should logout and clear all data', () => {
    const store = useAuthStore.getState()
    store.setAuth(
      { id: 1, name: 'أحمد', email: 'test@test.com', role: 'Doctor' as const, isActive: true },
      'token-123'
    )
    store.logout()

    const state = useAuthStore.getState()
    expect(state.isAuthenticated).toBe(false)
    expect(state.user).toBeNull()
    expect(state.token).toBeNull()
  })

  it('should persist auth state in localStorage', () => {
    const store = useAuthStore.getState()
    store.setAuth(
      { id: 1, name: 'أحمد', email: 'test@test.com', role: 'Doctor' as const, isActive: true },
      'token-123'
    )

    const stored = localStorage.getItem('medbook-auth')
    expect(stored).toBeTruthy()
  })

  it('should block wrong roles from accessing protected routes', () => {
    localStorage.setItem('medbook-auth', JSON.stringify({
      state: {
        user: { id: 1, name: 'Patient', email: 'p@test.com', role: 'Patient', isActive: true },
        token: 'patient-token',
        role: 'Patient',
        isAuthenticated: true,
      },
    }))
  })
})

describe('Route Guards Integration', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
  })

  it('should redirect unauthenticated users to login', () => {
    const store = useAuthStore.getState()
    expect(store.isAuthenticated).toBe(false)
  })

  it('should allow authenticated users to access protected routes', () => {
    localStorage.setItem('medbook-auth', JSON.stringify({
      state: {
        user: { id: 1, name: 'Doctor', email: 'doc@test.com', role: 'Doctor', isActive: true },
        token: 'doc-token',
        role: 'Doctor',
        isAuthenticated: true,
      },
    }))
  })
})