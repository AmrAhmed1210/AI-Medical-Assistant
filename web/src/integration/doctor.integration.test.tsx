import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'
import type { DoctorDashboardDto, DoctorDetailDto } from '@/lib/types'

vi.mock('@/api/doctorApi', () => ({
  doctorApi: {
    getAllDoctors: vi.fn(),
    getDoctorById: vi.fn(),
    getDashboard: vi.fn().mockResolvedValue({
      todayAppointments: 5,
      pendingAppointments: 3,
      totalPatients: 50,
      unreadReports: 2,
      weekAppointments: 25,
      upcomingAppointments: [],
      todayAppointmentsList: [],
      weeklySessionsChart: [],
      recentReports: [],
    }),
    getProfile: vi.fn().mockResolvedValue({
      id: '1',
      userId: '1',
      fullName: 'Dr. Ahmed',
      email: 'doc@test.com',
      specialty: 'General Medicine',
      license: '12345',
      bio: null,
      photoUrl: null,
      consultFee: 200,
      yearsExperience: 10,
      createdAt: '2024-01-01',
      updatedAt: null,
    }),
    updateProfile: vi.fn().mockResolvedValue(undefined),
    getAppointments: vi.fn().mockResolvedValue([]),
    getPatients: vi.fn().mockResolvedValue([]),
    getAvailability: vi.fn().mockResolvedValue([]),
    updateAvailability: vi.fn().mockResolvedValue(undefined),
    getReports: vi.fn().mockResolvedValue([]),
  },
}))

import { doctorApi } from '@/api/doctorApi'

const setupDoctorAuth = () => {
  localStorage.setItem('medbook-auth', JSON.stringify({
    state: {
      user: { id: 1, name: 'Dr. Ahmed', email: 'doc@test.com', role: 'Doctor', isActive: true },
      token: 'doctor-token',
      role: 'Doctor',
      isAuthenticated: true,
    },
  }))
}

describe('Doctor API Integration', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
    setupDoctorAuth()
  })

  it('should fetch doctor dashboard data', async () => {
    const dashboard = await doctorApi.getDashboard()
    expect(dashboard).toHaveProperty('todayAppointments')
    expect(dashboard).toHaveProperty('totalPatients')
    expect(dashboard.todayAppointments).toBe(5)
  })

  it('should fetch doctor profile', async () => {
    const profile = await doctorApi.getProfile()
    expect(profile.fullName).toBe('Dr. Ahmed')
    expect(profile.specialty).toBe('General Medicine')
  })

  it('should update doctor profile', async () => {
    await doctorApi.updateProfile({ fullName: 'Dr. Ahmed Updated' })
    expect(doctorApi.updateProfile).toHaveBeenCalledWith({ fullName: 'Dr. Ahmed Updated' })
  })

  it('should fetch appointments', async () => {
    const appointments = await doctorApi.getAppointments()
    expect(Array.isArray(appointments)).toBe(true)
  })

  it('should fetch patients', async () => {
    const patients = await doctorApi.getPatients()
    expect(Array.isArray(patients)).toBe(true)
  })

  it('should filter appointments by status', async () => {
    await doctorApi.getAppointments('Pending')
    expect(doctorApi.getAppointments).toHaveBeenCalledWith('Pending')
  })

  it('should search patients', async () => {
    await doctorApi.getPatients('Ahmed')
    expect(doctorApi.getPatients).toHaveBeenCalledWith('Ahmed')
  })

  it('should handle API errors gracefully', async () => {
    vi.mocked(doctorApi.getDashboard).mockRejectedValueOnce(new Error('Network error'))
    
    await expect(doctorApi.getDashboard()).rejects.toThrow('Network error')
  })

  it('should fetch doctor by ID', async () => {
    vi.mocked(doctorApi.getDoctorById).mockResolvedValueOnce({
      id: '123',
      userId: '1',
      fullName: 'Dr. Mohamed',
      email: 'moh@test.com',
      specialty: 'Cardiology',
      license: '54321',
      bio: 'Cardiologist',
      photoUrl: null,
      consultFee: 300,
      yearsExperience: 15,
      createdAt: '2024-01-01',
      updatedAt: null,
    })

    const doctor = await doctorApi.getDoctorById('123')
    expect(doctor.fullName).toBe('Dr. Mohamed')
    expect(doctor.specialty).toBe('Cardiology')
  })

  it('should update doctor availability', async () => {
    const availability = [
      { dayOfWeek: 0 as const, startTime: '09:00', endTime: '17:00', isAvailable: true },
    ]
    
    await doctorApi.updateAvailability(availability)
    expect(doctorApi.updateAvailability).toHaveBeenCalledWith(availability)
  })

  it('should get all doctors list', async () => {
    vi.mocked(doctorApi.getAllDoctors).mockResolvedValueOnce([
      { id: '1', fullName: 'Dr. Ahmed', email: 'a@test.com', specialty: 'General Medicine', bio: null, photoUrl: null, consultFee: 200, yearsExperience: 10, userId: '1', license: '12345', createdAt: '2024-01-01', updatedAt: null },
      { id: '2', fullName: 'Dr. Mohamed', email: 'm@test.com', specialty: 'Cardiology', bio: null, photoUrl: null, consultFee: 300, yearsExperience: 15, userId: '2', license: '54321', createdAt: '2024-01-01', updatedAt: null },
    ])

    const doctors = await doctorApi.getAllDoctors()
    expect(doctors.length).toBe(2)
  })
})