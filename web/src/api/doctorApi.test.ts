import { describe, it, expect, vi, beforeEach } from 'vitest'
import { doctorApi } from '@/api/doctorApi'

vi.mock('@/api/axiosInstance', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}))

import axiosInstance from '@/api/axiosInstance'

describe('doctorApi', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('getAllDoctors', () => {
    it('calls get without params when no specialtyId provided', async () => {
      const mockDoctors = [
        { id: '1', fullName: 'د. أحمد', specialty: 'طب عام' },
        { id: '2', fullName: 'د. محمد', specialty: 'قلب' },
      ]
      ;(axiosInstance.get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockDoctors })

      const result = await doctorApi.getAllDoctors()

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/doctors', { params: { specialtyId: undefined } })
      expect(result).toEqual(mockDoctors)
    })

    it('calls get with specialtyId when provided', async () => {
      const mockDoctors = [{ id: '1', fullName: 'د. أحمد', specialty: 'قلب' }]
      ;(axiosInstance.get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockDoctors })

      const result = await doctorApi.getAllDoctors(1)

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/doctors', { params: { specialtyId: 1 } })
      expect(result).toEqual(mockDoctors)
    })
  })

  describe('getDoctorById', () => {
    it('calls get with correct doctor id', async () => {
      const mockDoctor = { id: '123', fullName: 'د. أحمد', specialty: 'طب عام' }
      ;(axiosInstance.get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockDoctor })

      const result = await doctorApi.getDoctorById('123')

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/doctors/123')
      expect(result).toEqual(mockDoctor)
    })
  })

  describe('getDashboard', () => {
    it('calls get to dashboard endpoint', async () => {
      const mockDashboard = {
        todayAppointments: 5,
        pendingAppointments: 3,
        totalPatients: 50,
      }
      ;(axiosInstance.get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockDashboard })

      const result = await doctorApi.getDashboard()

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/doctors/dashboard')
      expect(result).toEqual(mockDashboard)
    })
  })

  describe('getProfile', () => {
    it('calls get to profile endpoint', async () => {
      const mockProfile = { id: '1', fullName: 'د. أحمد', email: 'doc@test.com' }
      ;(axiosInstance.get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockProfile })

      const result = await doctorApi.getProfile()

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/doctors/profile')
      expect(result).toEqual(mockProfile)
    })
  })

  describe('updateProfile', () => {
    it('calls put with updated data', async () => {
      ;(axiosInstance.put as ReturnType<typeof vi.fn>).mockResolvedValue({ data: undefined })

      await doctorApi.updateProfile({ fullName: 'أحمد الجديد' })

      expect(axiosInstance.put).toHaveBeenCalledWith('/api/doctors/profile', { fullName: 'أحمد الجديد' })
    })
  })

  describe('getAppointments', () => {
    it('calls get without status filter', async () => {
      const mockAppointments = [
        { id: '1', patientName: 'مريض 1', status: 'Pending' },
        { id: '2', patientName: 'مريض 2', status: 'Confirmed' },
      ]
      ;(axiosInstance.get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockAppointments })

      const result = await doctorApi.getAppointments()

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/doctors/appointments', { params: { status: undefined } })
      expect(result).toEqual(mockAppointments)
    })

    it('calls get with status filter', async () => {
      const mockAppointments = [{ id: '1', patientName: 'مريض 1', status: 'Pending' }]
      ;(axiosInstance.get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockAppointments })

      const result = await doctorApi.getAppointments('Pending')

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/doctors/appointments', { params: { status: 'Pending' } })
      expect(result).toEqual(mockAppointments)
    })
  })

  describe('getPatients', () => {
    it('calls get without search', async () => {
      const mockPatients = [{ id: '1', fullName: 'مريض 1' }, { id: '2', fullName: 'مريض 2' }]
      ;(axiosInstance.get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockPatients })

      const result = await doctorApi.getPatients()

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/doctors/patients', { params: { search: undefined } })
      expect(result).toEqual(mockPatients)
    })

    it('calls get with search parameter', async () => {
      const mockPatients = [{ id: '1', fullName: 'أحمد' }]
      ;(axiosInstance.get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockPatients })

      const result = await doctorApi.getPatients('أحمد')

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/doctors/patients', { params: { search: 'أحمد' } })
      expect(result).toEqual(mockPatients)
    })
  })

  describe('getAvailability', () => {
    it('calls get to availability endpoint', async () => {
      const mockAvailability = [
        { dayOfWeek: 0, startTime: '09:00', endTime: '17:00', isAvailable: true },
        { dayOfWeek: 1, startTime: '09:00', endTime: '17:00', isAvailable: false },
      ]
      ;(axiosInstance.get as ReturnType<typeof vi.fn>).mockResolvedValue({ data: mockAvailability })

      const result = await doctorApi.getAvailability()

      expect(axiosInstance.get).toHaveBeenCalledWith('/api/doctors/availability')
      expect(result).toEqual(mockAvailability)
    })
  })

  describe('updateAvailability', () => {
    it('calls put with availability data', async () => {
      ;(axiosInstance.put as ReturnType<typeof vi.fn>).mockResolvedValue({ data: undefined })

      const availabilityData = [
        { dayOfWeek: 0, startTime: '09:00', endTime: '17:00', isAvailable: true },
      ]
      await doctorApi.updateAvailability(availabilityData)

      expect(axiosInstance.put).toHaveBeenCalledWith('/api/doctors/availability', availabilityData)
    })
  })

  describe('apply', () => {
    it('calls post to apply endpoint', async () => {
      ;(axiosInstance.post as ReturnType<typeof vi.fn>).mockResolvedValue({ data: { success: true } })

      const applyData = {
        fullName: 'د. أحمد',
        email: 'doc@test.com',
        specialty: 'طب عام',
        yearsExperience: 5,
      }
      await doctorApi.apply(applyData)

      expect(axiosInstance.post).toHaveBeenCalledWith('/api/doctors/apply', applyData)
    })
  })
})