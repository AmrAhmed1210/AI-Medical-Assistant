import { create } from 'zustand'
import type {
  DoctorDashboardDto,
  DoctorDetailDto,
  AppointmentDto,
  PatientSummaryDto,
  AIReportDto,
  AvailabilityDto,
} from '@/lib/types'
import { doctorApi } from '@/api/doctorApi'

const defaultDashboard: DoctorDashboardDto = {
  todayAppointments: 0,
  pendingAppointments: 0,
  totalPatients: 0,
  unreadReports: 0,
  weekAppointments: 0,
  upcomingAppointments: [],
  todayAppointmentsList: [],
  weeklySessionsChart: [],
  recentReports: [],
}

interface DoctorState {
  dashboard: DoctorDashboardDto | null
  profile: DoctorDetailDto | null
  appointments: AppointmentDto[]
  patients: PatientSummaryDto[]
  reports: AIReportDto[]
  availability: AvailabilityDto[]

  isLoadingDashboard: boolean
  isLoadingProfile: boolean
  isLoadingAppointments: boolean
  isLoadingPatients: boolean
  isLoadingReports: boolean

  fetchDashboard: () => Promise<void>
  fetchProfile: () => Promise<void>
  updateProfile: (data: Partial<DoctorDetailDto>) => Promise<void>
  fetchAppointments: (status?: string) => Promise<void>
  fetchPatients: (search?: string) => Promise<void>
  fetchReports: (params?: object) => Promise<void>
  fetchAvailability: () => Promise<void>
  updateAvailability: (data: AvailabilityDto[]) => Promise<void>
  updateScheduleVisibility: (isVisible: boolean) => Promise<void>
}

export const useDoctorStore = create<DoctorState>((set) => ({
  dashboard: null,
  profile: null,
  appointments: [],
  patients: [],
  reports: [],
  availability: [],
  isLoadingDashboard: false,
  isLoadingProfile: false,
  isLoadingAppointments: false,
  isLoadingPatients: false,
  isLoadingReports: false,

  fetchDashboard: async () => {
    set({ isLoadingDashboard: true })
    try {
      const data = await doctorApi.getDashboard()
      set({ dashboard: { ...defaultDashboard, ...data } })
    } catch (error) {
      console.error('Failed to fetch dashboard:', error)
      set({ dashboard: defaultDashboard })
    } finally {
      set({ isLoadingDashboard: false })
    }
  },

  fetchProfile: async () => {
    set({ isLoadingProfile: true })
    try {
      const data = await doctorApi.getProfile()
      set({ profile: data })
    } catch (error) {
      console.error('Failed to fetch profile:', error)
      set({ profile: null })
    } finally {
      set({ isLoadingProfile: false })
    }
  },

  updateProfile: async (data) => {
    try {
      await doctorApi.updateProfile(data)
      const refreshed = await doctorApi.getProfile()
      set({ profile: refreshed })
    } catch (error) {
      console.error('Failed to update profile:', error)
    }
  },

  fetchAppointments: async (status) => {
    set({ isLoadingAppointments: true })
    try {
      const data = await doctorApi.getAppointments(status)
      set({ appointments: data })
    } catch (error) {
      console.error('Failed to fetch appointments:', error)
      set({ appointments: [] })
    } finally {
      set({ isLoadingAppointments: false })
    }
  },

  fetchPatients: async (search) => {
    set({ isLoadingPatients: true })
    try {
      const data = await doctorApi.getPatients(search)
      set({ patients: data })
    } catch (error) {
      console.error('Failed to fetch patients:', error)
      set({ patients: [] })
    } finally {
      set({ isLoadingPatients: false })
    }
  },

  fetchReports: async (params) => {
    set({ isLoadingReports: true })
    try {
      const data = await doctorApi.getReports(params)
      set({ reports: data })
    } catch (error) {
      console.error('Failed to fetch reports:', error)
      set({ reports: [] })
    } finally {
      set({ isLoadingReports: false })
    }
  },

  fetchAvailability: async () => {
    try {
      const data = await doctorApi.getAvailability()
      set({ availability: data })
    } catch (error) {
      console.error('Failed to fetch availability:', error)
      set({ availability: [] })
    }
  },

  updateAvailability: async (data) => {
    try {
      await doctorApi.updateAvailability(data)
      set({ availability: data })
    } catch (error) {
      console.error('Failed to update availability:', error)
    }
  },

  updateScheduleVisibility: async (isVisible) => {
    try {
      await doctorApi.updateScheduleVisibility(isVisible)
      set((state) => ({ 
        profile: state.profile ? { ...state.profile, isScheduleVisible: isVisible } : null 
      }))
    } catch (error) {
      console.error('Failed to update schedule visibility:', error)
    }
  },
}))
