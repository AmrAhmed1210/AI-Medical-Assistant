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
      set({ dashboard: data })
    } finally {
      set({ isLoadingDashboard: false })
    }
  },

  fetchProfile: async () => {
    set({ isLoadingProfile: true })
    try {
      const data = await doctorApi.getProfile()
      set({ profile: data })
    } finally {
      set({ isLoadingProfile: false })
    }
  },

  updateProfile: async (data) => {
    const updated = await doctorApi.updateProfile(data)
    set({ profile: updated })
  },

  fetchAppointments: async (status) => {
    set({ isLoadingAppointments: true })
    try {
      const data = await doctorApi.getAppointments(status)
      set({ appointments: data })
    } finally {
      set({ isLoadingAppointments: false })
    }
  },

  fetchPatients: async (search) => {
    set({ isLoadingPatients: true })
    try {
      const data = await doctorApi.getPatients(search)
      set({ patients: data })
    } finally {
      set({ isLoadingPatients: false })
    }
  },

  fetchReports: async (params) => {
    set({ isLoadingReports: true })
    try {
      const data = await doctorApi.getReports(params)
      set({ reports: data })
    } finally {
      set({ isLoadingReports: false })
    }
  },

  fetchAvailability: async () => {
    const data = await doctorApi.getAvailability()
    set({ availability: data })
  },

  updateAvailability: async (data) => {
    await doctorApi.updateAvailability(data)
    set({ availability: data })
  },
}))
