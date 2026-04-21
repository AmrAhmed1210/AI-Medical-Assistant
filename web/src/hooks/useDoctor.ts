import { useEffect, useMemo, useState } from 'react'
import { doctorApi } from '@/api/doctorApi'
import type {
  AIReportDto,
  AppointmentDto,
  DoctorDashboardDto,
  DoctorDetailDto,
  PatientSummaryDto,
} from '@/lib/types'

const emptyDashboard: DoctorDashboardDto = {
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

export function useDoctorDashboard() {
  const [dashboard, setDashboard] = useState<DoctorDashboardDto | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [refreshKey, setRefreshKey] = useState(0)

  const refresh = () => setRefreshKey(prev => prev + 1)

  useEffect(() => {
    let cancelled = false
    setIsLoading(true)

    doctorApi.getDashboard()
      .then((data) => {
        if (!cancelled) setDashboard({ ...emptyDashboard, ...data })
      })
      .catch(() => {
        if (!cancelled) setDashboard(emptyDashboard)
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })

    return () => { cancelled = true }
  }, [refreshKey])

  return { dashboard, isLoading, refresh }
}

export function useDoctorProfile() {
  const [profile, setProfile] = useState<DoctorDetailDto | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setIsLoading(true)

    doctorApi.getProfile()
      .then((data) => {
        if (!cancelled) setProfile(data)
      })
      .catch(() => {
        if (!cancelled) setProfile(null)
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })

    return () => { cancelled = true }
  }, [])

  const updateProfile = async (data: Partial<DoctorDetailDto>) => {
    await doctorApi.updateProfile(data)
    try {
      const refreshed = await doctorApi.getProfile()
      setProfile(refreshed)
    } catch {
      setProfile(null)
    }
  }

  return { profile, isLoading, updateProfile }
}

export function useDoctorAppointments(status?: string) {
  const [appointments, setAppointments] = useState<AppointmentDto[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setIsLoading(true)

    doctorApi.getAppointments(status)
      .then((data) => {
        if (!cancelled) setAppointments(data)
      })
      .catch(() => {
        if (!cancelled) setAppointments([])
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })

    return () => { cancelled = true }
  }, [status])

  return { appointments, isLoading }
}

export function useDoctorPatients(search?: string) {
  const [patients, setPatients] = useState<PatientSummaryDto[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setIsLoading(true)

    doctorApi.getPatients(search)
      .then((data) => {
        if (!cancelled) setPatients(data)
      })
      .catch(() => {
        if (!cancelled) setPatients([])
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })

    return () => { cancelled = true }
  }, [search])

  return { patients, isLoading }
}

export function useDoctorReports(params?: Record<string, unknown>) {
  const [reports, setReports] = useState<AIReportDto[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const paramsKey = useMemo(() => JSON.stringify(params ?? {}), [params])

  useEffect(() => {
    let cancelled = false
    setIsLoading(true)

    doctorApi.getReports(params as { urgency?: string; patientId?: string; from?: string; to?: string } | undefined)
      .then((data) => {
        if (!cancelled) setReports(data)
      })
      .catch(() => {
        if (!cancelled) setReports([])
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })

    return () => { cancelled = true }
  }, [paramsKey])

  return { reports, isLoading }
}
