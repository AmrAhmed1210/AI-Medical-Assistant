import { useEffect } from 'react'
import { useDoctorStore } from '@/store/doctorStore'

export function useDoctorDashboard() {
  const { dashboard, isLoadingDashboard, fetchDashboard } = useDoctorStore()
  useEffect(() => { fetchDashboard() }, [fetchDashboard])
  return { dashboard, isLoading: isLoadingDashboard, refetch: fetchDashboard }
}

export function useDoctorProfile() {
  const { profile, isLoadingProfile, fetchProfile, updateProfile } = useDoctorStore()
  useEffect(() => { fetchProfile() }, [fetchProfile])
  return { profile, isLoading: isLoadingProfile, updateProfile, refetch: fetchProfile }
}

export function useDoctorAppointments(status?: string) {
  const { appointments, isLoadingAppointments, fetchAppointments } = useDoctorStore()
  useEffect(() => { fetchAppointments(status) }, [fetchAppointments, status])
  return { appointments, isLoading: isLoadingAppointments, refetch: () => fetchAppointments(status) }
}

export function useDoctorPatients(search?: string) {
  const { patients, isLoadingPatients, fetchPatients } = useDoctorStore()
  useEffect(() => { fetchPatients(search) }, [fetchPatients, search])
  return { patients, isLoading: isLoadingPatients, refetch: () => fetchPatients(search) }
}

export function useDoctorReports(params?: object) {
  const { reports, isLoadingReports, fetchReports } = useDoctorStore()
  useEffect(() => { fetchReports(params) }, [fetchReports])
  return { reports, isLoading: isLoadingReports, refetch: () => fetchReports(params) }
}
