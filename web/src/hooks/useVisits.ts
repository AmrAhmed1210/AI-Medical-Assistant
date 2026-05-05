import { useEffect, useState } from 'react'
import { visitApi, type VisitDto, type VisitSummaryDto } from '@/api/visitApi'
import { doctorApi } from '@/api/doctorApi'
import type { AppointmentDto } from '@/lib/types'

export interface PatientHistoryDto {
  bloodType?: string
  allergies?: { allergenName: string; severity: string; reaction: string }[]
  chronicDiseases?: { id: string; diseaseName: string; targetValues: string }[]
  medications?: { id: string; medicationName: string; dosage: string; frequency: string }[]
  latestVitals?: Record<string, string>
  lastVisits?: { id: string; visitDate: string; chiefComplaint: string }[]
}

export function useTodayVisits() {
  const [appointments, setAppointments] = useState<AppointmentDto[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [refreshKey, setRefreshKey] = useState(0)

  const refresh = () => setRefreshKey((prev) => prev + 1)

  useEffect(() => {
    let cancelled = false
    setIsLoading(true)

    doctorApi.getAppointments('confirmed')
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
  }, [refreshKey])

  return { appointments, isLoading, refresh }
}

export function useVisit(id: number) {
  const [visit, setVisit] = useState<VisitDto | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (!id) {
      setIsLoading(false)
      return
    }
    let cancelled = false
    setIsLoading(true)

    visitApi.getVisit(id)
      .then((data) => {
        if (!cancelled) setVisit(data)
      })
      .catch(() => {
        if (!cancelled) setVisit(null)
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })

    return () => { cancelled = true }
  }, [id])

  return { visit, isLoading }
}

export function useVisitSummary(id: number) {
  const [summary, setSummary] = useState<VisitSummaryDto | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (!id) {
      setIsLoading(false)
      return
    }
    let cancelled = false
    setIsLoading(true)

    visitApi.getSummary(id)
      .then((data) => {
        if (!cancelled) setSummary(data)
      })
      .catch(() => {
        if (!cancelled) setSummary(null)
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })

    return () => { cancelled = true }
  }, [id])

  return { summary, isLoading }
}

export function usePatientHistory(patientId: number) {
  const [history, setHistory] = useState<PatientHistoryDto | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (!patientId) {
      setIsLoading(false)
      return
    }
    let cancelled = false
    setIsLoading(true)

    visitApi.getPatientHistory(patientId)
      .then((data) => {
        if (!cancelled) setHistory(data)
      })
      .catch(() => {
        if (!cancelled) setHistory(null)
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })

    return () => { cancelled = true }
  }, [patientId])

  return { history, isLoading }
}
