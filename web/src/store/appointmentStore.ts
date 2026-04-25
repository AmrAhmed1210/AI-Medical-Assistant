import { create } from 'zustand'
import type { AppointmentDto } from '@/lib/types'
import { appointmentApi } from '@/api/appointmentApi'
import { doctorApi } from '@/api/doctorApi'

interface AppointmentState {
  appointments: AppointmentDto[]
  isLoading: boolean

  // Computed getters
  activeAppointments: () => AppointmentDto[]
  historyAppointments: () => AppointmentDto[]

  fetchAppointments: () => Promise<void>
  confirm: (id: string) => Promise<AppointmentDto>
  cancel: (id: string, reason?: string) => Promise<void>
  complete: (id: string, notes?: string) => Promise<AppointmentDto>
  updateLocal: (id: string, update: Partial<AppointmentDto>) => void
  removeFromActive: (id: string) => void
  moveToHistory: (id: string) => void
  clearHistory: () => Promise<void>
  // Real-time sync
  handleAppointmentUpdated: (payload: { appointmentId: number | string; status: string; [key: string]: any }) => void
  handleNewBooking: (payload: any) => void
}

export const useAppointmentStore = create<AppointmentState>((set, get) => ({
  appointments: [],
  isLoading: false,

  // ... (rest of getters)
  activeAppointments: () => {
    const { appointments } = get()
    return appointments.filter(a => 
      a.status === 'Pending' || a.status === 'Confirmed'
    )
  },

  historyAppointments: () => {
    const { appointments } = get()
    return appointments.filter(a => 
      a.status === 'Completed' || a.status === 'Cancelled'
    )
  },

  fetchAppointments: async () => {
    set({ isLoading: true })
    try {
      const data = await doctorApi.getAppointments()
      
      const toScheduledAt = (item: any): string => {
        const fromApi = String(item?.scheduledAt ?? item?.ScheduledAt ?? '').trim()
        if (fromApi) return fromApi
        const datePart = String(item?.date ?? item?.Date ?? '').trim()
        const timePart = String(item?.time ?? item?.Time ?? '').trim()
        if (!datePart && !timePart) return ''
        return `${datePart} ${timePart}`.replace(/\s+/g, ' ').trim()
      }

      const normalizeStatus = (status: string) => {
        const lowered = (status || '').toLowerCase()
        if (lowered === 'confirmed') return 'Confirmed'
        if (lowered === 'cancelled' || lowered === 'canceled') return 'Cancelled'
        if (lowered === 'completed') return 'Completed'
        return 'Pending'
      }

      const normalized = (data as any[]).map((item) => ({
        ...item,
        id: String(item.id),
        patientName: item.patientName ?? 'Unknown',
        doctorName: item.doctorName ?? 'Doctor',
        paymentMethod: item.paymentMethod ?? item.PaymentMethod ?? '',
        status: normalizeStatus(item.status),
        scheduledAt: toScheduledAt(item),
      })).sort((a, b) => new Date(a.scheduledAt).getTime() - new Date(b.scheduledAt).getTime())

      set({ appointments: normalized })
    } finally {
      set({ isLoading: false })
    }
  },

  confirm: async (id) => {
    const updated = await appointmentApi.confirm(id)
    get().updateLocal(id, { status: 'Confirmed' })
    return updated
  },

  cancel: async (id, reason) => {
    await appointmentApi.cancel(id, reason)
    get().updateLocal(id, { status: 'Cancelled' })
  },

  complete: async (id, notes) => {
    const updated = await appointmentApi.complete(id, notes)
    get().updateLocal(id, { status: 'Completed' })
    return updated
  },

  updateLocal: (id, update) =>
    set((state) => ({
      appointments: state.appointments.map((a) =>
        (a.id === id) ? { ...a, ...update } : a
      ),
    })),

  removeFromActive: (id: string) => {
    get().updateLocal(id, { status: 'Cancelled' })
  },

  moveToHistory: (id: string) => {
    get().updateLocal(id, { status: 'Completed' })
  },

  clearHistory: async () => {
    await doctorApi.clearHistory()
    const today = new Date()
    today.setHours(0, 0, 0, 0)
    set((state) => ({
      appointments: state.appointments.filter((a) => {
        const date = new Date(a.scheduledAt)
        return isNaN(date.getTime()) ? true : date >= today
      })
    }))
  },

  handleAppointmentUpdated: (payload) => {
    const { appointmentId, status } = payload
    const id = String(appointmentId)
    const current = get().appointments.find(a => a.id === id)
    if (!current) return

    const normalizedStatus = (() => {
      const s = (status || '').toLowerCase()
      if (s === 'confirmed') return 'Confirmed'
      if (s === 'cancelled' || s === 'canceled') return 'Cancelled'
      if (s === 'completed') return 'Completed'
      if (s === 'pending') return 'Pending'
      return current.status
    })()

    get().updateLocal(id, { status: normalizedStatus })
  },

  handleNewBooking: (payload) => {
    const id = String(payload?.appointmentId ?? payload?.Id ?? Math.random())
    const scheduledAt = payload?.scheduledAt ?? payload?.ScheduledAt ?? `${payload?.date} ${payload?.time}`
    
    const newAppt: AppointmentDto = {
      id,
      patientName: payload?.patientName ?? payload?.PatientName ?? 'New Patient',
      doctorName: payload?.doctorName ?? 'Doctor',
      status: 'Confirmed', // Usually bookings from mobile are confirmed or pending. SignalR often sends confirmed.
      scheduledAt,
      paymentMethod: payload?.paymentMethod ?? 'Cash',
      notes: ''
    }

    set((state) => {
      if (state.appointments.some(a => a.id === id)) return state
      return { 
        appointments: [...state.appointments, newAppt].sort((a, b) => 
          new Date(a.scheduledAt).getTime() - new Date(b.scheduledAt).getTime()
        )
      }
    })
  }
}))