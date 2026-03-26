import { create } from 'zustand'
import type { AppointmentDto } from '@/lib/types'
import { appointmentApi } from '@/api/appointmentApi'

interface AppointmentState {
  appointments: AppointmentDto[]
  isLoading: boolean

  setAppointments: (appointments: AppointmentDto[]) => void
  confirm: (id: string) => Promise<AppointmentDto>
  cancel: (id: string, reason?: string) => Promise<void>
  complete: (id: string, notes?: string) => Promise<AppointmentDto>
  updateLocal: (id: string, update: Partial<AppointmentDto>) => void
}

export const useAppointmentStore = create<AppointmentState>((set, get) => ({
  appointments: [],
  isLoading: false,

  setAppointments: (appointments) => set({ appointments }),

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

  // ✅ بيدور على id أو appointmentId عشان يدعم الاتنين
  updateLocal: (id, update) =>
    set((state) => ({
      appointments: state.appointments.map((a) =>
        (a.id === id || a.id === id) ? { ...a, ...update } : a
      ),
    })),
}))