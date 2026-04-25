import { create } from 'zustand'

interface DoctorAlertState {
  unreadMessages: number
  unreadAppointments: number
  incrementMessages: () => void
  incrementAppointments: () => void
  clearMessages: () => void
  clearAppointments: () => void
}

export const useDoctorAlertStore = create<DoctorAlertState>((set) => ({
  unreadMessages: 0,
  unreadAppointments: 0,
  incrementMessages: () => set((state) => ({ unreadMessages: state.unreadMessages + 1 })),
  incrementAppointments: () => set((state) => ({ unreadAppointments: state.unreadAppointments + 1 })),
  clearMessages: () => set({ unreadMessages: 0 }),
  clearAppointments: () => set({ unreadAppointments: 0 }),
}))
