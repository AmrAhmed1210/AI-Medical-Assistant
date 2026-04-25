import { create } from 'zustand'
import type { NotificationDto, NotificationType } from '@/lib/types'
import { generateId } from '@/lib/utils'

export interface NotificationState {
  notifications: NotificationDto[]
  unreadCount: number

  addNotification: (type: NotificationType, title: string, message: string) => void
  removeNotification: (id: string) => void
  markAllRead: () => void
  clearAll: () => void

  unreadAppointments: number
  unreadMessages: number
  unreadDoctorApplications: number
  unreadCounts: Record<number, number>
  latestMessagePayload: any | null
  
  incrementAppointments: () => void
  clearAppointments: () => void
  
  incrementDoctorApplications: () => void
  clearDoctorApplications: () => void
  
  incrementSessionMessage: (sessionId: number) => void
  clearSessionMessages: (sessionId: number) => void
  clearAllMessages: () => void

  setLatestMessagePayload: (payload: any) => void
}

export const useNotificationStore = create<NotificationState>((set) => ({
  notifications: [],
  unreadCount: 0,

  addNotification: (type, title, message) => {
    const notification: NotificationDto = {
      id: generateId(),
      type,
      title,
      message,
      createdAt: new Date().toISOString(),
      read: false,
    }
    set((state) => ({
      notifications: [notification, ...state.notifications].slice(0, 50),
      unreadCount: state.unreadCount + 1,
    }))
  },

  removeNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
      unreadCount: state.notifications.filter((n) => n.id !== id && !n.read).length,
    })),

  markAllRead: () =>
    set((state) => ({
      notifications: state.notifications.map((n) => ({ ...n, read: true })),
      unreadCount: 0,
    })),

  clearAll: () => set({ notifications: [], unreadCount: 0 }),

  unreadAppointments: 0,
  unreadMessages: 0,
  unreadDoctorApplications: 0,
  unreadCounts: {},
  latestMessagePayload: null,
  
  incrementAppointments: () => set((state) => ({ unreadAppointments: state.unreadAppointments + 1 })),
  clearAppointments: () => set({ unreadAppointments: 0 }),
  
  incrementDoctorApplications: () => set((state) => ({ unreadDoctorApplications: state.unreadDoctorApplications + 1 })),
  clearDoctorApplications: () => set({ unreadDoctorApplications: 0 }),
  
  incrementSessionMessage: (sessionId) => set((state) => {
    const currentCount = state.unreadCounts[sessionId] || 0;
    return {
      unreadCounts: { ...state.unreadCounts, [sessionId]: currentCount + 1 },
      unreadMessages: state.unreadMessages + 1
    };
  }),
  
  clearSessionMessages: (sessionId) => set((state) => {
    const count = state.unreadCounts[sessionId] || 0;
    if (count === 0) return state;
    const newCounts = { ...state.unreadCounts };
    delete newCounts[sessionId];
    return {
      unreadCounts: newCounts,
      unreadMessages: Math.max(0, state.unreadMessages - count)
    };
  }),

  clearAllMessages: () => set({ unreadMessages: 0, unreadCounts: {} }),

  setLatestMessagePayload: (payload) => set({ latestMessagePayload: payload })
}))
