import { create } from 'zustand'
import type { NotificationDto, NotificationType } from '@/lib/types'
import { generateId } from '@/lib/utils'

interface NotificationState {
  notifications: NotificationDto[]
  unreadCount: number

  addNotification: (type: NotificationType, title: string, message: string) => void
  removeNotification: (id: string) => void
  markAllRead: () => void
  clearAll: () => void
}

export const useNotificationStore = create<NotificationState>((set, get) => ({
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
}))
