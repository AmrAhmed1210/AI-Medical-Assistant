import { useNotificationStore } from '@/store/notificationStore'
import type { NotificationType } from '@/lib/types'

export function useNotifications() {
  const store = useNotificationStore()

  const notify = (type: NotificationType, title: string, message = '') => {
    store.addNotification(type, title, message)
  }

  return {
    ...store,
    notify,
    success: (title: string, msg?: string) => notify('success', title, msg),
    error: (title: string, msg?: string) => notify('error', title, msg),
    warning: (title: string, msg?: string) => notify('warning', title, msg),
    info: (title: string, msg?: string) => notify('info', title, msg),
  }
}
