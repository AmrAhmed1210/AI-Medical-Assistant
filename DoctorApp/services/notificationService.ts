import AsyncStorage from '@react-native-async-storage/async-storage'

export interface Notification {
  id: string
  type:
    | 'schedule'
    | 'update'
    | 'confirmed'
    | 'message'
    | 'schedule_ready'
    | 'appointment_confirmed'
    | 'appointment_cancelled'
    | 'appointment_updated'
    | 'appointment_reminder'
    | 'missed_appointment'
    | 'rebook_offer'
    | 'rebook_confirmed'
  title: string
  message: string
  timestamp: number
  doctorId?: number
  doctorName?: string
  appointmentId?: number
  icon: string // emoji or icon name
  isRead?: boolean
}

const getNotificationsKey = async (): Promise<string> => {
  const patientId = await AsyncStorage.getItem("patientId") || await AsyncStorage.getItem("userId");
  return `@notifications_${patientId || 'guest'}`
}

export async function getNotifications(): Promise<Notification[]> {
  try {
    const key = await getNotificationsKey()
    const stored = await AsyncStorage.getItem(key)
    return stored ? JSON.parse(stored) : []
  } catch {
    return []
  }
}

export async function addNotification(notification: Notification): Promise<void> {
  try {
    const key = await getNotificationsKey()
    const existing = await getNotifications()
    const updated = [{ ...notification, isRead: false }, ...existing].slice(0, 50) // Keep last 50
    await AsyncStorage.setItem(key, JSON.stringify(updated))
  } catch (e) {
    console.error('Failed to save notification:', e)
  }
}

export async function markAllAsRead(): Promise<void> {
  try {
    const key = await getNotificationsKey()
    const existing = await getNotifications()
    const updated = existing.map(n => ({ ...n, isRead: true }))
    await AsyncStorage.setItem(key, JSON.stringify(updated))
  } catch (e) {
    console.error('Failed to mark read:', e)
  }
}

export async function clearNotifications(): Promise<void> {
  try {
    const key = await getNotificationsKey()
    await AsyncStorage.removeItem(key)
  } catch (e) {
    console.error('Failed to clear notifications:', e)
  }
}

export async function deleteNotification(id: string): Promise<void> {
  try {
    const key = await getNotificationsKey()
    const existing = await getNotifications()
    const updated = existing.filter(n => n.id !== id)
    await AsyncStorage.setItem(key, JSON.stringify(updated))
  } catch (e) {
    console.error('Failed to delete notification:', e)
  }
}

export function createScheduleReadyNotification(doctorName: string): Notification {
  return {
    id: `schedule_${Date.now()}`,
    type: 'schedule_ready',
    icon: '📅',
    title: 'Schedule Updated',
    message: `Dr. ${doctorName} has updated their schedule`,
    timestamp: Date.now(),
  }
}

export function createAppointmentConfirmedNotification(doctorName: string, date: string, time: string): Notification {
  return {
    id: `confirmed_${Date.now()}`,
    type: 'appointment_confirmed',
    icon: '✅',
    title: 'Appointment Confirmed',
    message: `Your appointment with Dr. ${doctorName} on ${date} at ${time} is confirmed`,
    timestamp: Date.now(),
  }
}

export function createAppointmentCancelledNotification(doctorName: string): Notification {
  return {
    id: `cancelled_${Date.now()}`,
    type: 'appointment_cancelled',
    icon: '❌',
    title: 'Appointment Cancelled',
    message: `Your appointment with Dr. ${doctorName} has been cancelled`,
    timestamp: Date.now(),
  }
}

export function createAppointmentUpdatedNotification(title: string, message: string, type: Notification['type'] = 'appointment_updated'): Notification {
  return {
    id: `appt_${Date.now()}`,
    type,
    icon: '🔔',
    title,
    message,
    timestamp: Date.now(),
  }
}
