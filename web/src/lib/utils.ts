import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'
import { format, formatDistanceToNow, parseISO } from 'date-fns'
import { ar } from 'date-fns/locale'
import type { UrgencyLevel, AppointmentStatus } from './types'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(dateStr: string, fmt = 'dd/MM/yyyy') {
  try {
    return format(parseISO(dateStr), fmt, { locale: ar })
  } catch {
    return dateStr
  }
}

export function formatDateTime(dateStr: string) {
  try {
    return format(parseISO(dateStr), 'dd/MM/yyyy - hh:mm a', { locale: ar })
  } catch {
    return dateStr
  }
}

export function formatTimeAgo(dateStr: string) {
  try {
    return formatDistanceToNow(parseISO(dateStr), { addSuffix: true, locale: ar })
  } catch {
    return dateStr
  }
}

export function formatCurrency(amount: number) {
  return new Intl.NumberFormat('ar-EG', {
    style: 'currency',
    currency: 'EGP',
  }).format(amount)
}

export const URGENCY_CONFIG: Record<UrgencyLevel, {
  label: string
  color: string
  bg: string
  border: string
  pulse: boolean
}> = {
  LOW: {
    label: 'منخفض',
    color: 'text-green-700',
    bg: 'bg-green-100',
    border: 'border-green-300',
    pulse: false,
  },
  MEDIUM: {
    label: 'متوسط',
    color: 'text-amber-700',
    bg: 'bg-amber-100',
    border: 'border-amber-300',
    pulse: false,
  },
  HIGH: {
    label: 'مرتفع',
    color: 'text-red-700',
    bg: 'bg-red-100',
    border: 'border-red-300',
    pulse: false,
  },

}

export const STATUS_CONFIG: Record<AppointmentStatus, {
  label: string
  color: string
  bg: string
}> = {
  Pending: { label: 'قيد الانتظار', color: 'text-amber-700', bg: 'bg-amber-100' },
  Confirmed: { label: 'Confirmed', color: 'text-green-700', bg: 'bg-green-100' },
  Cancelled: { label: 'Cancelled', color: 'text-gray-600', bg: 'bg-gray-100' },
  Completed: { label: 'Completed', color: 'text-blue-700', bg: 'bg-blue-100' },
}

export const DAY_NAMES_AR = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
export const DAY_NAMES_EN = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

export function getInitials(name: string) {
  return name
    .split(' ')
    .slice(0, 2)
    .map((n) => n[0])
    .join('')
}

export function generateId() {
  return Math.random().toString(36).substring(2, 9)
}

export function debounce<T extends (...args: unknown[]) => unknown>(fn: T, delay: number) {
  let timer: ReturnType<typeof setTimeout>
  return (...args: Parameters<T>) => {
    clearTimeout(timer)
    timer = setTimeout(() => fn(...args), delay)
  }
}
