import { cn } from '@/lib/utils'
import { URGENCY_CONFIG, STATUS_CONFIG } from '@/lib/utils'
import type { UrgencyLevel, AppointmentStatus } from '@/lib/types'

interface BadgeProps {
  children: React.ReactNode
  className?: string
  color?: 'blue' | 'green' | 'red' | 'yellow' | 'gray' | 'purple'
}

const colors = {
  blue: 'bg-blue-100 text-blue-700',
  green: 'bg-green-100 text-green-700',
  red: 'bg-red-100 text-red-700',
  yellow: 'bg-amber-100 text-amber-700',
  gray: 'bg-gray-100 text-gray-600',
  purple: 'bg-purple-100 text-purple-700',
}

export function Badge({ children, className, color = 'blue' }: BadgeProps) {
  return (
    <span className={cn('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium', colors[color], className)}>
      {children}
    </span>
  )
}

export function UrgencyBadge({ level }: { level: UrgencyLevel }) {
  const cfg = URGENCY_CONFIG[level]
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold border',
        cfg.bg, cfg.color, cfg.border,
        cfg.pulse && 'animate-pulse'
      )}
    >
      {cfg.pulse && <span className="w-1.5 h-1.5 rounded-full bg-current inline-block" />}
      {cfg.label}
    </span>
  )
}

export function StatusBadge({ status }: { status: AppointmentStatus }) {
  const cfg = STATUS_CONFIG[status]
  return (
    <span className={cn('inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium', cfg.bg, cfg.color)}>
      {cfg.label}
    </span>
  )
}
