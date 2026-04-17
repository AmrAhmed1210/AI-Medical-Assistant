// components/ui/Badge.tsx

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'admin'
}

export const Badge = ({ 
  variant = 'default', 
  className = '', 
  children,
  ...props 
}: BadgeProps) => {
  const variants = {
    default: 'bg-gray-100/80 text-gray-700 border border-gray-200',
    success: 'bg-gradient-to-r from-green-400 to-emerald-500 text-white shadow-md shadow-green-500/20',
    warning: 'bg-gradient-to-r from-amber-400 to-orange-500 text-white shadow-md shadow-amber-500/20',
    danger: 'bg-gradient-to-r from-red-400 to-rose-500 text-white shadow-md shadow-red-500/20',
    info: 'bg-gradient-to-r from-blue-400 to-indigo-500 text-white shadow-md shadow-blue-500/20',
    admin: 'bg-gradient-to-r from-violet-500 to-purple-600 text-white shadow-md shadow-purple-500/20',
  }
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${variants[variant]} ${className}`} {...props}>
      {children}
    </span>
  )
}

// ─────────────────────────────────────────────────────────────────────────────

type AppointmentStatus = 'Pending' | 'Confirmed' | 'Cancelled' | 'Completed' | string

interface StatusBadgeProps {
  status: AppointmentStatus
}

const statusConfig: Record<string, { label: string; variant: BadgeProps['variant']; dot: string }> = {
  Pending: {
    label: 'Pending',
    variant: 'warning',
    dot: 'bg-white/70',
  },
  Confirmed: {
    label: 'Confirmed',
    variant: 'info',
    dot: 'bg-white/70',
  },
  Completed: {
    label: 'Completed',
    variant: 'success',
    dot: 'bg-white/70',
  },
  Cancelled: {
    label: 'Cancelled',
    variant: 'danger',
    dot: 'bg-white/70',
  },
}

export const StatusBadge = ({ status }: StatusBadgeProps) => {
  const config = statusConfig[status] ?? {
    label: status,
    variant: 'default' as const,
    dot: 'bg-gray-400',
  }

  return (
    <Badge variant={config.variant}>
      <span className={`w-1.5 h-1.5 rounded-full ${config.dot}`} />
      {config.label}
    </Badge>
  )
}

type UrgencyLevel = 'LOW' | 'MEDIUM' | 'HIGH'

interface UrgencyBadgeProps {
  level: UrgencyLevel
}

const urgencyConfig: Record<UrgencyLevel, { label: string; variant: BadgeProps['variant'] }> = {
  LOW:    { label: 'Low',    variant: 'success' },
  MEDIUM: { label: 'Medium', variant: 'warning' },
  HIGH:   { label: 'High',   variant: 'danger'  },
}

export const UrgencyBadge = ({ level }: UrgencyBadgeProps) => {
  const config = urgencyConfig[level] ?? { label: level, variant: 'default' as const }

  return (
    <Badge variant={config.variant}>
      {config.label}
    </Badge>
  )
}