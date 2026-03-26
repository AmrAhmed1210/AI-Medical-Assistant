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