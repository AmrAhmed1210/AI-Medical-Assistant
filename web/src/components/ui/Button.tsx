import React from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { LoadingSpinner } from './LoadingSpinner'

type Variant = 'primary' | 'secondary' | 'danger' | 'ghost' | 'outline' | 'success'
type Size = 'sm' | 'md' | 'lg'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
  loading?: boolean
  icon?: React.ReactNode
  iconPosition?: 'start' | 'end'
}

const variants: Record<Variant, string> = {
  primary: 'bg-primary-600 hover:bg-primary-700 text-white shadow-sm',
  secondary: 'bg-gray-100 hover:bg-gray-200 text-gray-700',
  danger: 'bg-red-500 hover:bg-red-600 text-white shadow-sm',
  ghost: 'hover:bg-gray-100 text-gray-600',
  outline: 'border border-gray-300 hover:bg-gray-50 text-gray-700',
  success: 'bg-green-500 hover:bg-green-600 text-white shadow-sm',
}

const sizes: Record<Size, string> = {
  sm: 'px-3 py-1.5 text-sm gap-1.5',
  md: 'px-4 py-2 text-sm gap-2',
  lg: 'px-5 py-2.5 text-base gap-2',
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = 'primary', size = 'md', loading, icon, iconPosition = 'start', className, children, disabled, ...props }, ref) => {
    return (
      <motion.button
        ref={ref}
        whileTap={{ scale: 0.97 }}
        className={cn(
          'inline-flex items-center justify-center rounded-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-1 disabled:opacity-50 disabled:cursor-not-allowed',
          variants[variant],
          sizes[size],
          className
        )}
        disabled={disabled || loading}
        {...(props as React.ComponentProps<typeof motion.button>)}
      >
        {loading ? (
          <LoadingSpinner size="sm" className="text-current" />
        ) : (
          iconPosition === 'start' && icon && <span>{icon}</span>
        )}
        {children}
        {!loading && iconPosition === 'end' && icon && <span>{icon}</span>}
      </motion.button>
    )
  }
)

Button.displayName = 'Button'
