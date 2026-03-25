import React from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

interface CardProps {
  children: React.ReactNode
  className?: string
  hover?: boolean
  padding?: 'none' | 'sm' | 'md' | 'lg'
  onClick?: () => void
}

const paddings = {
  none: '',
  sm: 'p-4',
  md: 'p-5',
  lg: 'p-6',
}

export function Card({ children, className, hover, padding = 'md', onClick }: CardProps) {
  const Comp = hover || onClick ? motion.div : 'div'
  const motionProps = hover || onClick ? { whileHover: { y: -2 }, transition: { duration: 0.2 } } : {}

  return (
    <Comp
      className={cn(
        'bg-white rounded-xl border border-gray-100 shadow-sm',
        paddings[padding],
        hover && 'cursor-pointer',
        className
      )}
      onClick={onClick}
      {...motionProps}
    >
      {children}
    </Comp>
  )
}

export function CardHeader({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={cn('flex items-center justify-between mb-4', className)}>
      {children}
    </div>
  )
}

export function CardTitle({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <h3 className={cn('text-base font-semibold text-gray-800', className)}>{children}</h3>
  )
}
