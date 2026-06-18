import type { ReactNode } from 'react'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  glass?: boolean
  children: ReactNode
}

export const Card = ({ className = '', children, glass = true, ...props }: CardProps) => (
  <div className={`rounded-3xl border ${glass ? 'border-white/30 bg-white/30 backdrop-blur-xl' : 'border-white/40 bg-white/70 backdrop-blur-2xl'} shadow-[0_8px_30px_rgb(0,0,0,0.04)] dark:bg-slate-900/60 dark:border-slate-700/50 dark:shadow-[0_8px_30px_rgb(0,0,0,0.2)] text-gray-900 dark:text-slate-100 transition-all duration-300 ${className}`} {...props}>
    {children}
  </div>
)

export const CardHeader = ({ className = '', ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`flex flex-col space-y-2 p-6 ${className}`} {...props} />
)

export const CardTitle = ({ className = '', ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
  <h3 className={`text-xl font-bold tracking-tight text-gray-900 dark:text-white transition-colors ${className}`} {...props} />
)

export const CardDescription = ({ className = '', ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
  <p className={`text-sm text-gray-500 dark:text-slate-400 transition-colors ${className}`} {...props} />
)

export const CardContent = ({ className = '', ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`p-6 pt-0 ${className}`} {...props} />
)