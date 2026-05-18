import type { ReactNode } from 'react'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: ReactNode
}

export const Card = ({ className = '', children, ...props }: CardProps) => (
  <div className={`rounded-3xl border border-white/20 bg-white/80 backdrop-blur-xl shadow-xl dark:bg-slate-950/65 dark:border-slate-800/80 dark:shadow-2xl dark:shadow-black/20 text-gray-900 dark:text-slate-100 transition-all duration-300 ${className}`} {...props}>
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