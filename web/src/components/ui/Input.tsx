import { AlertCircle } from 'lucide-react'

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  error?: string
  icon?: React.ReactNode
}

export const Input = ({ className = '', error, icon, ...props }: InputProps) => (
  <div className="relative">
    {icon && (
      <div className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400">
        {icon}
      </div>
    )}
    <input
      className={`w-full ${icon ? 'pr-10' : 'pr-4'} pl-4 py-3 text-sm border rounded-2xl focus:outline-none focus:ring-4 transition-all duration-200 ${
        error 
          ? 'border-red-300 bg-red-50/50 focus:ring-red-200 focus:border-red-400' 
          : 'border-gray-200 bg-white/50 focus:ring-blue-200 focus:border-blue-400 hover:border-gray-300'
      } ${className}`}
      {...props}
    />
    {error && (
      <p className="mt-1.5 text-xs text-red-500 font-medium flex items-center gap-1">
        <AlertCircle className="w-3 h-3" />
        {error}
      </p>
    )}
  </div>
)