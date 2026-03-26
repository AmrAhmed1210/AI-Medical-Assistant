import { ChevronLeft } from 'lucide-react'

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  error?: string
}

export const Select = ({ className = '', error, children, ...props }: SelectProps) => (
  <div className="relative">
    <select
      className={`w-full px-4 py-3 text-sm border rounded-2xl focus:outline-none focus:ring-4 transition-all duration-200 appearance-none bg-white/50 ${
        error 
          ? 'border-red-300 bg-red-50/50 focus:ring-red-200 focus:border-red-400' 
          : 'border-gray-200 focus:ring-blue-200 focus:border-blue-400 hover:border-gray-300'
      } ${className}`}
      {...props}
    >
      {children}
    </select>
    <ChevronLeft className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
    {error && <p className="mt-1.5 text-xs text-red-500 font-medium">{error}</p>}
  </div>
)