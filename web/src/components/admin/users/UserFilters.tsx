import { motion } from 'framer-motion'
import { Search, Filter, X } from 'lucide-react'
import type { UserRole } from '@/lib/types'
import { Button } from '@/components/ui/Button'
import { ROLE_FILTERS } from './constants'

interface UserFiltersProps {
  search: string
  onSearch: (value: string) => void
  roleFilter: UserRole | ''
  onRoleFilter: (role: UserRole | '') => void
  onClear: () => void
}

export const UserFilters = ({ 
  search, 
  onSearch, 
  roleFilter, 
  onRoleFilter, 
  onClear 
}: UserFiltersProps) => {
  return (
    <div className="flex flex-wrap items-center gap-3 p-5 border-b border-gray-100 bg-white/50 rounded-t-3xl">
      {/* Search Input */}
      <div className="relative flex-1 min-w-48 max-w-md">
        <Search className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
        <input
          type="text"
          value={search}
          onChange={(e) => onSearch(e.target.value)}
          placeholder="🔍 بحث بالاسم أو البريد الإلكتروني..."
          className="w-full pr-10 pl-4 py-3 text-sm border border-gray-200 rounded-2xl bg-white/80 focus:outline-none focus:ring-4 focus:ring-blue-200 focus:border-blue-400 transition-all duration-200"
        />
      </div>
      
      {/* Role Filters */}
      <div className="flex items-center gap-2">
        <Filter className="w-4 h-4 text-gray-400" />
        <div className="flex items-center gap-1.5 bg-gray-100/80 rounded-2xl p-1">
          {ROLE_FILTERS.map(({ value, label, emoji }) => (
            <motion.button
              key={value}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onRoleFilter(value)}
              className={`px-4 py-2 text-xs font-semibold rounded-xl transition-all duration-200 ${
                roleFilter === value
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/30'
                  : 'text-gray-600 hover:bg-white hover:shadow-md'
              }`}
            >
              {emoji} {label}
            </motion.button>
          ))}
        </div>
      </div>
      
      {/* Clear Filters */}
      {(search || roleFilter) && (
        <Button variant="ghost" size="sm" onClick={onClear} className="text-gray-500 hover:text-gray-700">
          <X className="w-4 h-4" /> مسح
        </Button>
      )}
    </div>
  )
}