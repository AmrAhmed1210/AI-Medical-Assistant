import { useMemo } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from './Button'

interface PaginationProps {
  total: number
  page: number
  pageSize: number
  onChange: (page: number) => void
}

export const Pagination = ({ total, page, pageSize, onChange }: PaginationProps) => {
  const totalPages = Math.ceil(total / pageSize)
  if (totalPages <= 1) return null
  
  const pages = useMemo(() => {
    const items: (number | string)[] = []
    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) items.push(i)
    } else {
      items.push(1)
      if (page > 3) items.push('...')
      for (let i = Math.max(2, page - 1); i <= Math.min(totalPages - 1, page + 1); i++) items.push(i)
      if (page < totalPages - 2) items.push('...')
      items.push(totalPages)
    }
    return items
  }, [page, totalPages])
  
  return (
    <div className="flex items-center justify-center gap-1.5 p-4">
      <Button variant="outline" size="sm" onClick={() => onChange(page - 1)} disabled={page === 1} className="px-3">
        <ChevronRight className="w-4 h-4" />
      </Button>
      {pages.map((p, i) => (
        p === '...' ? (
          <span key={`ellipsis-${i}`} className="px-3 py-2 text-sm text-gray-400">...</span>
        ) : (
          <button
            key={p}
            onClick={() => onChange(p as number)}
            className={`min-w-[40px] h-10 px-3 py-2 text-sm font-medium rounded-xl transition-all duration-200 ${
              page === p
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/30'
                : 'text-gray-600 hover:bg-gray-100/80'
            }`}
          >
            {p}
          </button>
        )
      ))}
      <Button variant="outline" size="sm" onClick={() => onChange(page + 1)} disabled={page === totalPages} className="px-3">
        <ChevronLeft className="w-4 h-4" />
      </Button>
    </div>
  )
}