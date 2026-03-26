import { motion } from 'framer-motion'
import { Brain, AlertTriangle, ChevronLeft } from 'lucide-react'
import type { AIReportDto } from '@/lib/types'
import { UrgencyBadge } from '@/components/ui/Badge'
import { formatDate } from '@/lib/utils'

interface AIReportCardProps {
  report: AIReportDto
  onClick?: () => void
  index?: number
}

export function AIReportCard({ report, onClick, index = 0 }: AIReportCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      onClick={onClick}
      className="bg-white rounded-xl border border-gray-100 shadow-sm p-5 hover:shadow-md transition-shadow cursor-pointer group"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-primary-50 rounded-lg">
            <Brain size={16} className="text-primary-600" />
          </div>
          <div>
            <p className="font-semibold text-gray-800 text-sm">{report.patientName}</p>
            <p className="text-xs text-gray-400">{formatDate(report.createdAt)}</p>
          </div>
        </div>
        <UrgencyBadge level={report.urgencyLevel} />
      </div>

      {/* Symptoms */}
      <div className="flex items-center gap-1.5 text-xs text-gray-500 mb-3">
        <AlertTriangle size={11} />
        <span>{report.symptoms.length} أعراض</span>
        {report.symptoms.slice(0, 2).map((s, i) => (
          <span key={i} className="px-1.5 py-0.5 bg-gray-100 rounded-md text-gray-600">
            {s.term}
          </span>
        ))}
        {report.symptoms.length > 2 && (
          <span className="text-gray-400">+{report.symptoms.length - 2}</span>
        )}
      </div>

      {/* Disclaimer */}
      <p className="text-xs text-gray-400 line-clamp-2 mb-3 italic">{report.disclaimer}</p>

      {/* Footer */}
      <div className="flex items-center justify-end">
        <ChevronLeft
          size={14}
          className="text-gray-300 group-hover:text-primary-500 group-hover:-translate-x-0.5 transition-all"
        />
      </div>
    </motion.div>
  )
}