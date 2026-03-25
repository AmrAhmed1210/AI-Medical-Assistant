import { motion } from 'framer-motion'
import { Brain, AlertTriangle, BookOpen, ChevronLeft } from 'lucide-react'
import type { AIReportDto } from '@/lib/types'
import { UrgencyBadge } from '@/components/ui/Badge'
import { formatDate, formatTimeAgo } from '@/lib/utils'

interface AIReportCardProps {
  report: AIReportDto
  onClick?: () => void
  index?: number
}

export function AIReportCard({ report, onClick, index = 0 }: AIReportCardProps) {
  const hallucinationPct = Math.round(report.hallucinationScore * 100)
  const hallucinationColor = hallucinationPct < 20 ? 'bg-green-500' : hallucinationPct < 50 ? 'bg-amber-500' : 'bg-red-500'

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      onClick={onClick}
      className="bg-white rounded-xl border border-gray-100 shadow-sm p-5 hover:shadow-md transition-shadow cursor-pointer group"
    >
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

      <p className="text-sm text-gray-600 line-clamp-2 mb-3">{report.responseSummary}</p>

      <div className="flex items-center gap-3 text-xs text-gray-500 mb-3">
        <span className="flex items-center gap-1">
          <BookOpen size={11} />
          {report.recommendedSpecialty}
        </span>
        <span className="flex items-center gap-1">
          <AlertTriangle size={11} />
          {report.symptomsJson.length} أعراض
        </span>
      </div>

      {/* Hallucination score */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs text-gray-400">دقة AI:</span>
        <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${hallucinationColor}`}
            style={{ width: `${100 - hallucinationPct}%` }}
          />
        </div>
        <span className="text-xs font-medium text-gray-600">{100 - hallucinationPct}%</span>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex gap-1">
          {report.citations.slice(0, 3).map((_, i) => (
            <div key={i} className="w-4 h-4 rounded-full bg-blue-100 border border-white" />
          ))}
          {report.citations.length > 3 && (
            <div className="w-4 h-4 rounded-full bg-gray-100 flex items-center justify-center">
              <span className="text-[8px] text-gray-500">+{report.citations.length - 3}</span>
            </div>
          )}
        </div>
        <ChevronLeft size={14} className="text-gray-300 group-hover:text-primary-500 group-hover:-translate-x-0.5 transition-all" />
      </div>
    </motion.div>
  )
}
