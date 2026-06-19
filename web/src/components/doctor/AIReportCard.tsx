import { motion } from 'framer-motion'
import { Brain, AlertTriangle, ChevronLeft } from 'lucide-react'
import type { AIReportDto } from '@/lib/types'
import { UrgencyBadge } from '@/components/ui/Badge'
import { formatDate } from '@/lib/utils'

interface AIReportCardProps {
  report: AIReportDto
  onClick?: () => void
  index?: number
  lang?: 'en' | 'ar'
}

export function AIReportCard({ report, onClick, index = 0, lang = 'en' }: AIReportCardProps) {
  const isAr = lang === 'ar'

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      onClick={onClick}
      className="bg-white dark:bg-slate-900 rounded-xl border border-gray-100 dark:border-slate-800 shadow-sm p-5 hover:shadow-md transition-shadow cursor-pointer group relative overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-primary-50 dark:bg-primary-950/20 rounded-lg">
            <Brain size={16} className="text-primary-600" />
          </div>
          <div>
            <p className="font-semibold text-gray-800 dark:text-slate-100 text-sm">{report.patientName}</p>
            <p className="text-xs text-gray-400 dark:text-slate-500">{formatDate(report.createdAt)}</p>
          </div>
        </div>
        <UrgencyBadge level={report.urgencyLevel} />
      </div>

      {/* Symptoms */}
      <div className={`flex items-center gap-1.5 text-xs text-gray-500 dark:text-slate-400 mb-3 ${isAr ? 'flex-row-reverse' : ''}`}>
        <AlertTriangle size={11} className="text-amber-500 flex-shrink-0" />
        <span>{report.symptoms.length} {isAr ? 'أعراض' : 'symptoms'}</span>
        <div className={`flex gap-1.5 overflow-hidden ${isAr ? 'flex-row-reverse' : ''}`}>
          {report.symptoms.slice(0, 2).map((s, i) => (
            <span key={i} className="px-1.5 py-0.5 bg-gray-100 dark:bg-slate-800 rounded-md text-gray-600 dark:text-slate-350 truncate max-w-[80px]" title={isAr ? (s.termAr || s.term) : s.term}>
              {isAr ? (s.termAr || s.term) : s.term}
            </span>
          ))}
        </div>
        {report.symptoms.length > 2 && (
          <span className="text-gray-400 dark:text-slate-500">+{report.symptoms.length - 2}</span>
        )}
      </div>

      {/* Disclaimer */}
      <p className={`text-xs text-gray-400 dark:text-slate-500 line-clamp-2 mb-3 italic ${isAr ? 'text-right' : 'text-left'}`} dir={isAr ? 'rtl' : 'ltr'}>
        {isAr ? (report.disclaimerAr || report.disclaimer) : report.disclaimer}
      </p>

      {/* Footer */}
      <div className={`flex items-center ${isAr ? 'justify-start' : 'justify-end'}`}>
        <ChevronLeft
          size={14}
          className={`text-gray-300 group-hover:text-primary-500 group-hover:-translate-x-0.5 transition-all ${isAr ? 'rotate-180' : ''}`}
        />
      </div>
    </motion.div>
  )
}