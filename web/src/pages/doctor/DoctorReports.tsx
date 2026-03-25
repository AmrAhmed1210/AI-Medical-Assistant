import { useState } from 'react'
import { Brain, ExternalLink, AlertTriangle } from 'lucide-react'
import { useDoctorReports } from '@/hooks/useDoctor'
import { AIReportCard } from '@/components/doctor/AIReportCard'
import { Modal } from '@/components/ui/Modal'
import { UrgencyBadge } from '@/components/ui/Badge'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import type { AIReportDto, UrgencyLevel } from '@/lib/types'
import { formatDateTime } from '@/lib/utils'

const URGENCY_FILTERS: { label: string; value: UrgencyLevel | '' }[] = [
  { label: 'الكل', value: '' },
  { label: 'منخفض', value: 'LOW' },
  { label: 'متوسط', value: 'MEDIUM' },
  { label: 'مرتفع', value: 'HIGH' },
  { label: 'طوارئ', value: 'EMERGENCY' },
]

export default function DoctorReports() {
  const [urgencyFilter, setUrgencyFilter] = useState<UrgencyLevel | ''>('')
  const { reports, isLoading } = useDoctorReports({ urgency: urgencyFilter || undefined })
  const [selected, setSelected] = useState<AIReportDto | null>(null)

  const filtered = urgencyFilter ? reports.filter((r) => r.urgencyLevel === urgencyFilter) : reports

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-bold text-gray-800">تقارير AI</h1>
        <p className="text-sm text-gray-500 mt-0.5">{filtered.length} تقرير</p>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2 flex-wrap">
        {URGENCY_FILTERS.map((f) => (
          <button
            key={f.value}
            onClick={() => setUrgencyFilter(f.value)}
            className={`px-3 py-1.5 text-xs rounded-xl font-medium transition-colors ${
              urgencyFilter === f.value ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {isLoading ? <PageLoader /> : (
        filtered.length === 0 ? (
          <div className="text-center py-20 text-gray-400">
            <Brain size={48} className="mx-auto mb-3 opacity-30" />
            <p>لا توجد تقارير</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filtered.map((r, i) => (
              <AIReportCard key={r.reportId} report={r} index={i} onClick={() => setSelected(r)} />
            ))}
          </div>
        )
      )}

      {/* Report Detail Modal */}
      <Modal open={!!selected} onClose={() => setSelected(null)} title="تفاصيل التقرير" size="xl">
        {selected && (
          <div className="space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-gray-800">{selected.patientName}</h3>
                <p className="text-xs text-gray-400">{formatDateTime(selected.createdAt)}</p>
              </div>
              <UrgencyBadge level={selected.urgencyLevel} />
            </div>

            <div className="bg-gray-50 rounded-xl p-4">
              <p className="text-sm font-medium text-gray-700 mb-1">ملخص التشخيص</p>
              <p className="text-sm text-gray-600">{selected.responseSummary}</p>
            </div>

            <div>
              <p className="text-sm font-medium text-gray-700 mb-2">الأعراض ({selected.symptomsJson.length})</p>
              <div className="space-y-2">
                {selected.symptomsJson.map((s, i) => (
                  <div key={i} className="flex items-start gap-2 p-2 bg-gray-50 rounded-lg">
                    <AlertTriangle size={13} className="text-amber-500 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="text-sm font-medium text-gray-800">{s.symptom}</p>
                      {s.icdCode && <span className="text-xs text-gray-400 font-mono">{s.icdCode}</span>}
                      {s.description && <p className="text-xs text-gray-500 mt-0.5">{s.description}</p>}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-4 p-3 bg-blue-50 rounded-xl">
              <div>
                <p className="text-xs text-gray-500">التخصص المقترح</p>
                <p className="font-semibold text-blue-700">{selected.recommendedSpecialty}</p>
              </div>
              <div className="mr-auto">
                <p className="text-xs text-gray-500 mb-1">نسبة الهلوسة</p>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${selected.hallucinationScore < 0.2 ? 'bg-green-500' : selected.hallucinationScore < 0.5 ? 'bg-amber-500' : 'bg-red-500'}`}
                      style={{ width: `${selected.hallucinationScore * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-medium">{Math.round(selected.hallucinationScore * 100)}%</span>
                </div>
              </div>
            </div>

            {selected.citations.length > 0 && (
              <div>
                <p className="text-sm font-medium text-gray-700 mb-2">المصادر والاستشهادات</p>
                <div className="space-y-1.5">
                  {selected.citations.map((c, i) => (
                    <a
                      key={i}
                      href={c.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 p-2 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors text-sm text-blue-600 hover:text-blue-700"
                    >
                      <ExternalLink size={12} />
                      <span className="flex-1 truncate">{c.title}</span>
                      <span className="text-xs text-gray-400 flex-shrink-0">{c.source}</span>
                    </a>
                  ))}
                </div>
              </div>
            )}

            <div>
              <p className="text-sm font-medium text-gray-700 mb-2">التقرير الكامل</p>
              <div className="bg-gray-50 rounded-xl p-4 text-sm text-gray-600 whitespace-pre-wrap max-h-40 overflow-y-auto leading-relaxed">
                {selected.fullResponseText}
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}
