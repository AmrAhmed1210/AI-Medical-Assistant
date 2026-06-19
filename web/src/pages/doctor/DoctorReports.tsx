import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, AlertTriangle, FileText, Globe } from 'lucide-react'
import { useDoctorReports } from '@/hooks/useDoctor'
import { AIReportCard } from '@/components/doctor/AIReportCard'
import { Modal } from '@/components/ui/Modal'
import { UrgencyBadge } from '@/components/ui/Badge'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import type { AIReportDto, UrgencyLevel } from '@/lib/types'
import { formatDateTime } from '@/lib/utils'

const URGENCY_FILTERS: { label: string; value: UrgencyLevel | '' }[] = [
  { label: 'All', value: '' },
  { label: 'Low', value: 'LOW' },
  { label: 'Medium', value: 'MEDIUM' },
  { label: 'High', value: 'HIGH' },
]

export default function DoctorReports() {
  const [urgencyFilter, setUrgencyFilter] = useState<UrgencyLevel | ''>('')
  const { reports, isLoading } = useDoctorReports({ urgency: urgencyFilter || undefined })
  const [selected, setSelected] = useState<AIReportDto | null>(null)
  const [globalLang, setGlobalLang] = useState<'en' | 'ar'>('en')
  const [modalLang, setModalLang] = useState<'en' | 'ar'>('en')

  useEffect(() => {
    if (selected) {
      setModalLang(globalLang)
    }
  }, [selected, globalLang])

  const filtered = urgencyFilter ? reports.filter((r) => r.urgencyLevel === urgencyFilter) : reports

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3">
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-600 to-purple-400 rounded-xl blur-lg opacity-30" />
          <div className="relative bg-gradient-to-br from-purple-600 to-purple-500 rounded-xl p-3 shadow-lg">
            <Brain size={28} className="text-white" />
          </div>
        </div>
        <div>
          <h1 className="text-xl font-bold text-gray-800">AI Reports</h1>
          <p className="text-sm text-gray-500 mt-0.5">{filtered.length} report(s) found</p>
        </div>
      </div>

      {/* Filters and Language Switcher */}
      <motion.div
        className="flex items-center justify-between flex-wrap gap-4"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-2 flex-wrap">
          {URGENCY_FILTERS.map((f, idx) => (
            <motion.button
              key={f.value}
              onClick={() => setUrgencyFilter(f.value)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className={`px-4 py-2 text-xs rounded-xl font-medium transition-all shadow-sm ${urgencyFilter === f.value
                  ? 'bg-gradient-to-r from-primary-600 to-primary-500 text-white shadow-lg'
                  : 'bg-white border border-gray-200 dark:border-slate-800 text-gray-600 hover:border-primary-405'
                }`}
            >
              {f.label}
            </motion.button>
          ))}
        </div>

        {/* Global Language Toggle */}
        <div className="flex bg-gray-100 dark:bg-slate-800 p-0.5 rounded-xl border border-gray-200/50 dark:border-slate-700/50 shadow-inner items-center gap-1">
          <Globe size={14} className="text-gray-400 mx-2" />
          <button
            onClick={() => setGlobalLang('en')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${globalLang === 'en' ? 'bg-white dark:bg-slate-700 text-primary-600 dark:text-primary-400 shadow-sm' : 'text-gray-500 hover:text-gray-850 dark:hover:text-slate-205'}`}
          >
            English
          </button>
          <button
            onClick={() => setGlobalLang('ar')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${globalLang === 'ar' ? 'bg-white dark:bg-slate-700 text-primary-600 dark:text-primary-400 shadow-sm' : 'text-gray-500 hover:text-gray-850 dark:hover:text-slate-205'}`}
          >
            العربية
          </button>
        </div>
      </motion.div>

      {isLoading ? <PageLoader /> : (
        filtered.length === 0 ? (
          <motion.div
            className="text-center py-20 text-gray-400"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <motion.div animate={{ y: [0, -10, 0] }} transition={{ duration: 3, repeat: Infinity }}>
              <Brain size={48} className="mx-auto mb-3 opacity-20" />
            </motion.div>
            <p className="text-sm font-medium">No reports available</p>
            <p className="text-xs">Try adjusting your filters</p>
          </motion.div>
        ) : (
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <AnimatePresence>
              {filtered.map((r, i) => (
                <motion.div
                  key={r.reportId}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ delay: i * 0.05 }}
                  onClick={() => setSelected(r)}
                  className="cursor-pointer"
                >
                  <AIReportCard report={r} index={i} onClick={() => setSelected(r)} lang={globalLang} />
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>
        )
      )}

      {/* Report Detail Modal */}
      <Modal open={!!selected} onClose={() => setSelected(null)} title={modalLang === 'ar' ? 'تفاصيل التقرير' : 'Report Details'} size="xl">
        {selected && (
          <motion.div
            className="space-y-5"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {/* Modal Language Toggle */}
            <div className="flex justify-end">
              <div className="flex bg-gray-100 dark:bg-slate-800 p-0.5 rounded-lg border border-gray-250/30 dark:border-slate-700/50 shadow-inner">
                <button
                  onClick={() => setModalLang('en')}
                  className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-all ${modalLang === 'en' ? 'bg-white dark:bg-slate-700 text-primary-600 dark:text-primary-400 shadow-sm' : 'text-gray-500 hover:text-gray-800 dark:hover:text-slate-200'}`}
                >
                  EN
                </button>
                <button
                  onClick={() => setModalLang('ar')}
                  className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-all ${modalLang === 'ar' ? 'bg-white dark:bg-slate-700 text-primary-600 dark:text-primary-400 shadow-sm' : 'text-gray-500 hover:text-gray-800 dark:hover:text-slate-200'}`}
                >
                  عربي
                </button>
              </div>
            </div>

            {/* Header block */}
            <motion.div
              className={`flex items-center justify-between p-4 bg-gradient-to-r from-primary-50 to-blue-50 dark:from-slate-900 dark:to-slate-950 rounded-xl border border-primary-100 dark:border-slate-850 ${modalLang === 'ar' ? 'flex-row-reverse text-right' : 'text-left'}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.1 }}
            >
              <div>
                <h3 className="font-semibold text-gray-800 dark:text-slate-100">{selected.patientName}</h3>
                <p className="text-xs text-gray-500 mt-0.5">{formatDateTime(selected.createdAt)}</p>
              </div>
              <motion.div whileHover={{ scale: 1.05 }}>
                <UrgencyBadge level={selected.urgencyLevel} />
              </motion.div>
            </motion.div>

            {/* Symptoms Section */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="space-y-3"
            >
              <div className={`flex items-center gap-2 ${modalLang === 'ar' ? 'flex-row-reverse' : ''}`}>
                <AlertTriangle size={18} className="text-amber-500 flex-shrink-0" />
                <p className="text-sm font-semibold text-gray-800 dark:text-slate-200">
                  {modalLang === 'ar' ? `الأعراض المكتشفة (${selected.symptoms.length})` : `Symptoms Detected (${selected.symptoms.length})`}
                </p>
              </div>
              <div className="space-y-2">
                {selected.symptoms.map((s, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 + i * 0.05 }}
                    className={`flex items-start gap-3 p-3 bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950/10 dark:to-orange-950/10 rounded-lg border border-amber-100 dark:border-amber-900/30 hover:shadow-md transition-shadow ${modalLang === 'ar' ? 'flex-row-reverse text-right' : 'text-left'}`}
                  >
                    <div className="w-6 h-6 rounded-full bg-amber-500 flex items-center justify-center flex-shrink-0">
                      <span className="text-white text-xs font-bold">{i + 1}</span>
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-gray-800 dark:text-slate-100">{modalLang === 'ar' ? (s.termAr || s.term) : s.term}</p>
                      {s.icd11 && (
                        <span className="inline-block mt-1 text-xs text-gray-500 dark:text-slate-400 font-mono bg-white dark:bg-slate-850 px-2 py-1 rounded border border-gray-100 dark:border-slate-800">{s.icd11}</span>
                      )}
                      {s.severity && (
                        <div className={`mt-2 flex items-center gap-2 ${modalLang === 'ar' ? 'flex-row-reverse' : ''}`}>
                          <span className="text-xs text-gray-600 dark:text-slate-400">{modalLang === 'ar' ? 'الشدة:' : 'Severity:'}</span>
                          <div className="w-16 h-1.5 bg-gray-200 dark:bg-slate-800 rounded-full overflow-hidden">
                            <motion.div
                              className="h-full bg-gradient-to-r from-orange-500 to-red-500"
                              initial={{ width: 0 }}
                              animate={{ width: `${(s.severity / 10) * 100}%` }}
                              transition={{ delay: 0.3, duration: 0.5 }}
                            />
                          </div>
                          <span className="text-xs text-gray-600 dark:text-slate-400">{s.severity}/10</span>
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* Disclaimer Section */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="p-4 bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-slate-900/60 dark:to-slate-950/60 rounded-xl border border-blue-200 dark:border-slate-800"
            >
              <div className={`flex items-start gap-2 ${modalLang === 'ar' ? 'flex-row-reverse text-right' : 'text-left'}`}>
                <FileText size={16} className="text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
                <div className="flex-1" dir={modalLang === 'ar' ? 'rtl' : 'ltr'}>
                  <p className="text-xs font-semibold text-gray-600 dark:text-slate-400 mb-1">
                    {modalLang === 'ar' ? 'تنبيه هام' : 'Important Notice'}
                  </p>
                  <p className="text-xs text-gray-600 dark:text-slate-400 leading-relaxed">
                    {modalLang === 'ar' ? (selected.disclaimerAr || selected.disclaimer) : selected.disclaimer}
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Session Info */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="grid grid-cols-2 gap-3"
            >
              <div className="p-3 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-950/10 dark:to-pink-950/10 rounded-lg border border-purple-100 dark:border-purple-900/20">
                <p className="text-xs text-gray-600 dark:text-slate-405 font-medium mb-1">{modalLang === 'ar' ? 'معرف التقرير' : 'Report ID'}</p>
                <p className="text-sm font-mono text-purple-700 dark:text-purple-400">{selected.reportId.slice(0, 8)}</p>
              </div>
              <div className="p-3 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950/10 dark:to-emerald-950/10 rounded-lg border border-green-100 dark:border-green-900/20">
                <p className="text-xs text-gray-600 dark:text-slate-405 font-medium mb-1">{modalLang === 'ar' ? 'معرف الجلسة' : 'Session ID'}</p>
                <p className="text-sm font-mono text-green-700 dark:text-green-400">{selected.sessionId.slice(0, 8)}</p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </Modal>
    </div>
  )
}
