import { useLanguage } from '@/lib/language'
import { useParams, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  ChevronLeft,
  Download,
  Printer,
  FileText,
  Pill,
  Activity,
  Stethoscope,
  AlertTriangle,
  CheckCircle2,
  Calendar,
  Clock,
  User,
} from 'lucide-react'
import { visitApi } from '@/api/visitApi'
import { useVisitSummary } from '@/hooks/useVisits'
import { Card, Button, SkeletonCard } from '@/components/ui'

export default function DoctorVisitSummary() {
  const { t, isRTL } = useLanguage()
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const visitId = Number(id)

  const { summary, isLoading } = useVisitSummary(visitId)
  const [reportLang, setReportLang] = useState<'en' | 'ar'>('en')

  const handleDownloadPdf = async () => {
    try {
      const blob = await visitApi.downloadPdf(visitId)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `visit-summary-${visitId}.pdf`
      a.click()
      window.URL.revokeObjectURL(url)
    } catch {
      // ignore
    }
  }

  const handlePrint = () => {
    window.print()
  }

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <SkeletonCard count={4} />
      </div>
    )
  }

  if (!summary) {
    return (
      <div className="max-w-4xl mx-auto p-6 text-center">
        <FileText className="w-12 h-12 text-gray-300 mx-auto mb-4" />
        <p className="text-gray-500">{t('visitSummaryNotFound')}</p>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto p-4 sm:p-8 space-y-8 print:p-0 min-h-screen pb-20">
      {/* Actions bar */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between print:hidden bg-white/80 p-4 rounded-2xl border border-gray-200 shadow-sm backdrop-blur-md sticky top-4 z-10"
      >
        <Button variant="ghost" className="hover:bg-gray-100" onClick={() => navigate('/doctor/today')}>
          <ChevronLeft className="w-4 h-4 mr-2" />
          {t('back')}
        </Button>
        <div className="flex items-center gap-3">
          <Button variant="outline" className="bg-white shadow-sm border-gray-200 hover:bg-gray-50" onClick={handlePrint}>
            <Printer className="w-4 h-4 mr-2" />
            {t('printReport')}
          </Button>
          <Button onClick={handleDownloadPdf} className="shadow-sm">
            <Download className="w-4 h-4 mr-2" />
            {t('downloadPdf')}
          </Button>
        </div>
      </motion.div>

      {/* Report Container */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-[2rem] border border-gray-200 shadow-xl shadow-gray-200/40 overflow-hidden print:border-none print:shadow-none"
      >
        {/* Header Ribbon */}
        <div className="h-4 bg-gradient-to-r from-primary-600 via-blue-500 to-teal-400 w-full" />
        
        <div className="p-8 sm:p-12">
          {/* Header */}
          <div className="flex flex-col sm:flex-row justify-between items-start gap-6 border-b border-gray-100 pb-8 mb-8">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 bg-gradient-to-br from-primary-50 to-blue-50 rounded-2xl flex items-center justify-center border border-primary-100 shadow-inner">
                <FileText className="w-7 h-7 text-primary-600" />
              </div>
              <div>
                <h1 className="text-3xl font-black text-gray-900 tracking-tight">{t('visitSummary')}</h1>
                <p className="text-gray-500 font-medium mt-1">{t('finalMedicalReport')}</p>
              </div>
            </div>
            <div className="text-left sm:text-right bg-gray-50 p-4 rounded-2xl border border-gray-100">
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">{t('visitDetails')}</p>
              <div className="flex flex-col gap-1 text-gray-900">
                <span className="flex items-center gap-2 font-semibold">
                  <Calendar className="w-4 h-4 text-primary-500" />
                  {summary.visitDate}
                </span>
                <span className="flex items-center gap-2 font-semibold">
                  <span className="text-gray-400">{t('idLabel')}</span>
                  #{summary.id}
                </span>
              </div>
            </div>
          </div>

          {/* Patient Info Card */}
          <div className="bg-gradient-to-br from-gray-50 to-white rounded-3xl p-6 sm:p-8 mb-10 border border-gray-200 shadow-sm relative overflow-hidden">
            {/* Decorative background element */}
            <div className="absolute -right-10 -top-10 w-40 h-40 bg-primary-50 rounded-full blur-3xl opacity-50 pointer-events-none" />
            
            <div className="relative flex items-center justify-between">
              <div className="flex items-center gap-5">
                <div className="w-20 h-20 bg-white shadow-md rounded-full flex items-center justify-center border border-gray-100 shrink-0">
                  <User className="w-10 h-10 text-primary-600" />
                </div>
                <div>
                  <h2 className="text-2xl font-black text-gray-900">{summary.patientName}</h2>
                  <div className="flex flex-wrap items-center gap-3 sm:gap-4 mt-2">
                    <span className="text-gray-600 font-medium bg-white px-3 py-1 rounded-lg border border-gray-100 shadow-sm">{summary.patientAge} {t('yearsOld')}</span>
                    <span className="flex items-center gap-2 text-gray-600 font-medium bg-white px-3 py-1 rounded-lg border border-gray-100 shadow-sm">
                      {t('bloodTypeLabel')} <span className="font-bold text-red-600">{summary.bloodType}</span>
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Allergies */}
            {summary.allergies?.length > 0 && (
              <div className="relative mt-8 pt-6 border-t border-gray-200/60">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-red-500 mt-0.5 shrink-0" />
                  <div>
                    <span className="text-xs font-black text-red-500 block mb-3 uppercase tracking-widest">{t('knownAllergies')}</span>
                    <div className="flex flex-wrap gap-2">
                      {summary.allergies.map((a, idx) => (
                        <span
                          key={idx}
                          className={`px-4 py-2 rounded-xl text-sm font-bold border shadow-sm ${a.severity === 'life_threatening'
                            ? 'bg-red-50 text-red-700 border-red-200'
                            : 'bg-orange-50 text-orange-700 border-orange-200'
                            }`}
                        >
                          {a.allergenName} <span className="opacity-75 font-medium ml-1">({a.reaction})</span>
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Past Visits Summary (Last 8 Months) */}
          {(summary.recentVisits?.length ?? 0) > 0 || summary.visitsTimelineSummaryEn || summary.visitsTimelineSummaryAr ? (
            <div className="mb-10">
              <div className="flex items-center justify-between mb-4 gap-4 flex-wrap">
                <h3 className="text-xs font-black text-gray-400 uppercase tracking-widest flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  {reportLang === 'ar' ? 'ملخص الزيارات (آخر 8 أشهر)' : 'Past Visits Summary (Last 8 Months)'}
                </h3>
                <div className="flex bg-gray-100 rounded-xl p-1">
                  <button
                    type="button"
                    onClick={() => setReportLang('en')}
                    className={`px-3 py-1 text-xs font-bold rounded-lg transition-all ${reportLang === 'en' ? 'bg-white text-primary-600 shadow-sm' : 'text-gray-500'}`}
                  >
                    EN
                  </button>
                  <button
                    type="button"
                    onClick={() => setReportLang('ar')}
                    className={`px-3 py-1 text-xs font-bold rounded-lg transition-all ${reportLang === 'ar' ? 'bg-white text-primary-600 shadow-sm' : 'text-gray-500'}`}
                  >
                    عربي
                  </button>
                </div>
              </div>

              <div className="bg-slate-50 border-2 border-slate-100 rounded-3xl p-6 mb-4" dir={reportLang === 'ar' ? 'rtl' : 'ltr'}>
                <p className="text-gray-800 leading-loose font-medium whitespace-pre-wrap">
                  {reportLang === 'ar'
                    ? (summary.visitsTimelineSummaryAr || 'لا توجد زيارات مسجلة في آخر 8 أشهر.')
                    : (summary.visitsTimelineSummaryEn || 'No visits recorded in the last 8 months.')}
                </p>
              </div>

              {summary.recentVisits && summary.recentVisits.length > 0 && (
                <div className="space-y-3">
                  {summary.recentVisits.map((visit) => (
                    <div key={visit.id} className="bg-white border border-gray-100 rounded-2xl p-4 shadow-sm">
                      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                        <p className="font-bold text-gray-900">{visit.visitDate}</p>
                        {visit.doctorName && (
                          <p className="text-sm text-primary-600 font-semibold">
                            Dr. {visit.doctorName}
                            {visit.doctorSpecialty ? ` · ${visit.doctorSpecialty}` : ''}
                          </p>
                        )}
                      </div>
                      <p className="text-sm text-gray-700 font-medium mb-2" dir="auto">{visit.chiefComplaint}</p>
                      {(reportLang === 'ar' ? (visit.summaryAr || visit.summary) : (visit.summaryEn || visit.summary)) && (
                        <p className="text-sm text-gray-500 leading-relaxed" dir={reportLang === 'ar' ? 'rtl' : 'ltr'}>
                          {reportLang === 'ar' ? (visit.summaryAr || visit.summary) : (visit.summaryEn || visit.summary)}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : null}

          {/* Clinical Details Grid */}
          <div className="grid lg:grid-cols-2 gap-8 mb-10">
            <div className="space-y-8">
              {/* Chief Complaint */}
              <div className="group">
                <h3 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-gray-400 group-hover:text-primary-500 transition-colors" /> Chief Complaint
                </h3>
                <div className="bg-white border-2 border-gray-100 rounded-3xl p-6 shadow-sm hover:border-primary-100 transition-colors" dir="auto">
                  <p className="text-gray-800 leading-loose text-lg font-medium">{summary.chiefComplaint || '—'}</p>
                </div>
              </div>

              {/* Assessment */}
              {summary.assessment && (
                <div className="group">
                  <h3 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-gray-400 group-hover:text-green-500 transition-colors" /> Assessment / Diagnosis
                  </h3>
                  <div className="bg-white border-2 border-gray-100 rounded-3xl p-6 shadow-sm hover:border-green-100 transition-colors" dir="auto">
                    <p className="text-gray-800 leading-loose font-medium">{summary.assessment}</p>
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-8">
              {/* Plan */}
              {summary.plan && (
                <div className="group">
                  <h3 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                    <Stethoscope className="w-4 h-4 text-gray-400 group-hover:text-blue-500 transition-colors" /> Treatment Plan
                  </h3>
                  <div className="bg-blue-50/50 border-2 border-blue-100 rounded-3xl p-6 shadow-sm hover:border-blue-200 transition-colors" dir="auto">
                    <p className="text-blue-900 leading-loose font-medium">{summary.plan}</p>
                  </div>
                </div>
              )}

              {/* Follow Up */}
              {summary.followUpRequired && (
                <div className="group">
                  <h3 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-gray-400 group-hover:text-purple-500 transition-colors" /> Follow-up Required
                  </h3>
                  <div className="bg-gradient-to-br from-purple-50 to-white border-2 border-purple-100 rounded-3xl p-6 shadow-sm hover:border-purple-200 transition-colors">
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-10 h-10 bg-purple-100 text-purple-600 rounded-xl flex items-center justify-center">
                        <Calendar className="w-5 h-5" />
                      </div>
                      <p className="text-gray-900 font-black text-xl">
                        {summary.followUpDate} <span className="text-purple-600 font-bold text-base ml-1">{summary.followUpTime && `at ${summary.followUpTime}`}</span>
                      </p>
                    </div>
                    {summary.followUpNotes && (
                      <p className="text-gray-600 mt-4 leading-relaxed font-medium bg-white p-4 rounded-xl border border-purple-50/50" dir="auto">{summary.followUpNotes}</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Vitals */}
          {summary.vitalSigns?.length > 0 && (
            <div className="mb-10">
              <h3 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                <Activity className="w-4 h-4" /> Vital Signs
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {summary.vitalSigns.map((vital, idx) => (
                  <div
                    key={idx}
                    className={`p-5 rounded-3xl border-2 ${vital.isAbnormal ? 'bg-red-50 border-red-100' : 'bg-white border-gray-100 shadow-sm'
                      }`}
                  >
                    <p className="text-sm font-bold text-gray-500 mb-2 uppercase tracking-wide">{vital.type}</p>
                    <div className="flex items-baseline gap-1">
                      <p className={`font-black text-3xl tracking-tighter ${vital.isAbnormal ? 'text-red-600' : 'text-gray-900'}`}>
                        {vital.value}
                        {vital.value2 && <span className="text-xl opacity-50">/{vital.value2}</span>}
                      </p>
                      <p className="text-sm font-bold text-gray-400 ml-1">{vital.unit}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Prescriptions */}
          {summary.prescriptions?.length > 0 && (
            <div>
              <h3 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                <Pill className="w-4 h-4" /> Prescribed Medications
              </h3>
              <div className="bg-white border-2 border-gray-100 rounded-3xl overflow-hidden shadow-sm">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-gray-50/80 border-b-2 border-gray-100">
                      <th className="py-5 px-6 font-black text-gray-400 text-xs uppercase tracking-widest">{t('medicationTh')}</th>
                      <th className="py-5 px-6 font-black text-gray-400 text-xs uppercase tracking-widest">{t('dosageFreqTh')}</th>
                      <th className="py-5 px-6 font-black text-gray-400 text-xs uppercase tracking-widest">{t('durationTh')}</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y-2 divide-gray-50">
                    {summary.prescriptions.map((pres, idx) => (
                      <tr key={idx} className="hover:bg-gray-50/50 transition-colors">
                        <td className="py-5 px-6">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-primary-50 rounded-xl flex items-center justify-center border border-primary-100 shrink-0">
                              <Pill className="w-5 h-5 text-primary-600" />
                            </div>
                            <div>
                              <p className="font-black text-gray-900 text-lg">{pres.medicationName}</p>
                              {pres.isChronic && (
                                <span className="inline-block mt-1 px-2.5 py-0.5 bg-teal-50 text-teal-700 rounded-md text-[10px] font-black uppercase tracking-widest border border-teal-200">
                                  Chronic
                                </span>
                              )}
                            </div>
                          </div>
                        </td>
                        <td className="py-5 px-6">
                          <p className="text-gray-900 font-bold">{pres.dosage}</p>
                          <p className="text-gray-500 text-sm font-medium mt-1">{pres.frequency}</p>
                        </td>
                        <td className="py-5 px-6">
                          <span className="bg-gray-100 text-gray-700 px-3 py-1.5 rounded-lg font-bold text-sm">
                            {pres.duration}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </motion.div>

      {/* Footer */}
      <div className="text-center pt-8 pb-4 print:mt-12 opacity-60 hover:opacity-100 transition-opacity">
        <div className="w-12 h-1 bg-gray-200 mx-auto rounded-full mb-6" />
        <p className="text-sm font-black text-gray-400 tracking-widest uppercase mb-2">MedBook Health Systems</p>
        <p className="text-xs font-medium text-gray-400">{t('electronicPatientMgmt')}</p>
        <p className="text-xs font-medium text-gray-400 mt-1">{t('summaryDisclaimer')}</p>
      </div>
    </div>
  )
}
