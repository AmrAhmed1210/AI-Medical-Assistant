import { useLanguage } from '@/lib/language'
import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  User,
  Stethoscope,
  Heart,
  AlertTriangle,
  Save,
  Lock,
  ChevronLeft,
  Plus,
  X,
  Activity,
  Thermometer,
  Droplets,
  Gauge,
  Weight,
  Wind,
  Pill,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  Wand2,
  Loader2,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { visitApi, type VitalSignDto, type SymptomDto, type PrescriptionDto } from '@/api/visitApi'
import { useVisit, usePatientHistory } from '@/hooks/useVisits'
import { getAiReportText, parseAiDiagnosisSummary } from '@/lib/aiReport'
import { Card, Button, SkeletonCard } from '@/components/ui'

const normalRanges: Record<string, { min: number; max: number; unit: string }> = {
  bp_systolic: { min: 90, max: 120, unit: 'mmHg' },
  bp_diastolic: { min: 60, max: 80, unit: 'mmHg' },
  blood_sugar: { min: 70, max: 100, unit: 'mg/dL' },
  heart_rate: { min: 60, max: 100, unit: 'bpm' },
  temperature: { min: 36.1, max: 37.2, unit: '°C' },
  spo2: { min: 95, max: 100, unit: '%' },
}

interface WorkspaceFormData {
  chiefComplaint: string
  historyOfIllness: string
  examinationFindings: string
  assessment: string
  plan: string
  notes: string
  bpSystolic: string
  bpDiastolic: string
  heartRate: string
  temperature: string
  bloodSugar: string
  weight: string
  spo2: string
  symptoms: SymptomDto[]
  prescriptions: PrescriptionDto[]
  followUpRequired: boolean
  followUpDate: string
  followUpTime: string
  followUpNotes: string
}

export default function DoctorWorkspace() {
  const { t, isRTL } = useLanguage()
  const { visitId } = useParams<{ visitId: string }>()
  const navigate = useNavigate()
  const id = Number(visitId)

  const [isClosing, setIsClosing] = useState(false)
  const [showConfirmClose, setShowConfirmClose] = useState(false)
  const [criticalAlert, setCriticalAlert] = useState<string | null>(null)

  const [form, setForm] = useState<WorkspaceFormData>({
    chiefComplaint: '',
    historyOfIllness: '',
    examinationFindings: '',
    assessment: '',
    plan: '',
    notes: '',
    bpSystolic: '',
    bpDiastolic: '',
    heartRate: '',
    temperature: '',
    bloodSugar: '',
    weight: '',
    spo2: '',
    symptoms: [],
    prescriptions: [],
    followUpRequired: false,
    followUpDate: '',
    followUpTime: '',
    followUpNotes: '',
  })

  const { visit, isLoading } = useVisit(id)
  const { history: patientHistory } = usePatientHistory(visit?.patientId ?? 0)

  const [isSaving, setIsSaving] = useState(false)
  const [aiLang, setAiLang] = useState<'ar' | 'en'>('ar')
  const [visitLang, setVisitLang] = useState<'ar' | 'en'>('ar')
  const [isAssisting, setIsAssisting] = useState(false)

  const handleAiAssist = async () => {
    if (!form.chiefComplaint) {
      toast.error(t('errEnterComplaint'))
      return
    }

    setIsAssisting(true)
    try {
      const aiServerUrl = import.meta.env.VITE_AI_SERVER_URL || 'http://localhost:8000'
      const response = await fetch(`${aiServerUrl}/doctor-ai-assist`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-internal-token': 'LuxuryMedicalAiSecretKey2026'
        },
        body: JSON.stringify({
          chief_complaint: form.chiefComplaint,
          history_of_illness: form.historyOfIllness,
          vitals: {
            bp_systolic: form.bpSystolic,
            bp_diastolic: form.bpDiastolic,
            heart_rate: form.heartRate,
            sugar: form.bloodSugar
          },
          // Send patient background for better AI context
          background: {
            allergies: patientHistory?.allergies?.map(a => a.allergenName).join(', '),
            chronic_diseases: patientHistory?.chronicDiseases?.map(d => d.diseaseName).join(', '),
            bloodType: patientHistory?.bloodType,
          }
        })
      })
      const data = await response.json()
      if (data.error) throw new Error(data.error)

      setForm(f => ({
        ...f,
        assessment: f.assessment ? f.assessment + '\n' + data.assessment_ar : data.assessment_ar,
        plan: f.plan ? f.plan + '\n' + data.plan_ar : data.plan_ar
      }))
      toast.success(t('aiSuggestionsGenerated'))
    } catch (err) {
      console.error(err)
      toast.error(t('errAiAssist'))
    } finally {
      setIsAssisting(false)
    }
  }

  const handleSaveDraft = async () => {
    setIsSaving(true)
    try {
      await visitApi.updateVisit(id, {
        chiefComplaint: form.chiefComplaint,
        examinationFindings: form.examinationFindings,
        assessment: form.assessment,
        plan: form.plan,
        notes: form.notes,
        symptoms: form.symptoms,
        prescriptions: form.prescriptions,
        followUpRequired: form.followUpRequired,
        followUpDate: form.followUpDate,
        followUpTime: form.followUpTime,
        followUpNotes: form.followUpNotes,
      })
      toast.success(t('draftSaved'))
    } catch {
      toast.error(t('errSaveDraft'))
    } finally {
      setIsSaving(false)
    }
  }

  const handleCloseVisit = async () => {
    setIsClosing(true)
    try {
      await visitApi.closeVisit(id)
      toast.success(t('visitClosed'))
      navigate(`/doctor/visits/${id}/summary`)
    } catch {
      toast.error(t('errCloseVisit'))
      setIsClosing(false)
    }
  }

  useEffect(() => {
    if (visit?.status === 'closed') {
      navigate(`/doctor/visits/${id}/summary`)
    }
  }, [visit, id, navigate])

  // Check critical values
  useEffect(() => {
    const sys = Number(form.bpSystolic)
    const dia = Number(form.bpDiastolic)
    const sugar = Number(form.bloodSugar)

    if (sys > 180 || dia > 120) {
      setCriticalAlert('⚠ Critical Value — Extremely High Blood Pressure — Immediate Intervention Required')
    } else if (sugar > 200) {
      setCriticalAlert('⚠ Critical Value — Extremely High Blood Sugar — Immediate Intervention Required')
    } else {
      setCriticalAlert(null)
    }
  }, [form.bpSystolic, form.bpDiastolic, form.bloodSugar])

  const handleAddSymptom = () => {
    setForm((f) => ({
      ...f,
      symptoms: [
        ...f.symptoms,
        { name: '', severity: 'moderate', onset: 'sudden', location: '', duration: '', isChronic: false },
      ],
    }))
  }

  const handleAddPrescription = () => {
    setForm((f) => ({
      ...f,
      prescriptions: [
        ...f.prescriptions,
        { medicationName: '', dosage: '', frequency: '', duration: '', quantity: 0, instructions: '', isChronic: false },
      ],
    }))
  }

  const getVitalStatus = (type: string, value: string) => {
    const num = Number(value)
    if (!value || isNaN(num)) return 'neutral'
    const range = normalRanges[type]
    if (!range) return 'neutral'
    if (num < range.min || num > range.max) return 'abnormal'
    return 'normal'
  }

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <SkeletonCard count={6} />
      </div>
    )
  }

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col gap-6 relative">
      {/* Decorative background */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-primary-400/5 rounded-full blur-3xl -z-10 pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-96 h-96 bg-blue-400/5 rounded-full blur-3xl -z-10 pointer-events-none" />

      {/* Header Card */}
      <div className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl rounded-3xl px-6 py-5 flex items-center justify-between shrink-0 border border-white/50 dark:border-slate-800/80 shadow-sm">
        <div className="flex items-center gap-5">
          <button 
            onClick={() => navigate('/doctor/today')}
            className="w-10 h-10 rounded-full flex items-center justify-center bg-slate-100/80 hover:bg-slate-200/80 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 transition-colors"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white tracking-tight">
                {visit?.patientName}
              </h1>
              <span className={`px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider ${
                visit?.status === 'open' 
                  ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400' 
                  : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400'
              }`}>
                {visit?.status === 'open' ? t('inProgress') : t('closed')}
              </span>
            </div>
            <p className="text-sm text-slate-500 dark:text-slate-400 font-medium flex items-center gap-2 mt-0.5">
              <Stethoscope className="w-3.5 h-3.5" />
              {t('visitWorkspace')}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button 
            onClick={handleSaveDraft} 
            disabled={isSaving}
            className="flex items-center gap-2 px-5 py-2.5 rounded-2xl font-bold text-slate-700 bg-white border border-slate-200 shadow-sm hover:bg-slate-50 dark:bg-slate-800 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-700 transition-all disabled:opacity-50"
          >
            <Save className="w-4 h-4 text-slate-400" />
            {isSaving ? t('saving') : t('saveDraft')}
          </button>
          <button
            onClick={() => setShowConfirmClose(true)}
            disabled={!form.chiefComplaint}
            className="flex items-center gap-2 px-6 py-2.5 rounded-2xl font-bold text-white bg-gradient-primary shadow-lg shadow-primary-500/25 hover:shadow-primary-500/40 hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:hover:translate-y-0"
          >
            <Lock className="w-4 h-4" />
            {t('finishAndClose')}
          </button>
        </div>
      </div>

      {/* Critical Alert Banner */}
      <AnimatePresence>
        {criticalAlert && (
          <motion.div
            initial={{ height: 0, opacity: 0, scale: 0.95 }}
            animate={{ height: 'auto', opacity: 1, scale: 1 }}
            exit={{ height: 0, opacity: 0, scale: 0.95 }}
            className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-900/50 rounded-2xl px-6 py-4 flex items-center gap-3 shadow-sm"
          >
            <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-900/50 flex items-center justify-center shrink-0">
              <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400" />
            </div>
            <div>
              <h4 className="text-red-800 dark:text-red-400 font-bold text-sm">{t('criticalAlertHeader')}</h4>
              <p className="text-red-600 dark:text-red-300 text-sm font-medium">{criticalAlert}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Split View */}
      <div className="flex-1 flex flex-col lg:flex-row gap-6 min-h-0">
        
        {/* Left Panel — Patient History (35%) */}
        <div className="w-full lg:w-[35%] bg-white/60 dark:bg-slate-900/60 backdrop-blur-xl rounded-3xl border border-white/50 dark:border-slate-800/80 shadow-sm overflow-hidden flex flex-col">
          <div className="px-6 py-4 border-b border-slate-100 dark:border-slate-800/60 bg-white/50 dark:bg-slate-900/50 shrink-0">
            <h2 className="font-extrabold text-lg text-slate-800 dark:text-slate-200 flex items-center gap-2">
              <User className="w-5 h-5 text-slate-400" />
              {t('patientProfile')}
            </h2>
          </div>
          
          <div className="overflow-y-auto p-6 space-y-8 custom-scrollbar">
            
            {/* AI Health Report */}
            {patientHistory?.aiDiagnosisSummary && (
              <div className="bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-2xl p-5 border border-purple-100/50 dark:border-purple-800/30 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4 opacity-10 pointer-events-none">
                  <Sparkles className="w-24 h-24 text-purple-600" />
                </div>
                <div className="flex items-center justify-between mb-3 relative z-10">
                  <h3 className="font-bold text-purple-900 dark:text-purple-300 flex items-center gap-2 text-sm uppercase tracking-wider">
                    <Wand2 className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                    {t('aiHealthReport')}
                  </h3>
                  <div className="flex bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm rounded-lg p-1 shadow-sm">
                    <button
                      onClick={() => setAiLang('ar')}
                      className={`px-2.5 py-1 text-[10px] font-bold rounded-md transition-all ${aiLang === 'ar' ? 'bg-purple-600 text-white shadow-md' : 'text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/30'}`}
                    >AR</button>
                    <button
                      onClick={() => setAiLang('en')}
                      className={`px-2.5 py-1 text-[10px] font-bold rounded-md transition-all ${aiLang === 'en' ? 'bg-purple-600 text-white shadow-md' : 'text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/30'}`}
                    >EN</button>
                  </div>
                </div>
                <div className="text-sm text-purple-800 dark:text-purple-200 leading-relaxed font-medium relative z-10" dir={aiLang === 'ar' ? 'rtl' : 'ltr'}>
                  {(() => {
                    const parsed = parseAiDiagnosisSummary(patientHistory.aiDiagnosisSummary)
                    if (!parsed) return patientHistory.aiDiagnosisSummary
                    return getAiReportText(parsed, aiLang)
                  })()}
                </div>
              </div>
            )}

            {/* Emergency Info */}
            <div className="space-y-4">
              <h3 className="text-xs font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest flex items-center gap-2">
                <Heart className="w-3.5 h-3.5 text-red-400" />
                {t('vitalDetails')}
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-50 dark:bg-slate-800/50 rounded-2xl p-4 border border-slate-100 dark:border-slate-700/50">
                  <p className="text-xs text-slate-500 font-medium mb-1">{t('bloodType')}</p>
                  <p className="font-extrabold text-red-600 dark:text-red-400 text-lg">{patientHistory?.bloodType || t('unknown')}</p>
                </div>
                <div className="bg-slate-50 dark:bg-slate-800/50 rounded-2xl p-4 border border-slate-100 dark:border-slate-700/50">
                  <p className="text-xs text-slate-500 font-medium mb-1">{t('allergies')}</p>
                  <p className="font-bold text-slate-800 dark:text-slate-200">
                    {patientHistory?.allergies?.length ? `${patientHistory.allergies.length} Recorded` : t('none')}
                  </p>
                </div>
              </div>
              {(patientHistory?.allergies?.filter((a: Record<string, string>) => a.severity === 'life_threatening').length ?? 0) > 0 && (
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-100 dark:border-red-900/50 rounded-xl p-3 flex gap-2">
                  <AlertTriangle className="w-4 h-4 text-red-600 dark:text-red-400 shrink-0 mt-0.5" />
                  <p className="text-xs text-red-700 dark:text-red-300 font-medium">
                    {t('lifeThreatening')} {patientHistory?.allergies
                      ?.filter((a: Record<string, string>) => a.severity === 'life_threatening')
                      .map((a: Record<string, string>) => a.allergenName)
                      .join(', ')}
                  </p>
                </div>
              )}
            </div>

            {/* Medical Context Lists */}
            <div className="space-y-6">
              {/* Chronic Diseases */}
              {(patientHistory?.chronicDiseases?.length ?? 0) > 0 && (
                <div>
                  <h3 className="text-xs font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest flex items-center gap-2 mb-3">
                    <Activity className="w-3.5 h-3.5" />
                    {t('chronicConditions')}
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {patientHistory?.chronicDiseases?.map((d: Record<string, string>) => (
                      <div key={d.id} className="bg-orange-50 dark:bg-orange-900/20 text-orange-700 dark:text-orange-300 px-3 py-1.5 rounded-lg text-xs font-bold border border-orange-100/50 dark:border-orange-800/30">
                        {d.diseaseName}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Current Medications */}
              {(patientHistory?.medications?.length ?? 0) > 0 && (
                <div>
                  <h3 className="text-xs font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest flex items-center gap-2 mb-3">
                    <Pill className="w-3.5 h-3.5" />
                    {t('activeMedications')}
                  </h3>
                  <div className="space-y-2">
                    {patientHistory?.medications?.map((m: Record<string, string>) => (
                      <div key={m.id} className="flex justify-between items-center p-3 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-100 dark:border-slate-700/50">
                        <span className="font-bold text-sm text-slate-700 dark:text-slate-300">{m.medicationName}</span>
                        <span className="text-xs text-slate-500 bg-white dark:bg-slate-800 px-2 py-1 rounded-md shadow-sm border border-slate-100 dark:border-slate-700">{m.dosage}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recent Visits Timeline */}
              {(patientHistory?.lastVisits?.length ?? 0) > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-3 gap-2">
                    <h3 className="text-xs font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest">
                      {visitLang === 'ar' ? 'ملخص الزيارات (آخر 8 أشهر)' : 'Recent Visits (Last 8 Months)'}
                    </h3>
                    <div className="flex bg-slate-100 dark:bg-slate-800 rounded-lg p-1">
                      <button
                        type="button"
                        onClick={() => setVisitLang('ar')}
                        className={`px-2 py-1 text-[10px] font-bold rounded-md transition-all ${visitLang === 'ar' ? 'bg-white dark:bg-slate-700 text-emerald-600 shadow-sm' : 'text-slate-500'}`}
                      >
                        AR
                      </button>
                      <button
                        type="button"
                        onClick={() => setVisitLang('en')}
                        className={`px-2 py-1 text-[10px] font-bold rounded-md transition-all ${visitLang === 'en' ? 'bg-white dark:bg-slate-700 text-emerald-600 shadow-sm' : 'text-slate-500'}`}
                      >
                        EN
                      </button>
                    </div>
                  </div>
                  <div className="relative pl-3 space-y-4 before:absolute before:inset-y-0 before:left-[5px] before:w-px before:bg-slate-200 dark:before:bg-slate-700">
                    {patientHistory?.lastVisits?.map((v: Record<string, string>, i: number) => (
                      <div 
                        key={v.id} 
                        className="relative pl-4 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 p-2 -ml-1 rounded-lg transition-colors"
                        onClick={() => window.open(`/doctor/visits/${v.id}/summary`, '_blank')}
                      >
                        <div className="absolute left-[-1px] top-3 w-2 h-2 rounded-full bg-slate-300 dark:bg-slate-600 ring-4 ring-white dark:ring-slate-900" />
                        <p className="text-xs font-bold text-emerald-600 dark:text-emerald-400 mb-0.5 hover:underline">{v.visitDate} <span className="text-[10px] text-slate-400 font-normal ml-1">{t('clickForSummary')}</span></p>
                        {v.doctorName && (
                          <p className="text-[11px] font-semibold text-primary-600 dark:text-primary-400">
                            Dr. {v.doctorName}
                            {v.doctorSpecialty && <span className="text-slate-400 font-normal ml-1">· {v.doctorSpecialty}</span>}
                          </p>
                        )}
                        <p className="text-sm font-medium text-slate-700 dark:text-slate-300 line-clamp-2" dir="auto">{v.chiefComplaint}</p>
                        {(visitLang === 'ar' ? (v.summaryAr || v.summary) : (v.summaryEn || v.summary)) && (
                          <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-1 line-clamp-3 italic" dir={visitLang === 'ar' ? 'rtl' : 'ltr'}>
                            {visitLang === 'ar' ? (v.summaryAr || v.summary) : (v.summaryEn || v.summary)}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

          </div>
        </div>

        {/* Right Panel — Current Visit Document (65%) */}
        <div className="w-full lg:w-[65%] bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-white/50 dark:border-slate-800/80 shadow-[0_8px_30px_rgb(0,0,0,0.04)] overflow-hidden flex flex-col relative">
          
          <div className="overflow-y-auto p-6 md:p-8 space-y-10 custom-scrollbar">
            
            {/* Subjective Section */}
            <section className="space-y-6">
              <div className="flex items-center gap-3 border-b border-slate-100 dark:border-slate-800/60 pb-2">
                <div className="w-8 h-8 rounded-full bg-primary-50 dark:bg-primary-900/30 flex items-center justify-center text-primary-600 dark:text-primary-400 font-black text-sm">S</div>
                <h2 className="text-lg font-extrabold text-slate-800 dark:text-slate-200">{t('subjective')}</h2>
              </div>
              
              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-2">Chief Complaint <span className="text-red-500">*</span></label>
                  <textarea
                    className="w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-200/60 dark:border-slate-700/50 rounded-2xl p-4 text-sm text-slate-800 dark:text-slate-200 focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none transition-all placeholder:text-slate-400 resize-none shadow-inner"
                    rows={2}
                    value={form.chiefComplaint}
                    onChange={(e) => setForm((f) => ({ ...f, chiefComplaint: e.target.value }))}
                    placeholder="E.g., Patient presents with chest pain and shortness of breath..."
                  />
                </div>
                <div>
                  <label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-2">{t('hpi')}</label>
                  <textarea
                    className="w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-200/60 dark:border-slate-700/50 rounded-2xl p-4 text-sm text-slate-800 dark:text-slate-200 focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none transition-all placeholder:text-slate-400 resize-none shadow-inner"
                    rows={3}
                    value={form.historyOfIllness}
                    onChange={(e) => setForm((f) => ({ ...f, historyOfIllness: e.target.value }))}
                    placeholder="Details about onset, duration, character, aggravating/relieving factors..."
                  />
                </div>
              </div>
            </section>

            {/* Objective Section */}
            <section className="space-y-6">
              <div className="flex items-center gap-3 border-b border-slate-100 dark:border-slate-800/60 pb-2">
                <div className="w-8 h-8 rounded-full bg-emerald-50 dark:bg-emerald-900/30 flex items-center justify-center text-emerald-600 dark:text-emerald-400 font-black text-sm">O</div>
                <h2 className="text-lg font-extrabold text-slate-800 dark:text-slate-200">{t('objectiveVitals')}</h2>
              </div>

              <div className="bg-slate-50/50 dark:bg-slate-800/20 border border-slate-100 dark:border-slate-800/50 rounded-3xl p-5">
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  {[
                    { key: 'bpSystolic' as const, label: t('sysBp'), icon: Gauge, type: 'bp_systolic', placeholder: '120' },
                    { key: 'bpDiastolic' as const, label: t('diaBp'), icon: Gauge, type: 'bp_diastolic', placeholder: '80' },
                    { key: 'heartRate' as const, label: t('heartRate'), icon: Heart, type: 'heart_rate', placeholder: '72' },
                    { key: 'temperature' as const, label: t('temp'), icon: Thermometer, type: 'temperature', placeholder: '37.0' },
                    { key: 'bloodSugar' as const, label: t('sugar'), icon: Droplets, type: 'blood_sugar', placeholder: '90' },
                    { key: 'weight' as const, label: t('weightL'), icon: Weight, type: '', placeholder: '70' },
                    { key: 'spo2' as const, label: t('spo2'), icon: Wind, type: 'spo2', placeholder: '98' },
                  ].map((field) => {
                    const status = field.type ? getVitalStatus(field.type, form[field.key]) : 'neutral'
                    const Icon = field.icon
                    return (
                      <div key={field.key} className="relative group">
                        <label className="text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
                          <Icon className="w-3 h-3" />
                          {field.label}
                        </label>
                        <div className="relative">
                          <input
                            type="text"
                            inputMode="decimal"
                            className={`w-full bg-white dark:bg-slate-900 border rounded-xl p-2.5 text-sm font-bold text-slate-800 dark:text-slate-200 outline-none transition-all placeholder:text-slate-300 dark:placeholder:text-slate-600 shadow-sm ${
                              status === 'abnormal'
                                ? 'border-red-300 dark:border-red-800 focus:ring-2 focus:ring-red-500/20 focus:border-red-500'
                                : status === 'normal'
                                  ? 'border-emerald-200 dark:border-emerald-800 focus:border-emerald-400'
                                  : 'border-slate-200 dark:border-slate-700 focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500'
                            }`}
                            placeholder={field.placeholder}
                            value={form[field.key]}
                            onChange={(e) => setForm((f) => ({ ...f, [field.key]: e.target.value }))}
                          />
                          {status === 'normal' && <CheckCircle2 className="w-4 h-4 text-emerald-500 absolute right-3 top-3 opacity-0 group-hover:opacity-100 transition-opacity" />}
                          {status === 'abnormal' && <AlertCircle className="w-4 h-4 text-red-500 absolute right-3 top-3" />}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              <div>
                <label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-2">{t('examinationFindings')}</label>
                <textarea
                  className="w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-200/60 dark:border-slate-700/50 rounded-2xl p-4 text-sm text-slate-800 dark:text-slate-200 focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none transition-all placeholder:text-slate-400 resize-none shadow-inner"
                  rows={3}
                  value={form.examinationFindings}
                  onChange={(e) => setForm((f) => ({ ...f, examinationFindings: e.target.value }))}
                  placeholder="Clinical observations, palpation, auscultation results..."
                />
              </div>
            </section>

            {/* Assessment Section */}
            <section className="space-y-6">
              <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800/60 pb-2">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-purple-50 dark:bg-purple-900/30 flex items-center justify-center text-purple-600 dark:text-purple-400 font-black text-sm">A</div>
                  <h2 className="text-lg font-extrabold text-slate-800 dark:text-slate-200">{t('assessment')}</h2>
                </div>
                <button
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold text-purple-600 bg-purple-50 hover:bg-purple-100 dark:text-purple-400 dark:bg-purple-900/20 dark:hover:bg-purple-900/40 transition-colors disabled:opacity-50"
                  onClick={handleAiAssist}
                  disabled={isAssisting}
                >
                  {isAssisting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Wand2 className="w-3.5 h-3.5" />}
                  {t('aiAnalyze')}
                </button>
              </div>

              <div>
                <textarea
                  className="w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-200/60 dark:border-slate-700/50 rounded-2xl p-4 text-sm text-slate-800 dark:text-slate-200 focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none transition-all placeholder:text-slate-400 resize-none shadow-inner"
                  rows={3}
                  value={form.assessment}
                  onChange={(e) => setForm((f) => ({ ...f, assessment: e.target.value }))}
                  placeholder="Primary diagnosis and differential diagnosis..."
                />
              </div>
            </section>

            {/* Plan Section */}
            <section className="space-y-6">
              <div className="flex items-center gap-3 border-b border-slate-100 dark:border-slate-800/60 pb-2">
                <div className="w-8 h-8 rounded-full bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center text-blue-600 dark:text-blue-400 font-black text-sm">P</div>
                <h2 className="text-lg font-extrabold text-slate-800 dark:text-slate-200">{t('planPrescriptions')}</h2>
              </div>

              <div>
                <label className="block text-sm font-bold text-slate-700 dark:text-slate-300 mb-2">{t('treatmentPlan')}</label>
                <textarea
                  className="w-full bg-slate-50 dark:bg-slate-900/50 border border-slate-200/60 dark:border-slate-700/50 rounded-2xl p-4 text-sm text-slate-800 dark:text-slate-200 focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none transition-all placeholder:text-slate-400 resize-none shadow-inner"
                  rows={3}
                  value={form.plan}
                  onChange={(e) => setForm((f) => ({ ...f, plan: e.target.value }))}
                  placeholder="Patient instructions, lifestyle advice, next steps..."
                />
              </div>

              <div className="bg-slate-50/50 dark:bg-slate-800/20 border border-slate-100 dark:border-slate-800/50 rounded-3xl p-5">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-bold text-slate-700 dark:text-slate-300 text-sm flex items-center gap-2">
                    <Pill className="w-4 h-4 text-blue-500" />
                    {t('medicationsRx')}
                  </h3>
                  <button onClick={handleAddPrescription} className="text-xs font-bold text-blue-600 hover:text-blue-700 dark:text-blue-400 flex items-center gap-1">
                    <Plus className="w-3.5 h-3.5" /> {t('addDrug')}
                  </button>
                </div>
                
                <div className="space-y-3">
                  {form.prescriptions.map((pres, idx) => (
                    <div key={idx} className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl p-4 shadow-sm relative group">
                      <button
                        className="absolute -right-2 -top-2 w-6 h-6 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-full flex items-center justify-center text-red-500 opacity-0 group-hover:opacity-100 transition-opacity shadow-sm hover:bg-red-50"
                        onClick={() => {
                          const p = form.prescriptions.filter((_, i) => i !== idx)
                          setForm((f) => ({ ...f, prescriptions: p }))
                        }}
                      >
                        <X className="w-3 h-3" />
                      </button>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <input
                          className="bg-transparent text-sm font-bold text-slate-800 dark:text-slate-200 outline-none placeholder:font-medium placeholder:text-slate-400"
                          placeholder="Drug name"
                          value={pres.medicationName}
                          onChange={(e) => {
                            const p = [...form.prescriptions]
                            p[idx].medicationName = e.target.value
                            setForm((f) => ({ ...f, prescriptions: p }))
                          }}
                        />
                        <input
                          className="bg-transparent text-sm text-slate-600 dark:text-slate-400 outline-none placeholder:text-slate-400"
                          placeholder="Dosage (e.g. 500mg)"
                          value={pres.dosage}
                          onChange={(e) => {
                            const p = [...form.prescriptions]
                            p[idx].dosage = e.target.value
                            setForm((f) => ({ ...f, prescriptions: p }))
                          }}
                        />
                        <input
                          className="bg-transparent text-sm text-slate-600 dark:text-slate-400 outline-none placeholder:text-slate-400"
                          placeholder="Frequency (e.g. 1x12h)"
                          value={pres.frequency}
                          onChange={(e) => {
                            const p = [...form.prescriptions]
                            p[idx].frequency = e.target.value
                            setForm((f) => ({ ...f, prescriptions: p }))
                          }}
                        />
                        <input
                          className="bg-transparent text-sm text-slate-600 dark:text-slate-400 outline-none placeholder:text-slate-400"
                          placeholder="Duration (e.g. 5 days)"
                          value={pres.duration}
                          onChange={(e) => {
                            const p = [...form.prescriptions]
                            p[idx].duration = e.target.value
                            setForm((f) => ({ ...f, prescriptions: p }))
                          }}
                        />
                      </div>
                    </div>
                  ))}
                  {form.prescriptions.length === 0 && (
                    <div className="text-center py-6 text-sm text-slate-400 font-medium border-2 border-dashed border-slate-200 dark:border-slate-700 rounded-2xl">
                      No medications prescribed yet.
                    </div>
                  )}
                </div>
              </div>
            </section>

            {/* Follow-up & Wrap up */}
            <section className="space-y-6 pt-4">
              <div className="bg-blue-50/50 dark:bg-blue-900/10 border border-blue-100 dark:border-blue-900/30 rounded-3xl p-5 flex flex-col md:flex-row gap-6">
                <div className="flex-1">
                  <label className="flex items-center gap-3 mb-4 cursor-pointer">
                    <div className={`w-5 h-5 rounded flex items-center justify-center border transition-colors ${
                      form.followUpRequired 
                        ? 'bg-blue-500 border-blue-500 text-white' 
                        : 'bg-white border-slate-300 dark:bg-slate-800 dark:border-slate-600'
                    }`}>
                      {form.followUpRequired && <CheckCircle2 className="w-3.5 h-3.5" />}
                    </div>
                    <input
                      type="checkbox"
                      className="hidden"
                      checked={form.followUpRequired}
                      onChange={(e) => setForm((f) => ({ ...f, followUpRequired: e.target.checked }))}
                    />
                    <span className="font-bold text-slate-800 dark:text-slate-200">{t('scheduleFollowUp')}</span>
                  </label>
                  
                  <AnimatePresence>
                    {form.followUpRequired && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="space-y-3 overflow-hidden">
                        <div className="flex flex-wrap items-center gap-3">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-slate-600 dark:text-slate-400">{t('date')}</span>
                            <input
                              type="date"
                              min={new Date(new Date().setDate(new Date().getDate() + 1)).toISOString().split('T')[0]}
                              className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-2 text-sm font-bold text-slate-800 dark:text-slate-200 outline-none"
                              value={form.followUpDate}
                              onChange={(e) => setForm((f) => ({ ...f, followUpDate: e.target.value }))}
                            />
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-slate-600 dark:text-slate-400">{t('time')}</span>
                            <input
                              type="time"
                              className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-2 text-sm font-bold text-slate-800 dark:text-slate-200 outline-none"
                              value={form.followUpTime}
                              onChange={(e) => setForm((f) => ({ ...f, followUpTime: e.target.value }))}
                            />
                          </div>
                        </div>
                        <textarea
                          className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl p-3 text-sm text-slate-800 dark:text-slate-200 outline-none resize-none"
                          rows={2}
                          placeholder="Notes for the receptionist/patient..."
                          value={form.followUpNotes}
                          onChange={(e) => setForm((f) => ({ ...f, followUpNotes: e.target.value }))}
                        />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </section>

          </div>
        </div>
      </div>

      {/* Confirm Close Modal */}
      <AnimatePresence>
        {showConfirmClose && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0, y: 10 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 10 }}
              className="bg-white dark:bg-slate-900 rounded-3xl p-8 max-w-sm w-full shadow-2xl border border-slate-100 dark:border-slate-800"
            >
              <div className="w-12 h-12 rounded-full bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center mb-5">
                <Lock className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="font-extrabold text-xl text-slate-900 dark:text-white mb-2">{t('lockAndCloseVisit')}</h3>
              <p className="text-slate-500 dark:text-slate-400 text-sm mb-8 leading-relaxed">
                This will finalize the medical record. You won't be able to edit these notes afterwards.
              </p>
              <div className="flex gap-3">
                <button 
                  className="flex-1 px-4 py-3 rounded-2xl font-bold text-slate-600 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700 transition-colors"
                  onClick={() => setShowConfirmClose(false)}
                >
                  {t('goBack')}
                </button>
                <button
                  className="flex-1 px-4 py-3 rounded-2xl font-bold text-white bg-blue-600 hover:bg-blue-700 transition-colors disabled:opacity-50"
                  onClick={() => {
                    setShowConfirmClose(false)
                    handleCloseVisit()
                  }}
                  disabled={isClosing}
                >
                  {isClosing ? 'Closing...' : 'Confirm'}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
