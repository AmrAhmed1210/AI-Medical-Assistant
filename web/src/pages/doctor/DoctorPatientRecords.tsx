import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronLeft, Plus, X, Trash2, User, Shield, Activity, Pill, Heart, FileText, Scissors, Loader2, Sparkles } from 'lucide-react'
import toast from 'react-hot-toast'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Modal } from '@/components/ui/Modal'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { patientRecordsApi, type AllergyRecord, type ChronicDiseaseRecord, type MedicationRecord, type VitalRecord, type SurgeryRecord, type PatientDocument } from '@/api/patientRecordsApi'
import axiosInstance from '@/api/axiosInstance'
import { doctorApi } from '@/api/doctorApi'
import { useDoctorPatients } from '@/hooks/useDoctor'
import { usePatientHistory } from '@/hooks/useVisits'
import { useLanguage } from '@/lib/language'
import { getAiReportText, hasDistinctArabicReport, parseAiDiagnosisSummary } from '@/lib/aiReport'
import { Lock, Clock } from 'lucide-react'

const TABS = [
  { id: 'profile', label: 'Profile', icon: User },
  { id: 'visits', label: 'Visits', icon: Clock },
  { id: 'allergies', label: 'Allergies', icon: Shield },
  { id: 'chronic', label: 'Chronic', icon: Activity },
  { id: 'medications', label: 'Medications', icon: Pill },
  { id: 'vitals', label: 'Vitals', icon: Heart },
  { id: 'surgeries', label: 'Surgeries', icon: Scissors },
  { id: 'scans', label: 'Scans / Labs', icon: FileText },
] as const

type TabId = typeof TABS[number]['id']

export default function DoctorPatientRecords() {
  const { t, isRTL } = useLanguage()
  const { patientId } = useParams<{ patientId: string }>()
  const navigate = useNavigate()
  const pid = patientId ?? '0'
  const { patients = [] } = useDoctorPatients('')
  const patient = patients.find(p => String(p.id) === pid)
  const { history: patientHistory } = usePatientHistory(Number(pid))
  const [expandedVisits, setExpandedVisits] = useState<Record<string, boolean>>({})

  const [activeTab, setActiveTab] = useState<TabId>('profile')
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [saving, setSaving] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [hasAccessToday, setHasAccessToday] = useState<boolean | null>(null)
  const [aiLang, setAiLang] = useState<'en' | 'ar'>('ar')

  const [allergies, setAllergies] = useState<AllergyRecord[]>([])
  const [chronic, setChronic] = useState<ChronicDiseaseRecord[]>([])
  const [medications, setMedications] = useState<MedicationRecord[]>([])
  const [vitals, setVitals] = useState<VitalRecord[]>([])
  const [surgeries, setSurgeries] = useState<SurgeryRecord[]>([])
  const [documents, setDocuments] = useState<PatientDocument[]>([])
  const [medProfile, setMedProfile] = useState<any>(null)

  const fetchAll = async () => {
    setLoading(true)
    try {
      const [a, c, m, v, s, d, mp, appts] = await Promise.all([
        patientRecordsApi.getAllergies(pid).catch(() => []),
        patientRecordsApi.getChronicDiseases(pid).catch(() => []),
        patientRecordsApi.getMedications(pid).catch(() => []),
        patientRecordsApi.getVitals(pid).catch(() => []),
        patientRecordsApi.getSurgeries(pid).catch(() => []),
        patientRecordsApi.getDocuments(pid).catch(() => []),
        patientRecordsApi.getMedicalProfile(pid).catch(() => null),
        doctorApi.getAppointments().catch(() => [])
      ])
      setAllergies(a); setChronic(c); setMedications(m)
      setVitals(v); setSurgeries(s); setDocuments(d); setMedProfile(mp)
      
      const now = new Date()
      const todayKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`

      const toDayKey = (dateStr: string) => {
        const date = new Date(dateStr)
        if (isNaN(date.getTime())) return 'unknown'
        const y = date.getFullYear()
        const m = `${date.getMonth() + 1}`.padStart(2, '0')
        const day = `${date.getDate()}`.padStart(2, '0')
        return `${y}-${m}-${day}`
      }

      const hasApptToday = appts.some((appt: any) => 
        String(appt.patientId) === pid && toDayKey(appt.scheduledAt) === todayKey
      )
      setHasAccessToday(hasApptToday)
    } finally { setLoading(false) }
  }

  useEffect(() => { fetchAll() }, [pid])

  const runAiAnalysis = async () => {
    setAnalyzing(true)
    try {
      const cutoff = new Date()
      cutoff.setMonth(cutoff.getMonth() - 8)

      const recentVisits = (patientHistory?.lastVisits || [])
        .filter((v) => new Date(v.visitDate) >= cutoff)
        .map((v) => ({
          date: v.visitDate,
          doctor: v.doctorName,
          specialty: v.doctorSpecialty,
          complaint: v.chiefComplaint,
          summary_en: v.summaryEn || v.summary,
          summary_ar: v.summaryAr || v.summary,
        }))

      const payload = {
        vitals: vitals.map(v => ({ type: v.readingType, value: v.value, recordedAt: v.recordedAt })),
        surgeries: surgeries.map(s => s.surgeryName),
        medications: medications.map(m => m.medicationName),
        allergies: allergies.map(a => a.allergenName),
        chronic_diseases: chronic.map(c => ({ diseaseName: c.diseaseName })),
        documents_analysis: documents.map(d => ({
          title: d.title,
          ai_analysis: d.description || '',
        })),
        recent_visits: recentVisits,
      }

      const { data: analysis } = await axiosInstance.post<Record<string, unknown>>('/api/chat/analyze-history', payload)

      const normalized = {
        analysis_en: String(analysis.analysis_en ?? analysis.en ?? ''),
        analysis_ar: String(analysis.analysis_ar ?? analysis.ar ?? ''),
        needsDoctor: analysis.needsDoctor,
      }

      if (!normalized.analysis_ar.trim()) {
        normalized.analysis_ar = 'لم يتم إنشاء التقرير بالعربية. يرجى إعادة التحليل.'
      }
      if (!normalized.analysis_en.trim()) {
        normalized.analysis_en = normalized.analysis_ar
      }

      await patientRecordsApi.updateMedicalProfile(pid, {
        aiDiagnosisSummary: JSON.stringify(normalized)
      })

      toast.success(t('aiAnalysisUpdated'))
      fetchAll()
    } catch (e) {
      toast.error(t('errAiAnalysis'))
      console.error(e)
    } finally {
      setAnalyzing(false)
    }
  }

  const age = patient?.dateOfBirth
    ? Math.floor((Date.now() - new Date(patient.dateOfBirth).getTime()) / (365.25*24*60*60*1000))
    : null

  // --- Form states ---
  const [allergyForm, setAllergyForm] = useState({ allergenName: '', allergyType: 'Drug', severity: 'mild', reactionDescription: '', isActive: true })
  const [chronicForm, setChronicForm] = useState({ diseaseName: '', diseaseType: '', severity: 'moderate', doctorNotes: '', isActive: true })
  const [medForm, setMedForm] = useState({
    medicationName: '',
    genericName: '',
    dosage: '',
    form: 'Pill',
    frequency: 'Daily',
    timesPerDay: 1,
    doseTimes: '09:00',
    daysOfWeek: 'Saturday,Sunday,Monday,Tuesday,Wednesday,Thursday,Friday',
    startDate: new Date().toISOString().split('T')[0],
    endDate: '',
    instructions: '',
    pillsRemaining: 30,
    refillThreshold: 5,
    isChronic: false,
    isActive: true
  })
  const [vitalForm, setVitalForm] = useState({ readingType: 'Blood Pressure', value: '', value2: '', unit: 'mmHg', notes: '' })
  const [surgeryForm, setSurgeryForm] = useState({ surgeryName: '', surgeryDate: '', hospitalName: '', doctorName: '', notes: '' })

  const handleAdd = async () => {
    setSaving(true)
    try {
      if (activeTab === 'allergies') {
        await patientRecordsApi.createAllergy(pid, allergyForm as any)
        setAllergyForm({ allergenName: '', allergyType: 'Drug', severity: 'mild', reactionDescription: '', isActive: true })
      } else if (activeTab === 'chronic') {
        await patientRecordsApi.createChronicDisease(pid, chronicForm)
        setChronicForm({ diseaseName: '', diseaseType: '', severity: 'moderate', doctorNotes: '', isActive: true })
      } else if (activeTab === 'medications') {
        const payload = {
          ...medForm,
          timesPerDay: Number(medForm.timesPerDay),
          pillsRemaining: medForm.pillsRemaining ? Number(medForm.pillsRemaining) : null,
          refillThreshold: Number(medForm.refillThreshold),
          endDate: medForm.endDate || null
        }
        await patientRecordsApi.createMedication(pid, payload as any)
        setMedForm({
          medicationName: '', genericName: '', dosage: '', form: 'Pill', frequency: 'Daily',
          timesPerDay: 1, doseTimes: '09:00', daysOfWeek: 'Saturday,Sunday,Monday,Tuesday,Wednesday,Thursday,Friday',
          startDate: new Date().toISOString().split('T')[0], endDate: '', instructions: '',
          pillsRemaining: 30, refillThreshold: 5, isChronic: false, isActive: true
        })
      } else if (activeTab === 'vitals') {
        await patientRecordsApi.createVital(pid, {
          readingType: vitalForm.readingType, value: Number(vitalForm.value),
          value2: vitalForm.value2 ? Number(vitalForm.value2) : undefined,
          unit: vitalForm.unit, isNormal: true, recordedAt: new Date().toISOString(), notes: vitalForm.notes
        } as any)
        setVitalForm({ readingType: 'Blood Pressure', value: '', value2: '', unit: 'mmHg', notes: '' })
      } else if (activeTab === 'surgeries') {
        await patientRecordsApi.createSurgery(pid, surgeryForm as any)
        setSurgeryForm({ surgeryName: '', surgeryDate: '', hospitalName: '', doctorName: '', notes: '' })
      }
      toast.success(t('recordAdded'))
      setShowForm(false)
      fetchAll()
    } catch { toast.error(t('errAddRecord')) }
    finally { setSaving(false) }
  }

  const handleDelete = async (type: string, id: number) => {
    try {
      if (type === 'allergy') await patientRecordsApi.deleteAllergy(id)
      else if (type === 'chronic') await patientRecordsApi.deleteChronicDisease(id)
      else if (type === 'medication') await patientRecordsApi.deleteMedication(id)
      else if (type === 'vital') await patientRecordsApi.deleteVital(id)
      else if (type === 'surgery') await patientRecordsApi.deleteSurgery(id)
      else if (type === 'document') await patientRecordsApi.deleteDocument(id)
      toast.success(t('deleted')); fetchAll()
    } catch { toast.error(t('errDelete')) }
  }

  const canAdd = !['profile', 'scans', 'visits'].includes(activeTab) || activeTab === 'scans'

  if (loading || hasAccessToday === null) return <PageLoader />

  if (!hasAccessToday) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-950 flex items-center justify-center p-6">
        <div className="max-w-md w-full glass-card bg-white/50 dark:bg-slate-900/50 p-8 rounded-3xl text-center shadow-xl">
          <div className="w-20 h-20 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-500 rounded-full flex items-center justify-center mx-auto mb-6">
            <Lock size={40} />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-3">{t('accessRestricted')}</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-8 leading-relaxed">
            For privacy and security reasons, you can only access and modify a patient's medical profile on the actual day of their scheduled appointment.
          </p>
          <Button onClick={() => navigate(-1)} className="w-full bg-gradient-to-r from-gray-800 to-gray-900 hover:from-gray-700 hover:to-gray-800 text-white py-3 rounded-xl font-bold">
            Return to Patients List
          </Button>
        </div>
      </div>
    )
  }

  const renderEmpty = (msg: string) => (
    <div className="flex flex-col items-center py-16 text-center">
      <div className="w-16 h-16 rounded-2xl bg-gray-100 flex items-center justify-center mb-4">
        <FileText className="w-7 h-7 text-gray-300" />
      </div>
      <p className="text-gray-500 font-medium">{msg}</p>
      <p className="text-sm text-gray-400 mt-1">{t('noRecordsFound')}</p>
    </div>
  )

  const inputCls = "w-full border border-gray-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary-400/40 focus:border-primary-400 transition"
  const labelCls = "block text-xs font-semibold text-gray-500 mb-1.5"

  const renderForm = () => {
    if (activeTab === 'allergies') return (
      <div className="space-y-4">
        <div><label className={labelCls}>{t('allergenName')}</label><input className={inputCls} value={allergyForm.allergenName} onChange={e => setAllergyForm(f => ({...f, allergenName: e.target.value}))} /></div>
        <div className="grid grid-cols-2 gap-3">
          <div><label className={labelCls}>{t('type')}</label><select className={inputCls} value={allergyForm.allergyType} onChange={e => setAllergyForm(f => ({...f, allergyType: e.target.value}))}><option>Drug</option><option>Food</option><option>Environmental</option><option>Other</option></select></div>
          <div><label className={labelCls}>{t('severity')}</label><select className={inputCls} value={allergyForm.severity} onChange={e => setAllergyForm(f => ({...f, severity: e.target.value}))}><option value="mild">Mild</option><option value="moderate">Moderate</option><option value="severe">Severe</option><option value="life_threatening">Life Threatening</option></select></div>
        </div>
        <div><label className={labelCls}>{t('reaction')}</label><textarea className={inputCls} rows={2} value={allergyForm.reactionDescription} onChange={e => setAllergyForm(f => ({...f, reactionDescription: e.target.value}))} /></div>
      </div>
    )
    if (activeTab === 'chronic') return (
      <div className="space-y-4">
        <div><label className={labelCls}>{t('diseaseName')}</label><input className={inputCls} value={chronicForm.diseaseName} onChange={e => setChronicForm(f => ({...f, diseaseName: e.target.value}))} /></div>
        <div className="grid grid-cols-2 gap-3">
          <div><label className={labelCls}>{t('type')}</label><input className={inputCls} value={chronicForm.diseaseType} onChange={e => setChronicForm(f => ({...f, diseaseType: e.target.value}))} /></div>
          <div><label className={labelCls}>{t('severity')}</label><select className={inputCls} value={chronicForm.severity} onChange={e => setChronicForm(f => ({...f, severity: e.target.value}))}><option value="mild">Mild</option><option value="moderate">Moderate</option><option value="severe">Severe</option></select></div>
        </div>
        <div><label className={labelCls}>{t('doctorNotes')}</label><textarea className={inputCls} rows={2} value={chronicForm.doctorNotes} onChange={e => setChronicForm(f => ({...f, doctorNotes: e.target.value}))} /></div>
      </div>
    )
    if (activeTab === 'medications') return (
      <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2 custom-scrollbar">
        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <label className={labelCls}>{t('medicationName')}</label>
            <input className={inputCls} value={medForm.medicationName} onChange={e => setMedForm(f => ({...f, medicationName: e.target.value}))} placeholder="e.g. Panadol" />
          </div>
          <div>
            <label className={labelCls}>{t('genericName')}</label>
            <input className={inputCls} value={medForm.genericName} onChange={e => setMedForm(f => ({...f, genericName: e.target.value}))} placeholder="e.g. Paracetamol" />
          </div>
          <div>
            <label className={labelCls}>{t('form')}</label>
            <select className={inputCls} value={medForm.form} onChange={e => setMedForm(f => ({...f, form: e.target.value}))}>
              {["Pill", "Syrup", "Injection", "Inhaler", "Cream", "Drops", "Patch", "Powder"].map(o => <option key={o}>{o}</option>)}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className={labelCls}>{t('dosage')}</label>
            <input className={inputCls} value={medForm.dosage} onChange={e => setMedForm(f => ({...f, dosage: e.target.value}))} placeholder="e.g. 500mg" />
          </div>
          <div>
            <label className={labelCls}>{t('frequency')}</label>
            <select className={inputCls} value={medForm.frequency} onChange={e => setMedForm(f => ({...f, frequency: e.target.value}))}>
              <option>{t('daily')}</option>
              <option>{t('specificDays')}</option>
              <option>{t('everyOtherDay')}</option>
            </select>
          </div>
          <div>
            <label className={labelCls}>{t('timesPerDay')}</label>
            <input type="number" className={inputCls} value={medForm.timesPerDay} onChange={e => setMedForm(f => ({...f, timesPerDay: Number(e.target.value)}))} />
          </div>
        </div>

        <div>
          <label className={labelCls}>{t('doseTimes')}</label>
          <input className={inputCls} value={medForm.doseTimes} onChange={e => setMedForm(f => ({...f, doseTimes: e.target.value}))} placeholder="09:00, 21:00" />
        </div>

        {medForm.frequency === 'Specific Days' && (
          <div>
            <label className={labelCls}>{t('daysOfWeek')}</label>
            <input className={inputCls} value={medForm.daysOfWeek} onChange={e => setMedForm(f => ({...f, daysOfWeek: e.target.value}))} placeholder="Monday, Wednesday, Friday" />
          </div>
        )}

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={labelCls}>{t('startDate')}</label>
            <input type="date" className={inputCls} value={medForm.startDate} onChange={e => setMedForm(f => ({...f, startDate: e.target.value}))} />
          </div>
          <div>
            <label className={labelCls}>{t('endDateOpt')}</label>
            <input type="date" className={inputCls} value={medForm.endDate} onChange={e => setMedForm(f => ({...f, endDate: e.target.value}))} />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={labelCls}>{t('pillsRemaining')}</label>
            <input type="number" className={inputCls} value={medForm.pillsRemaining} onChange={e => setMedForm(f => ({...f, pillsRemaining: Number(e.target.value)}))} />
          </div>
          <div>
            <label className={labelCls}>{t('refillThreshold')}</label>
            <input type="number" className={inputCls} value={medForm.refillThreshold} onChange={e => setMedForm(f => ({...f, refillThreshold: Number(e.target.value)}))} />
          </div>
        </div>

        <div className="flex items-center gap-4 py-1">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={medForm.isChronic} onChange={e => setMedForm(f => ({...f, isChronic: e.target.checked}))} className="w-4 h-4 rounded text-primary-600 focus:ring-primary-500" />
            <span className="text-sm font-medium text-gray-700">{t('chronicMedication')}</span>
          </label>
        </div>

        <div>
          <label className={labelCls}>{t('instructions')}</label>
          <textarea className={inputCls} rows={2} value={medForm.instructions} onChange={e => setMedForm(f => ({...f, instructions: e.target.value}))} placeholder="Take after meals" />
        </div>
      </div>
    )
    if (activeTab === 'vitals') return (
      <div className="space-y-4">
        <div><label className={labelCls}>Reading {t('type')}</label><select className={inputCls} value={vitalForm.readingType} onChange={e => setVitalForm(f => ({...f, readingType: e.target.value}))}><option>Blood Pressure</option><option>Blood Sugar</option><option>Heart Rate</option><option>Temperature</option><option>SpO2</option><option>Weight</option></select></div>
        <div className="grid grid-cols-3 gap-3">
          <div><label className={labelCls}>{t('valueReq')}</label><input type="number" className={inputCls} value={vitalForm.value} onChange={e => setVitalForm(f => ({...f, value: e.target.value}))} /></div>
          <div><label className={labelCls}>{t('value2')}</label><input type="number" className={inputCls} value={vitalForm.value2} onChange={e => setVitalForm(f => ({...f, value2: e.target.value}))} placeholder="e.g. diastolic" /></div>
          <div><label className={labelCls}>{t('unit')}</label><input className={inputCls} value={vitalForm.unit} onChange={e => setVitalForm(f => ({...f, unit: e.target.value}))} /></div>
        </div>
        <div><label className={labelCls}>{t('notes')}</label><textarea className={inputCls} rows={2} value={vitalForm.notes} onChange={e => setVitalForm(f => ({...f, notes: e.target.value}))} /></div>
      </div>
    )
    if (activeTab === 'surgeries') return (
      <div className="space-y-4">
        <div><label className={labelCls}>{t('surgeryName')}</label><input className={inputCls} value={surgeryForm.surgeryName} onChange={e => setSurgeryForm(f => ({...f, surgeryName: e.target.value}))} /></div>
        <div className="grid grid-cols-2 gap-3">
          <div><label className={labelCls}>{t('date')}</label><input type="date" className={inputCls} value={surgeryForm.surgeryDate} onChange={e => setSurgeryForm(f => ({...f, surgeryDate: e.target.value}))} /></div>
          <div><label className={labelCls}>{t('hospital')}</label><input className={inputCls} value={surgeryForm.hospitalName} onChange={e => setSurgeryForm(f => ({...f, hospitalName: e.target.value}))} /></div>
        </div>
        <div><label className={labelCls}>{t('notes')}</label><textarea className={inputCls} rows={2} value={surgeryForm.notes} onChange={e => setSurgeryForm(f => ({...f, notes: e.target.value}))} /></div>
      </div>
    )
    return null
  }

  const sevColor = (s: string) => {
    if (s === 'severe' || s === 'life_threatening') return 'bg-red-100 text-red-700'
    if (s === 'moderate') return 'bg-amber-100 text-amber-700'
    return 'bg-green-100 text-green-700'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100/50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950 p-6 text-slate-900 dark:text-slate-100">
      {/* Header */}
      <div className="flex items-center gap-5 mb-8">
        <button onClick={() => navigate(-1)} className="p-2.5 hover:bg-white dark:hover:bg-slate-800 rounded-xl transition shadow-sm border border-gray-100 dark:border-slate-800">
          <ChevronLeft className="w-5 h-5 text-gray-500 dark:text-slate-400" />
        </button>
        <div className="flex items-center gap-4 flex-1">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-lg shadow-primary-500/20">
            <span className="text-white font-black text-xl">{(patient?.fullName || 'P').charAt(0).toUpperCase()}</span>
          </div>
          <div>
            <h1 className="text-2xl font-black text-gray-900 dark:text-white">{patient?.fullName || 'Patient'}</h1>
            <div className="flex items-center gap-3 mt-1">
              {age && <span className="text-xs font-semibold bg-gray-100 dark:bg-slate-800 text-gray-500 dark:text-slate-400 px-2.5 py-0.5 rounded-full">{age} years</span>}
              {patient?.gender && <span className="text-xs font-semibold bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 px-2.5 py-0.5 rounded-full">{patient.gender}</span>}
              {medProfile?.bloodType && <span className="text-xs font-semibold bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 px-2.5 py-0.5 rounded-full">🩸 {medProfile.bloodType}</span>}
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1.5 mb-6 overflow-x-auto pb-2 scrollbar-hide bg-white/60 dark:bg-slate-900/60 backdrop-blur-sm rounded-2xl p-1.5 border border-gray-100 dark:border-slate-800/80 shadow-sm">
        {TABS.map(tab => {
          const Icon = tab.icon
          const active = activeTab === tab.id
          return (
            <button key={tab.id} onClick={() => { setActiveTab(tab.id); setShowForm(false) }}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold whitespace-nowrap transition-all ${
                active ? 'bg-primary-600 text-white shadow-lg shadow-primary-500/25' : 'text-gray-500 dark:text-slate-400 hover:bg-gray-100 dark:hover:bg-slate-800 hover:text-gray-700 dark:hover:text-slate-300'
              }`}>
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Content */}
      <Card className="relative min-h-[400px] shadow-xl shadow-black/[0.03] border-gray-100/80 dark:border-slate-800/60">
        <div className="p-6">
          {activeTab === 'profile' && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {[
                  { l: t('fullName'), v: patient?.fullName },
                  { l: t('email'), v: patient?.email },
                  { l: t('phone'), v: patient?.phoneNumber },
                  { l: t('bloodType'), v: medProfile?.bloodType },
                  { l: t('weightL'), v: medProfile?.weightKg ? `${medProfile.weightKg} kg` : null },
                  { l: t('height'), v: medProfile?.heightCm ? `${medProfile.heightCm} cm` : null },
                  { l: t('smoking'), v: medProfile?.isSmoker ? `${t('yes')} (${medProfile.smokingDetails || t('daily')})` : t('no') },
                  { l: t('emergencyContact'), v: medProfile?.emergencyContactName },
                  { l: t('emergencyPhone'), v: medProfile?.emergencyContactPhone },
                ].map((item, i) => (
                  <div key={i} className="p-4 bg-gray-50 dark:bg-slate-900/50 rounded-xl border border-gray-100 dark:border-slate-800/80">
                    <p className="text-xs text-gray-400 dark:text-slate-500 font-semibold mb-1 uppercase tracking-wider">{item.l}</p>
                    <p className="text-sm font-semibold text-gray-800 dark:text-slate-200">{item.v || '—'}</p>
                  </div>
                ))}
              </div>

              {/* AI REPORT SECTION */}
              <div className="mt-8 border-t border-gray-100 dark:border-slate-800/85 pt-8">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-purple-50 dark:bg-purple-950/20 flex items-center justify-center">
                      <Shield className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                    </div>
                    <div>
                      <h3 className="text-lg font-bold text-gray-900 dark:text-white">{t('aiHealthInsights')}</h3>
                      <p className="text-xs text-gray-500 dark:text-slate-400">{t('automatedAnalysis')}</p>
                    </div>
                  </div>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={runAiAnalysis} 
                    disabled={analyzing}
                    className="gap-2 border-purple-200 text-purple-700 hover:bg-purple-50"
                  >
                    {analyzing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                    {analyzing ? 'Analyzing...' : (medProfile?.aiDiagnosisSummary ? t('refreshAnalysis') : t('generateAnalysis'))}
                  </Button>
                </div>
                
                {medProfile?.aiDiagnosisSummary ? (
                  <div className="w-full">
                    {(() => {
                      const report = parseAiDiagnosisSummary(medProfile.aiDiagnosisSummary)
                      if (!report) {
                        return <p className="text-sm text-gray-500 italic">{t('reportProcessing')}</p>
                      }

                      const showArabicFallbackNote = aiLang === 'ar' && !hasDistinctArabicReport(report)

                      return (
                        <div className="p-6 bg-purple-50/50 dark:bg-purple-950/10 rounded-2xl border border-purple-100 dark:border-purple-900/30">
                          <div className="flex items-center justify-between mb-3 gap-3 flex-wrap">
                            <span className="text-[10px] font-bold text-purple-600 dark:text-purple-400 uppercase tracking-widest bg-white dark:bg-slate-900 px-3 py-1 rounded-full border border-purple-100 dark:border-purple-900/30">
                              {aiLang === 'ar' ? 'تقرير التحليل الذكي' : 'AI Analysis Report'}
                            </span>

                            <div className="flex bg-white dark:bg-slate-900 p-1 rounded-xl border border-purple-200 dark:border-purple-800 shadow-sm">
                              <button
                                type="button"
                                onClick={() => setAiLang('en')}
                                className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${aiLang === 'en' ? 'bg-purple-600 text-white shadow-md' : 'text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/30'}`}
                              >
                                English
                              </button>
                              <button
                                type="button"
                                onClick={() => setAiLang('ar')}
                                className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${aiLang === 'ar' ? 'bg-purple-600 text-white shadow-md' : 'text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/30'}`}
                              >
                                العربية
                              </button>
                            </div>
                          </div>

                          {showArabicFallbackNote && (
                            <p className="text-xs text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900/40 rounded-xl px-3 py-2 mb-3 text-right">
                              التقرير العربي غير متوفر في النسخة الحالية. اضغط Refresh Analysis لإعادة توليد التقرير بالعربية.
                            </p>
                          )}

                          <p
                            className={`text-sm text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap ${aiLang === 'ar' ? 'text-right' : 'text-left'}`}
                            dir={aiLang === 'ar' ? 'rtl' : 'ltr'}
                          >
                            {getAiReportText(report, aiLang)}
                          </p>
                        </div>
                      )
                    })()}
                  </div>
                ) : (
                  <div className="p-8 bg-gray-50 rounded-2xl border border-dashed border-gray-200 text-center">
                    <Sparkles className="w-8 h-8 text-purple-300 mx-auto mb-3" />
                    <p className="text-sm text-gray-500">{t('noHealthAnalysis')}</p>
                    <p className="text-xs text-gray-400 mt-1">{t('clickToAnalyze')}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'visits' && ((patientHistory?.lastVisits || []).length === 0 ? renderEmpty(t('noVisitsHistory')) : (
            <div className="space-y-1">
              {/* Timeline */}
              <div className="relative">
                {/* Vertical timeline line */}
                <div className="absolute left-[19px] top-4 bottom-4 w-px bg-gradient-to-b from-primary-300 via-primary-200 to-transparent dark:from-primary-600 dark:via-primary-800" />

                {patientHistory?.lastVisits?.map((v, idx) => (
                  <div key={v.id} className="relative flex gap-4 pb-6 last:pb-0 group">
                    {/* Timeline dot */}
                    <div className="relative z-10 flex-shrink-0 mt-1">
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center shadow-md transition-transform group-hover:scale-110 ${
                        idx === 0 
                          ? 'bg-gradient-to-br from-primary-500 to-primary-700 text-white ring-4 ring-primary-100 dark:ring-primary-900/40' 
                          : 'bg-white dark:bg-slate-800 text-primary-500 border-2 border-primary-200 dark:border-primary-700'
                      }`}>
                        <Clock className="w-4 h-4" />
                      </div>
                    </div>

                    {/* Visit card */}
                    <div className={`flex-1 rounded-2xl border transition-all group-hover:shadow-lg ${
                      idx === 0 
                        ? 'bg-gradient-to-br from-white to-primary-50/30 dark:from-slate-800 dark:to-primary-950/20 border-primary-100 dark:border-primary-800/50 shadow-md' 
                        : 'bg-white dark:bg-slate-800/60 border-gray-100 dark:border-slate-700/50 shadow-sm'
                    }`}>
                      {/* Card header */}
                      <div className="flex items-center justify-between px-5 py-3.5 border-b border-gray-50 dark:border-slate-700/30">
                        <div className="flex items-center gap-3">
                          <div className={`w-9 h-9 rounded-xl flex items-center justify-center text-sm font-black ${
                            idx === 0 
                              ? 'bg-primary-100 dark:bg-primary-900/40 text-primary-700 dark:text-primary-300' 
                              : 'bg-gray-100 dark:bg-slate-700 text-gray-500 dark:text-slate-400'
                          }`}>
                            {v.doctorName ? v.doctorName.charAt(0).toUpperCase() : '?'}
                          </div>
                          <div>
                            <p className="font-bold text-sm text-gray-900 dark:text-white">
                              {v.doctorName ? `Dr. ${v.doctorName}` : t('unknownDoctor')}
                            </p>
                            <p className="text-[11px] text-gray-400 dark:text-slate-500">
                              {v.doctorSpecialty || t('specialist')}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-bold ${
                            idx === 0 
                              ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300' 
                              : 'bg-gray-100 dark:bg-slate-700 text-gray-500 dark:text-slate-400'
                          }`}>
                            <Clock className="w-3 h-3" />
                            {v.visitDate}
                          </span>
                        </div>
                      </div>

                      {/* Card body */}
                      <div className="px-5 py-4 space-y-3">
                        {/* Chief complaint */}
                        <div className="flex items-start gap-2">
                          <span className="text-[10px] font-bold uppercase tracking-widest text-gray-400 dark:text-slate-500 mt-0.5 shrink-0 w-20">{t('complaintLabel')}</span>
                          <p className="text-sm text-gray-700 dark:text-slate-300 font-medium leading-relaxed">
                            {v.chiefComplaint || t('noneRecorded')}
                          </p>
                        </div>

                        {/* Summary */}
                        {v.summary && (
                          <div className="mt-2 bg-gradient-to-br from-slate-50 to-gray-50 dark:from-slate-800/80 dark:to-slate-900/50 rounded-xl border border-gray-100/80 dark:border-slate-700/40 overflow-hidden transition-all">
                            <div className="p-4">
                              <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                  <Sparkles className="w-3.5 h-3.5 text-purple-500" />
                                  <span className="text-[10px] font-bold text-purple-600 dark:text-purple-400 uppercase tracking-widest">{t('aiSummary')}</span>
                                </div>
                                <button
                                  onClick={() => setExpandedVisits(prev => ({ ...prev, [v.id]: !prev[v.id] }))}
                                  className="text-[10px] font-bold text-purple-600 hover:text-purple-700 bg-purple-50 hover:bg-purple-100 dark:bg-purple-900/30 dark:hover:bg-purple-900/50 dark:text-purple-300 px-2.5 py-1 rounded-md transition"
                                >
                                  {expandedVisits[v.id] ? t('readLess') : t('readMore')}
                                </button>
                              </div>
                              {expandedVisits[v.id] && (
                                <p className="text-sm text-gray-600 dark:text-slate-400 leading-relaxed whitespace-pre-wrap mt-3 border-t border-purple-100/50 dark:border-purple-900/20 pt-3">{v.summary}</p>
                              )}
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Quick view link */}
                      <div className="px-5 py-2.5 border-t border-gray-50 dark:border-slate-700/30 flex justify-end">
                        <button
                          onClick={() => window.open(`/doctor/visits/${v.id}/summary`, '_blank')}
                          className="text-[11px] font-bold text-primary-600 dark:text-primary-400 hover:text-primary-700 transition flex items-center gap-1"
                        >
                          {t('viewFullDetails')}
                          <ChevronLeft className="w-3 h-3 rotate-180" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}

          {activeTab === 'allergies' && (allergies.length === 0 ? renderEmpty(t('noAllergies')) : (
            <div className="space-y-3">
              {allergies.map(a => (
                <div key={a.id} className="flex items-center justify-between p-4 bg-white dark:bg-slate-800/60 border border-gray-100 dark:border-slate-700/50 rounded-xl hover:shadow-md transition">
                  <div>
                    <div className="flex items-center gap-2">
                      <p className="font-semibold text-gray-800 dark:text-white">{a.allergenName}</p>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${sevColor(a.severity)}`}>{a.severity}</span>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">{a.allergyType}{a.reactionDescription ? ` · ${a.reactionDescription}` : ''}</p>
                  </div>
                  <button onClick={() => handleDelete('allergy', a.id)} className="p-2 text-gray-400 dark:text-slate-500 hover:text-red-500 dark:hover:text-red-400 transition"><Trash2 className="w-4 h-4" /></button>
                </div>
              ))}
            </div>
          ))}

          {activeTab === 'chronic' && (chronic.length === 0 ? renderEmpty(t('noChronic')) : (
            <div className="space-y-3">
              {chronic.map(d => (
                <div key={d.id} className="flex items-center justify-between p-4 bg-white dark:bg-slate-800/60 border border-gray-100 dark:border-slate-700/50 rounded-xl hover:shadow-md transition">
                  <div>
                    <div className="flex items-center gap-2">
                      <p className="font-semibold text-gray-800 dark:text-white">{d.diseaseName}</p>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${sevColor(d.severity)}`}>{d.severity}</span>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">{d.diseaseType}{d.doctorNotes ? ` · ${d.doctorNotes}` : ''}</p>
                  </div>
                  <button onClick={() => handleDelete('chronic', d.id)} className="p-2 text-gray-400 dark:text-slate-500 hover:text-red-500 dark:hover:text-red-400 transition"><Trash2 className="w-4 h-4" /></button>
                </div>
              ))}
            </div>
          ))}

          {activeTab === 'medications' && (medications.length === 0 ? renderEmpty(t('noMedications')) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {medications.map(m => (
                <div key={m.id} className="p-4 bg-white dark:bg-slate-800/60 border border-gray-100 dark:border-slate-700/50 rounded-xl hover:shadow-md transition">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <p className="font-bold text-gray-800 dark:text-white text-lg">{m.medicationName}</p>
                      {m.genericName && <p className="text-xs text-gray-400 dark:text-slate-500 font-medium">{m.genericName}</p>}
                    </div>
                    {m.isChronic && <span className="bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider border border-amber-200 dark:border-amber-800/50">{t('chronic')}</span>}
                  </div>
                  <div className="space-y-2 mt-4">
                    <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-slate-300">
                      <div className="w-6 h-6 rounded-md bg-primary-50 dark:bg-primary-900/30 flex items-center justify-center text-primary-600 dark:text-primary-400"><Heart className="w-3.5 h-3.5" /></div>
                      <span className="font-medium">{m.dosage} <span className="text-gray-400 dark:text-slate-500 mx-1">·</span> {m.form}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-slate-300">
                      <div className="w-6 h-6 rounded-md bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center text-blue-600 dark:text-blue-400"><Activity className="w-3.5 h-3.5" /></div>
                      <span className="font-medium">{m.frequency} <span className="text-gray-400 dark:text-slate-500 mx-1">·</span> {m.timesPerDay}x/day <span className="text-gray-400 dark:text-slate-500 text-xs">({m.doseTimes})</span></span>
                    </div>
                    {m.instructions && (
                      <div className="mt-3 bg-gray-50 dark:bg-slate-900/50 p-3 rounded-lg border border-gray-100 dark:border-slate-700/50">
                        <p className="text-xs text-gray-600 dark:text-slate-400 italic">"{m.instructions}"</p>
                      </div>
                    )}
                  </div>
                  <div className="mt-4 pt-4 border-t border-gray-50 dark:border-slate-700/30 flex justify-between items-center">
                    <span className="text-[10px] font-medium text-gray-400 dark:text-slate-500">{t('added')}: {new Date(m.createdAt).toLocaleDateString()}</span>
                    <button onClick={() => handleDelete('medication', m.id)} className="p-1.5 text-gray-400 dark:text-slate-500 hover:bg-red-50 dark:hover:bg-red-900/20 hover:text-red-500 dark:hover:text-red-400 rounded-lg transition"><Trash2 className="w-4 h-4" /></button>
                  </div>
                </div>
              ))}
            </div>
          ))}

          {activeTab === 'vitals' && (vitals.length === 0 ? renderEmpty(t('noVitals')) : (
            <div className="space-y-3">
              {vitals.map(v => (
                <div key={v.id} className="flex items-center justify-between p-4 bg-white border border-gray-100 rounded-xl hover:shadow-md transition">
                  <div>
                    <p className="font-semibold text-gray-800">{v.readingType}: {v.value}{v.value2 != null ? `/${v.value2}` : ''} {v.unit}</p>
                    <p className="text-xs text-gray-500 mt-1">{new Date(v.recordedAt).toLocaleDateString()} {v.isNormal ? '✅ Normal' : '⚠️ Abnormal'}</p>
                  </div>
                  <button onClick={() => handleDelete('vital', v.id)} className="p-2 text-gray-400 hover:text-red-500 transition"><Trash2 className="w-4 h-4" /></button>
                </div>
              ))}
            </div>
          ))}

          {activeTab === 'surgeries' && (surgeries.length === 0 ? renderEmpty('No surgeries') : (
            <div className="space-y-3">
              {surgeries.map(s => (
                <div key={s.id} className="flex items-center justify-between p-4 bg-white border border-gray-100 rounded-xl hover:shadow-md transition">
                  <div>
                    <p className="font-semibold text-gray-800">{s.surgeryName}</p>
                    <p className="text-xs text-gray-500 mt-1">{s.surgeryDate ? new Date(s.surgeryDate).toLocaleDateString() : ''}{s.hospitalName ? ` · ${s.hospitalName}` : ''}</p>
                  </div>
                  <button onClick={() => handleDelete('surgery', s.id)} className="p-2 text-gray-400 hover:text-red-500 transition"><Trash2 className="w-4 h-4" /></button>
                </div>
              ))}
            </div>
          ))}

          {activeTab === 'scans' && (documents.length === 0 ? renderEmpty('No scans or labs') : (
            <div className="space-y-3">
              {documents.map(d => (
                <div key={d.id} className="flex items-center justify-between p-4 bg-white border border-gray-100 rounded-xl hover:shadow-md transition">
                  <div>
                    <p className="font-semibold text-gray-800">{d.title}</p>
                    <p className="text-xs text-gray-500 mt-1">{d.documentType} · {new Date(d.uploadedAt).toLocaleDateString()}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {d.fileUrl && <a href={d.fileUrl} target="_blank" rel="noreferrer" className="text-xs text-primary-600 font-medium hover:underline">{t('view')}</a>}
                    <button onClick={() => handleDelete('document', d.id)} className="p-2 text-gray-400 hover:text-red-500 transition"><Trash2 className="w-4 h-4" /></button>
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </Card>

      {/* FAB */}
      {activeTab !== 'profile' && (
        <motion.button
          whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}
          onClick={() => setShowForm(true)}
          className="fixed bottom-8 right-8 w-14 h-14 bg-primary-600 hover:bg-primary-700 text-white rounded-2xl shadow-xl shadow-primary-500/30 flex items-center justify-center z-40 transition"
        >
          <Plus className="w-6 h-6" />
        </motion.button>
      )}

      {/* Add Record Modal */}
      <Modal open={showForm} onClose={() => setShowForm(false)} title={`Add ${TABS.find(t => t.id === activeTab)?.label || 'Record'}`} size="md"
        footer={
          <>
            <Button variant="outline" onClick={() => setShowForm(false)}>{t('cancel')}</Button>
            <Button onClick={handleAdd} disabled={saving} className="bg-primary-600 hover:bg-primary-700 text-white">
              {saving ? <><Loader2 className="w-4 h-4 animate-spin mr-2" />{t('saving')}</> : 'Save'}
            </Button>
          </>
        }
      >
        {renderForm()}
      </Modal>
    </div>
  )
}
