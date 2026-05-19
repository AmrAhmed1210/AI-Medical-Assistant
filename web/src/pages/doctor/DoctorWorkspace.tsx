import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
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
  followUpAfterDays: string
  followUpNotes: string
}

export default function DoctorWorkspace() {
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
    followUpAfterDays: '',
    followUpNotes: '',
  })

  const { visit, isLoading } = useVisit(id)
  const { history: patientHistory } = usePatientHistory(visit?.patientId ?? 0)

  const [isSaving, setIsSaving] = useState(false)
  const [aiLang, setAiLang] = useState<'ar' | 'en'>('ar')
  const [isAssisting, setIsAssisting] = useState(false)

  const handleAiAssist = async () => {
    if (!form.chiefComplaint) {
      toast.error('Please enter the chief complaint first')
      return
    }

    setIsAssisting(true)
    try {
      const aiServerUrl = import.meta.env.VITE_AI_SERVER_URL || 'http://localhost:8000'
      const response = await fetch(`${aiServerUrl}/doctor-ai-assist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
      toast.success('AI suggestions generated successfully')
    } catch (err) {
      console.error(err)
      toast.error('Failed to get AI assistance')
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
        followUpAfterDays: form.followUpAfterDays ? Number(form.followUpAfterDays) : undefined,
        followUpNotes: form.followUpNotes,
      })
      toast.success('Draft saved successfully')
    } catch {
      toast.error('Failed to save draft')
    } finally {
      setIsSaving(false)
    }
  }

  const handleCloseVisit = async () => {
    setIsClosing(true)
    try {
      await visitApi.closeVisit(id)
      toast.success('Visit closed successfully')
      navigate(`/doctor/visits/${id}/summary`)
    } catch {
      toast.error('Failed to close visit')
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
    <div className="h-[calc(100vh-4rem)] flex flex-col">
      {/* Top bar */}
      <div className="bg-white border-b px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" onClick={() => navigate('/doctor/today')}>
            <ChevronLeft className="w-4 h-4" />
            Back
          </Button>
          <div>
            <h1 className="font-bold text-gray-900 flex items-center gap-2">
              <Stethoscope className="w-5 h-5 text-primary-600" />
              Visit: {visit?.patientName}
            </h1>
            <p className="text-sm text-gray-500">
              {visit?.status === 'open' ? 'In Progress — Editable' : 'Closed'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" onClick={handleSaveDraft} disabled={isSaving}>
            <Save className="w-4 h-4 ml-2" />
            Save Draft
          </Button>
          <Button
            onClick={() => setShowConfirmClose(true)}
            className="bg-primary-600 hover:bg-primary-700"
            disabled={!form.chiefComplaint}
          >
            <Lock className="w-4 h-4 ml-2" />
            Finish & Close
          </Button>
        </div>
      </div>

      {/* Critical Alert Banner */}
      <AnimatePresence>
        {criticalAlert && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="bg-red-600 text-white px-6 py-3 text-center font-bold text-sm"
          >
            {criticalAlert}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Split View */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel — Patient History (40%) */}
        <div className="w-2/5 border-l overflow-y-auto bg-gray-50 p-4 space-y-4">
          {/* AI Health Report */}
          {patientHistory?.aiDiagnosisSummary && (
            <Card className="border-purple-200 bg-purple-50/50">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-bold text-purple-900 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-purple-600" />
                  AI Health Report
                </h3>
                <div className="flex bg-white rounded-lg p-1 border border-purple-100">
                  <button 
                    onClick={() => setAiLang('ar')}
                    className={`px-2 py-0.5 text-[10px] rounded-md transition-all ${aiLang === 'ar' ? 'bg-purple-600 text-white shadow-sm' : 'text-purple-600 hover:bg-purple-50'}`}
                  >AR</button>
                  <button 
                    onClick={() => setAiLang('en')}
                    className={`px-2 py-0.5 text-[10px] rounded-md transition-all ${aiLang === 'en' ? 'bg-purple-600 text-white shadow-sm' : 'text-purple-600 hover:bg-purple-50'}`}
                  >EN</button>
                </div>
              </div>
              <div className="text-sm text-purple-900 leading-relaxed max-h-48 overflow-y-auto pr-2 custom-scrollbar">
                {(() => {
                  try {
                    const parsed = JSON.parse(patientHistory.aiDiagnosisSummary);
                    return aiLang === 'ar' ? parsed.analysis_ar : parsed.analysis_en;
                  } catch {
                    return patientHistory.aiDiagnosisSummary;
                  }
                })()}
              </div>
            </Card>
          )}

          {/* SOS Bar */}
          <Card className="bg-red-50 border-red-200">
            <div className="flex items-center gap-2 text-red-700 font-bold mb-2">
              <Heart className="w-4 h-4 fill-red-600" />
              Emergency Info
            </div>
            <div className="space-y-1 text-sm">
              <p><span className="text-gray-500">Blood Type:</span> {patientHistory?.bloodType || '—'}</p>
              {(patientHistory?.allergies?.filter((a: Record<string, string>) => a.severity === 'life_threatening').length ?? 0) > 0 && (
                <div className="flex items-start gap-1 text-red-600">
                  <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                  <span>
                    Life-threatening allergy:{' '}
                    {patientHistory?.allergies
                      ?.filter((a: Record<string, string>) => a.severity === 'life_threatening')
                      .map((a: Record<string, string>) => `${a.allergenName} (${a.reaction})`)
                      .join(', ')}
                  </span>
                </div>
              )}
            </div>
          </Card>

          {/* Chronic Diseases */}
          {(patientHistory?.chronicDiseases?.length ?? 0) > 0 && (
            <Card>
              <h3 className="font-bold text-gray-900 mb-3 flex items-center gap-2">
                <Activity className="w-4 h-4 text-primary-600" />
                Chronic Diseases
              </h3>
              <div className="space-y-2">
                {patientHistory?.chronicDiseases?.map((d: Record<string, string>) => (
                  <div key={d.id} className="p-2 bg-gray-50 rounded-lg text-sm">
                    <p className="font-medium">{d.diseaseName}</p>
                    <p className="text-gray-500">Target: {d.targetValues}</p>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Current Medications */}
          {(patientHistory?.medications?.length ?? 0) > 0 && (
            <Card>
              <h3 className="font-bold text-gray-900 mb-3 flex items-center gap-2">
                <Pill className="w-4 h-4 text-primary-600" />
                Current Medications
              </h3>
              <div className="space-y-2">
                {patientHistory?.medications?.map((m: Record<string, string>) => (
                  <div key={m.id} className="p-2 bg-gray-50 rounded-lg text-sm">
                    <p className="font-medium">{m.medicationName}</p>
                    <p className="text-gray-500">{m.dosage} — {m.frequency}</p>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Latest Vitals */}
          {patientHistory?.latestVitals && (
            <Card>
              <h3 className="font-bold text-gray-900 mb-3">Latest Readings</h3>
              <div className="grid grid-cols-2 gap-2 text-sm">
                {Object.entries(patientHistory.latestVitals).map(([key, val]: [string, unknown]) => (
                  <div key={key} className="p-2 bg-gray-50 rounded">
                    <p className="text-gray-500">{key}</p>
                    <p className="font-medium">{val as string}</p>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Last Visits */}
          {(patientHistory?.lastVisits?.length ?? 0) > 0 && (
            <Card>
              <h3 className="font-bold text-gray-900 mb-3">Recent Visits</h3>
              <div className="space-y-2">
                {patientHistory?.lastVisits?.slice(0, 3).map((v: Record<string, string>) => (
                  <div key={v.id} className="p-2 bg-gray-50 rounded-lg text-sm">
                    <p className="font-medium">{v.visitDate}</p>
                    <p className="text-gray-500 line-clamp-2">{v.chiefComplaint}</p>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>

        {/* Right Panel — Current Visit (60%) */}
        <div className="w-3/5 overflow-y-auto p-6 space-y-6">
          {/* Chief Complaint */}
          <Card>
            <label className="block font-bold text-gray-900 mb-2">Chief Complaint *</label>
            <textarea
              className="w-full border rounded-lg p-3 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none"
              rows={2}
              value={form.chiefComplaint}
              onChange={(e) => setForm((f) => ({ ...f, chiefComplaint: e.target.value }))}
              placeholder="e.g. Chest pain, difficulty breathing"
            />
          </Card>

          {/* History of Present Illness */}
          <Card>
            <label className="block font-bold text-gray-900 mb-2">History of Present Illness</label>
            <textarea
              className="w-full border rounded-lg p-3 text-sm focus:ring-2 focus:ring-primary-500 outline-none"
              rows={3}
              value={form.historyOfIllness}
              onChange={(e) => setForm((f) => ({ ...f, historyOfIllness: e.target.value }))}
              placeholder="Onset, progression, aggravating/relieving factors..."
            />
          </Card>

          {/* Examination Findings */}
          <Card>
            <label className="block font-bold text-gray-900 mb-2">Examination Findings</label>
            <textarea
              className="w-full border rounded-lg p-3 text-sm focus:ring-2 focus:ring-primary-500 outline-none"
              rows={3}
              value={form.examinationFindings}
              onChange={(e) => setForm((f) => ({ ...f, examinationFindings: e.target.value }))}
              placeholder="Physical exam, vital signs, findings..."
            />
          </Card>

          {/* Vital Signs */}
          <Card>
            <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary-600" />
              Vital Signs
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {[
                { key: 'bpSystolic' as const, label: 'Systolic BP', icon: Gauge, type: 'bp_systolic', placeholder: '120' },
                { key: 'bpDiastolic' as const, label: 'Diastolic BP', icon: Gauge, type: 'bp_diastolic', placeholder: '80' },
                { key: 'heartRate' as const, label: 'Heart Rate', icon: Heart, type: 'heart_rate', placeholder: '72' },
                { key: 'temperature' as const, label: 'Temperature', icon: Thermometer, type: 'temperature', placeholder: '37.0' },
                { key: 'bloodSugar' as const, label: 'Blood Sugar', icon: Droplets, type: 'blood_sugar', placeholder: '90' },
                { key: 'weight' as const, label: 'Weight (kg)', icon: Weight, type: '', placeholder: '70' },
                { key: 'spo2' as const, label: 'SpO2', icon: Wind, type: 'spo2', placeholder: '98' },
              ].map((field) => {
                const status = field.type ? getVitalStatus(field.type, form[field.key]) : 'neutral'
                const Icon = field.icon
                return (
                  <div key={field.key} className="relative">
                    <label className="text-xs text-gray-500 mb-1 flex items-center gap-1">
                      <Icon className="w-3 h-3" />
                      {field.label}
                    </label>
                    <div className="relative">
                      <input
                        type="text"
                        inputMode="decimal"
                        className={`w-full border rounded-lg p-2 text-sm outline-none transition-colors ${
                          status === 'abnormal'
                            ? 'border-red-400 bg-red-50 focus:border-red-500 focus:ring-1 focus:ring-red-500'
                            : status === 'normal'
                            ? 'border-green-400 bg-green-50'
                            : 'focus:border-primary-500 focus:ring-1 focus:ring-primary-500'
                        }`}
                        placeholder={field.placeholder}
                        value={form[field.key]}
                        onChange={(e) => setForm((f) => ({ ...f, [field.key]: e.target.value }))}
                      />
                      {status === 'normal' && (
                        <CheckCircle2 className="w-4 h-4 text-green-500 absolute left-2 top-2.5" />
                      )}
                      {status === 'abnormal' && (
                        <AlertCircle className="w-4 h-4 text-red-500 absolute left-2 top-2.5" />
                      )}
                    </div>
                    {status === 'abnormal' && field.type && (
                      <p className="text-xs text-red-500 mt-1">
                        Normal range: {normalRanges[field.type].min}–{normalRanges[field.type].max} {normalRanges[field.type].unit}
                      </p>
                    )}
                  </div>
                )
              })}
            </div>
          </Card>

          {/* Symptoms */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-gray-900">Symptoms</h3>
              <Button size="sm" variant="outline" onClick={handleAddSymptom}>
                <Plus className="w-4 h-4 ml-1" />
                Add Symptom
              </Button>
            </div>
            <div className="space-y-3">
              {form.symptoms.map((sym, idx) => (
                <div key={idx} className="p-3 bg-gray-50 rounded-lg grid grid-cols-2 md:grid-cols-4 gap-3">
                  <input
                    className="border rounded p-2 text-sm"
                    placeholder="Symptom name"
                    value={sym.name}
                    onChange={(e) => {
                      const s = [...form.symptoms]
                      s[idx].name = e.target.value
                      setForm((f) => ({ ...f, symptoms: s }))
                    }}
                  />
                  <select
                    className="border rounded p-2 text-sm"
                    value={sym.severity}
                    onChange={(e) => {
                      const s = [...form.symptoms]
                      s[idx].severity = e.target.value
                      setForm((f) => ({ ...f, symptoms: s }))
                    }}
                  >
                    <option value="mild">Mild</option>
                    <option value="moderate">Moderate</option>
                    <option value="severe">Severe</option>
                  </select>
                  <input
                    className="border rounded p-2 text-sm"
                    placeholder="Location"
                    value={sym.location}
                    onChange={(e) => {
                      const s = [...form.symptoms]
                      s[idx].location = e.target.value
                      setForm((f) => ({ ...f, symptoms: s }))
                    }}
                  />
                  <button
                    className="text-red-400 hover:text-red-600 text-sm"
                    onClick={() => {
                      const s = form.symptoms.filter((_, i) => i !== idx)
                      setForm((f) => ({ ...f, symptoms: s }))
                    }}
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
              {form.symptoms.length === 0 && (
                <p className="text-sm text-gray-400 text-center py-4">No symptoms recorded</p>
              )}
            </div>
          </Card>

          {/* Assessment */}
          <Card>
            <div className="flex items-center justify-between mb-2">
              <label className="block font-bold text-gray-900">Assessment / Diagnosis</label>
              <Button 
                size="sm" 
                variant="outline" 
                className="text-purple-600 border-purple-200 bg-purple-50 hover:bg-purple-100"
                onClick={handleAiAssist}
                disabled={isAssisting}
              >
                {isAssisting ? <Loader2 className="w-3 h-3 animate-spin ml-1" /> : <Wand2 className="w-3 h-3 ml-1" />}
                AI Assist
              </Button>
            </div>
            <textarea
              className="w-full border rounded-lg p-3 text-sm focus:ring-2 focus:ring-primary-500 outline-none"
              rows={2}
              value={form.assessment}
              onChange={(e) => setForm((f) => ({ ...f, assessment: e.target.value }))}
              placeholder="Differential diagnosis..."
            />
          </Card>

          {/* Plan */}
          <Card>
            <label className="block font-bold text-gray-900 mb-2">Treatment Plan</label>
            <textarea
              className="w-full border rounded-lg p-3 text-sm focus:ring-2 focus:ring-primary-500 outline-none"
              rows={2}
              value={form.plan}
              onChange={(e) => setForm((f) => ({ ...f, plan: e.target.value }))}
              placeholder="Treatment steps, recommendations..."
            />
          </Card>

          {/* Prescriptions */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-gray-900">Prescription</h3>
              <Button size="sm" variant="outline" onClick={handleAddPrescription}>
                <Plus className="w-4 h-4 ml-1" />
                Add Medication
              </Button>
            </div>
            <div className="space-y-3">
              {form.prescriptions.map((pres, idx) => (
                <div key={idx} className="p-3 bg-gray-50 rounded-lg space-y-2">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    <input
                      className="border rounded p-2 text-sm"
                      placeholder="Medication name"
                      value={pres.medicationName}
                      onChange={(e) => {
                        const p = [...form.prescriptions]
                        p[idx].medicationName = e.target.value
                        setForm((f) => ({ ...f, prescriptions: p }))
                      }}
                    />
                    <input
                      className="border rounded p-2 text-sm"
                      placeholder="Dosage"
                      value={pres.dosage}
                      onChange={(e) => {
                        const p = [...form.prescriptions]
                        p[idx].dosage = e.target.value
                        setForm((f) => ({ ...f, prescriptions: p }))
                      }}
                    />
                    <input
                      className="border rounded p-2 text-sm"
                      placeholder="Frequency"
                      value={pres.frequency}
                      onChange={(e) => {
                        const p = [...form.prescriptions]
                        p[idx].frequency = e.target.value
                        setForm((f) => ({ ...f, prescriptions: p }))
                      }}
                    />
                    <input
                      className="border rounded p-2 text-sm"
                      placeholder="Duration"
                      value={pres.duration}
                      onChange={(e) => {
                        const p = [...form.prescriptions]
                        p[idx].duration = e.target.value
                        setForm((f) => ({ ...f, prescriptions: p }))
                      }}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={pres.isChronic}
                        onChange={(e) => {
                          const p = [...form.prescriptions]
                          p[idx].isChronic = e.target.checked
                          setForm((f) => ({ ...f, prescriptions: p }))
                        }}
                      />
                      <span>Chronic medication (tracked)</span>
                    </label>
                    <button
                      className="text-red-400 hover:text-red-600 text-sm"
                      onClick={() => {
                        const p = form.prescriptions.filter((_, i) => i !== idx)
                        setForm((f) => ({ ...f, prescriptions: p }))
                      }}
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
              {form.prescriptions.length === 0 && (
                <p className="text-sm text-gray-400 text-center py-4">No medications added</p>
              )}
            </div>
          </Card>

          {/* Follow-up */}
          <Card>
            <h3 className="font-bold text-gray-900 mb-4">Follow-up</h3>
            <label className="flex items-center gap-2 mb-3">
              <input
                type="checkbox"
                checked={form.followUpRequired}
                onChange={(e) => setForm((f) => ({ ...f, followUpRequired: e.target.checked }))}
              />
              <span>Follow-up required</span>
            </label>
            {form.followUpRequired && (
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-gray-500">After (days)</label>
                  <input
                    type="number"
                    className="w-full border rounded-lg p-2 text-sm mt-1"
                    value={form.followUpAfterDays}
                    onChange={(e) => setForm((f) => ({ ...f, followUpAfterDays: e.target.value }))}
                  />
                </div>
                <div>
                  <label className="text-sm text-gray-500">Notes for patient</label>
                  <textarea
                    className="w-full border rounded-lg p-2 text-sm mt-1"
                    rows={2}
                    value={form.followUpNotes}
                    onChange={(e) => setForm((f) => ({ ...f, followUpNotes: e.target.value }))}
                  />
                </div>
              </div>
            )}
          </Card>

          {/* Notes */}
          <Card>
            <label className="block font-bold text-gray-900 mb-2">Additional Notes</label>
            <textarea
              className="w-full border rounded-lg p-3 text-sm focus:ring-2 focus:ring-primary-500 outline-none"
              rows={2}
              value={form.notes}
              onChange={(e) => setForm((f) => ({ ...f, notes: e.target.value }))}
              placeholder="Any additional notes..."
            />
          </Card>

          {/* Bottom actions */}
          <div className="flex gap-3 pb-6">
            <Button variant="outline" className="flex-1" onClick={handleSaveDraft} disabled={isSaving}>
              <Save className="w-4 h-4 ml-2" />
              Save Draft
            </Button>
            <Button
              className="flex-1 bg-primary-600 hover:bg-primary-700"
              onClick={() => setShowConfirmClose(true)}
              disabled={!form.chiefComplaint}
            >
              <Lock className="w-4 h-4 ml-2" />
              Finish & Close Visit
            </Button>
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
            className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-xl p-6 max-w-md w-full"
            >
              <h3 className="font-bold text-lg text-gray-900 mb-2">Confirm Close Visit</h3>
              <p className="text-gray-500 text-sm mb-6">
                Once closed, the visit cannot be edited. Are you sure?
              </p>
              <div className="flex gap-3">
                <Button variant="outline" className="flex-1" onClick={() => setShowConfirmClose(false)}>
                  Cancel
                </Button>
                <Button
                  className="flex-1 bg-primary-600 hover:bg-primary-700"
                  onClick={() => {
                    setShowConfirmClose(false)
                    handleCloseVisit()
                  }}
                  disabled={isClosing}
                >
                  {isClosing ? 'Closing...' : 'Confirm Close'}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
