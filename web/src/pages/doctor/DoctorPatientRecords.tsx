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
import { useDoctorPatients } from '@/hooks/useDoctor'

const TABS = [
  { id: 'profile', label: 'Profile', icon: User },
  { id: 'allergies', label: 'Allergies', icon: Shield },
  { id: 'chronic', label: 'Chronic', icon: Activity },
  { id: 'medications', label: 'Medications', icon: Pill },
  { id: 'vitals', label: 'Vitals', icon: Heart },
  { id: 'surgeries', label: 'Surgeries', icon: Scissors },
  { id: 'scans', label: 'Scans / Labs', icon: FileText },
] as const

type TabId = typeof TABS[number]['id']

export default function DoctorPatientRecords() {
  const { patientId } = useParams<{ patientId: string }>()
  const navigate = useNavigate()
  const pid = patientId ?? '0'
  const { patients = [] } = useDoctorPatients('')
  const patient = patients.find(p => String(p.id) === pid)

  const [activeTab, setActiveTab] = useState<TabId>('profile')
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [saving, setSaving] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)

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
      const [a, c, m, v, s, d, mp] = await Promise.all([
        patientRecordsApi.getAllergies(pid).catch(() => []),
        patientRecordsApi.getChronicDiseases(pid).catch(() => []),
        patientRecordsApi.getMedications(pid).catch(() => []),
        patientRecordsApi.getVitals(pid).catch(() => []),
        patientRecordsApi.getSurgeries(pid).catch(() => []),
        patientRecordsApi.getDocuments(pid).catch(() => []),
        patientRecordsApi.getMedicalProfile(pid).catch(() => null),
      ])
      setAllergies(a); setChronic(c); setMedications(m)
      setVitals(v); setSurgeries(s); setDocuments(d); setMedProfile(mp)
    } finally { setLoading(false) }
  }

  useEffect(() => { fetchAll() }, [pid])

  const runAiAnalysis = async () => {
    setAnalyzing(true)
    try {
      const payload = {
        vitals: vitals.map(v => ({ type: v.readingType, value: v.value, recordedAt: v.recordedAt })),
        surgeries: surgeries.map(s => s.surgeryName),
        medications: medications.map(m => m.medicationName),
        allergies: allergies.map(a => a.allergenName),
        chronic_diseases: chronic.map(c => ({ diseaseName: c.diseaseName })),
      }

      const aiServerUrl = import.meta.env.VITE_AI_SERVER_URL || 'http://localhost:8000'
      const response = await fetch(`${aiServerUrl}/analyze-history`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) throw new Error('AI Analysis failed')
      const analysis = await response.json()

      await patientRecordsApi.updateMedicalProfile(pid, {
        aiDiagnosisSummary: JSON.stringify(analysis)
      })

      toast.success('AI Health Analysis Updated')
      fetchAll()
    } catch (e) {
      toast.error('AI Analysis failed. Check if AI server is running.')
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
      toast.success('Record added successfully')
      setShowForm(false)
      fetchAll()
    } catch { toast.error('Failed to add record') }
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
      toast.success('Deleted'); fetchAll()
    } catch { toast.error('Failed to delete') }
  }

  const canAdd = !['profile', 'scans'].includes(activeTab) || activeTab === 'scans'

  if (loading) return <PageLoader />

  const renderEmpty = (msg: string) => (
    <div className="flex flex-col items-center py-16 text-center">
      <div className="w-16 h-16 rounded-2xl bg-gray-100 flex items-center justify-center mb-4">
        <FileText className="w-7 h-7 text-gray-300" />
      </div>
      <p className="text-gray-500 font-medium">{msg}</p>
      <p className="text-sm text-gray-400 mt-1">No records found.</p>
    </div>
  )

  const inputCls = "w-full border border-gray-200 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary-400/40 focus:border-primary-400 transition"
  const labelCls = "block text-xs font-semibold text-gray-500 mb-1.5"

  const renderForm = () => {
    if (activeTab === 'allergies') return (
      <div className="space-y-4">
        <div><label className={labelCls}>Allergen Name *</label><input className={inputCls} value={allergyForm.allergenName} onChange={e => setAllergyForm(f => ({...f, allergenName: e.target.value}))} /></div>
        <div className="grid grid-cols-2 gap-3">
          <div><label className={labelCls}>Type</label><select className={inputCls} value={allergyForm.allergyType} onChange={e => setAllergyForm(f => ({...f, allergyType: e.target.value}))}><option>Drug</option><option>Food</option><option>Environmental</option><option>Other</option></select></div>
          <div><label className={labelCls}>Severity</label><select className={inputCls} value={allergyForm.severity} onChange={e => setAllergyForm(f => ({...f, severity: e.target.value}))}><option value="mild">Mild</option><option value="moderate">Moderate</option><option value="severe">Severe</option><option value="life_threatening">Life Threatening</option></select></div>
        </div>
        <div><label className={labelCls}>Reaction</label><textarea className={inputCls} rows={2} value={allergyForm.reactionDescription} onChange={e => setAllergyForm(f => ({...f, reactionDescription: e.target.value}))} /></div>
      </div>
    )
    if (activeTab === 'chronic') return (
      <div className="space-y-4">
        <div><label className={labelCls}>Disease Name *</label><input className={inputCls} value={chronicForm.diseaseName} onChange={e => setChronicForm(f => ({...f, diseaseName: e.target.value}))} /></div>
        <div className="grid grid-cols-2 gap-3">
          <div><label className={labelCls}>Type</label><input className={inputCls} value={chronicForm.diseaseType} onChange={e => setChronicForm(f => ({...f, diseaseType: e.target.value}))} /></div>
          <div><label className={labelCls}>Severity</label><select className={inputCls} value={chronicForm.severity} onChange={e => setChronicForm(f => ({...f, severity: e.target.value}))}><option value="mild">Mild</option><option value="moderate">Moderate</option><option value="severe">Severe</option></select></div>
        </div>
        <div><label className={labelCls}>Doctor Notes</label><textarea className={inputCls} rows={2} value={chronicForm.doctorNotes} onChange={e => setChronicForm(f => ({...f, doctorNotes: e.target.value}))} /></div>
      </div>
    )
    if (activeTab === 'medications') return (
      <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2 custom-scrollbar">
        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <label className={labelCls}>Medication Name *</label>
            <input className={inputCls} value={medForm.medicationName} onChange={e => setMedForm(f => ({...f, medicationName: e.target.value}))} placeholder="e.g. Panadol" />
          </div>
          <div>
            <label className={labelCls}>Generic Name</label>
            <input className={inputCls} value={medForm.genericName} onChange={e => setMedForm(f => ({...f, genericName: e.target.value}))} placeholder="e.g. Paracetamol" />
          </div>
          <div>
            <label className={labelCls}>Form</label>
            <select className={inputCls} value={medForm.form} onChange={e => setMedForm(f => ({...f, form: e.target.value}))}>
              {["Pill", "Syrup", "Injection", "Inhaler", "Cream", "Drops", "Patch", "Powder"].map(o => <option key={o}>{o}</option>)}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className={labelCls}>Dosage</label>
            <input className={inputCls} value={medForm.dosage} onChange={e => setMedForm(f => ({...f, dosage: e.target.value}))} placeholder="e.g. 500mg" />
          </div>
          <div>
            <label className={labelCls}>Frequency</label>
            <select className={inputCls} value={medForm.frequency} onChange={e => setMedForm(f => ({...f, frequency: e.target.value}))}>
              <option>Daily</option>
              <option>Specific Days</option>
              <option>Every Other Day</option>
            </select>
          </div>
          <div>
            <label className={labelCls}>Times/Day</label>
            <input type="number" className={inputCls} value={medForm.timesPerDay} onChange={e => setMedForm(f => ({...f, timesPerDay: Number(e.target.value)}))} />
          </div>
        </div>

        <div>
          <label className={labelCls}>Dose Times (comma separated)</label>
          <input className={inputCls} value={medForm.doseTimes} onChange={e => setMedForm(f => ({...f, doseTimes: e.target.value}))} placeholder="09:00, 21:00" />
        </div>

        {medForm.frequency === 'Specific Days' && (
          <div>
            <label className={labelCls}>Days of Week</label>
            <input className={inputCls} value={medForm.daysOfWeek} onChange={e => setMedForm(f => ({...f, daysOfWeek: e.target.value}))} placeholder="Monday, Wednesday, Friday" />
          </div>
        )}

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={labelCls}>Start Date</label>
            <input type="date" className={inputCls} value={medForm.startDate} onChange={e => setMedForm(f => ({...f, startDate: e.target.value}))} />
          </div>
          <div>
            <label className={labelCls}>End Date (Optional)</label>
            <input type="date" className={inputCls} value={medForm.endDate} onChange={e => setMedForm(f => ({...f, endDate: e.target.value}))} />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={labelCls}>Pills Remaining</label>
            <input type="number" className={inputCls} value={medForm.pillsRemaining} onChange={e => setMedForm(f => ({...f, pillsRemaining: Number(e.target.value)}))} />
          </div>
          <div>
            <label className={labelCls}>Refill Threshold</label>
            <input type="number" className={inputCls} value={medForm.refillThreshold} onChange={e => setMedForm(f => ({...f, refillThreshold: Number(e.target.value)}))} />
          </div>
        </div>

        <div className="flex items-center gap-4 py-1">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={medForm.isChronic} onChange={e => setMedForm(f => ({...f, isChronic: e.target.checked}))} className="w-4 h-4 rounded text-primary-600 focus:ring-primary-500" />
            <span className="text-sm font-medium text-gray-700">Chronic Medication</span>
          </label>
        </div>

        <div>
          <label className={labelCls}>Instructions</label>
          <textarea className={inputCls} rows={2} value={medForm.instructions} onChange={e => setMedForm(f => ({...f, instructions: e.target.value}))} placeholder="Take after meals" />
        </div>
      </div>
    )
    if (activeTab === 'vitals') return (
      <div className="space-y-4">
        <div><label className={labelCls}>Reading Type</label><select className={inputCls} value={vitalForm.readingType} onChange={e => setVitalForm(f => ({...f, readingType: e.target.value}))}><option>Blood Pressure</option><option>Blood Sugar</option><option>Heart Rate</option><option>Temperature</option><option>SpO2</option><option>Weight</option></select></div>
        <div className="grid grid-cols-3 gap-3">
          <div><label className={labelCls}>Value *</label><input type="number" className={inputCls} value={vitalForm.value} onChange={e => setVitalForm(f => ({...f, value: e.target.value}))} /></div>
          <div><label className={labelCls}>Value 2</label><input type="number" className={inputCls} value={vitalForm.value2} onChange={e => setVitalForm(f => ({...f, value2: e.target.value}))} placeholder="e.g. diastolic" /></div>
          <div><label className={labelCls}>Unit</label><input className={inputCls} value={vitalForm.unit} onChange={e => setVitalForm(f => ({...f, unit: e.target.value}))} /></div>
        </div>
        <div><label className={labelCls}>Notes</label><textarea className={inputCls} rows={2} value={vitalForm.notes} onChange={e => setVitalForm(f => ({...f, notes: e.target.value}))} /></div>
      </div>
    )
    if (activeTab === 'surgeries') return (
      <div className="space-y-4">
        <div><label className={labelCls}>Surgery Name *</label><input className={inputCls} value={surgeryForm.surgeryName} onChange={e => setSurgeryForm(f => ({...f, surgeryName: e.target.value}))} /></div>
        <div className="grid grid-cols-2 gap-3">
          <div><label className={labelCls}>Date</label><input type="date" className={inputCls} value={surgeryForm.surgeryDate} onChange={e => setSurgeryForm(f => ({...f, surgeryDate: e.target.value}))} /></div>
          <div><label className={labelCls}>Hospital</label><input className={inputCls} value={surgeryForm.hospitalName} onChange={e => setSurgeryForm(f => ({...f, hospitalName: e.target.value}))} /></div>
        </div>
        <div><label className={labelCls}>Notes</label><textarea className={inputCls} rows={2} value={surgeryForm.notes} onChange={e => setSurgeryForm(f => ({...f, notes: e.target.value}))} /></div>
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
    <div className="min-h-screen bg-slate-50/50 dark:bg-transparent p-6 text-slate-900 dark:text-slate-100">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <button onClick={() => navigate(-1)} className="p-2 hover:bg-gray-100 dark:hover:bg-slate-800 rounded-xl transition">
          <ChevronLeft className="w-5 h-5 text-gray-600 dark:text-slate-400" />
        </button>
        <div>
          <h1 className="text-xl font-bold text-gray-900 dark:text-white">{patient?.fullName || 'Patient'}</h1>
          <p className="text-sm text-gray-500 dark:text-slate-400">{age ? `${age} years` : 'N/A'} / {patient?.gender || 'N/A'} / {medProfile?.bloodType || 'No blood type'}</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {TABS.map(tab => {
          const Icon = tab.icon
          const active = activeTab === tab.id
          return (
            <button key={tab.id} onClick={() => { setActiveTab(tab.id); setShowForm(false) }}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium whitespace-nowrap transition-all ${
                active ? 'bg-primary-600 text-white shadow-lg shadow-primary-500/25' : 'bg-white dark:bg-slate-900 text-gray-600 dark:text-slate-350 hover:bg-gray-100 dark:hover:bg-slate-800 border border-gray-200 dark:border-slate-800/80'
              }`}>
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Content */}
      <Card className="relative min-h-[400px]">
        <div className="p-6">
          {activeTab === 'profile' && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {[
                  { l: 'Full Name', v: patient?.fullName },
                  { l: 'Email', v: patient?.email },
                  { l: 'Phone', v: patient?.phoneNumber },
                  { l: 'Blood Type', v: medProfile?.bloodType },
                  { l: 'Weight', v: medProfile?.weightKg ? `${medProfile.weightKg} kg` : null },
                  { l: 'Height', v: medProfile?.heightCm ? `${medProfile.heightCm} cm` : null },
                  { l: 'Smoking', v: medProfile?.isSmoker ? `Yes (${medProfile.smokingDetails || 'Daily'})` : 'No' },
                  { l: 'Emergency Contact', v: medProfile?.emergencyContactName },
                  { l: 'Emergency Phone', v: medProfile?.emergencyContactPhone },
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
                      <h3 className="text-lg font-bold text-gray-900 dark:text-white">AI Health Insights</h3>
                      <p className="text-xs text-gray-500 dark:text-slate-400">Automated analysis of patient history</p>
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
                    {analyzing ? 'Analyzing...' : (medProfile?.aiDiagnosisSummary ? 'Refresh Analysis' : 'Generate Analysis')}
                  </Button>
                </div>
                
                {medProfile?.aiDiagnosisSummary ? (
                  <div className="w-full">
                    {(() => {
                      try {
                        const report = JSON.parse(medProfile.aiDiagnosisSummary);
                        return (
                          <div className="p-6 bg-purple-50/50 dark:bg-purple-950/10 rounded-2xl border border-purple-100 dark:border-purple-900/30">
                            <div className="flex items-center justify-between mb-3">
                              <span className="text-[10px] font-bold text-purple-600 dark:text-purple-400 uppercase tracking-widest bg-white dark:bg-slate-900 px-3 py-1 rounded-full border border-purple-100 dark:border-purple-900/30">AI Analysis Report</span>
                            </div>
                            <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap">{report.analysis_en}</p>
                          </div>
                        );
                      } catch (e) {
                        return <p className="text-sm text-gray-500 italic">Report data is currently being processed or unavailable.</p>;
                      }
                    })()}
                  </div>
                ) : (
                  <div className="p-8 bg-gray-50 rounded-2xl border border-dashed border-gray-200 text-center">
                    <Sparkles className="w-8 h-8 text-purple-300 mx-auto mb-3" />
                    <p className="text-sm text-gray-500">No health analysis has been generated for this patient yet.</p>
                    <p className="text-xs text-gray-400 mt-1">Click the button above to analyze the medical history.</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'allergies' && (allergies.length === 0 ? renderEmpty('No allergies') : (
            <div className="space-y-3">
              {allergies.map(a => (
                <div key={a.id} className="flex items-center justify-between p-4 bg-white border border-gray-100 rounded-xl hover:shadow-md transition">
                  <div>
                    <div className="flex items-center gap-2">
                      <p className="font-semibold text-gray-800">{a.allergenName}</p>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${sevColor(a.severity)}`}>{a.severity}</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{a.allergyType}{a.reactionDescription ? ` · ${a.reactionDescription}` : ''}</p>
                  </div>
                  <button onClick={() => handleDelete('allergy', a.id)} className="p-2 text-gray-400 hover:text-red-500 transition"><Trash2 className="w-4 h-4" /></button>
                </div>
              ))}
            </div>
          ))}

          {activeTab === 'chronic' && (chronic.length === 0 ? renderEmpty('No chronic diseases') : (
            <div className="space-y-3">
              {chronic.map(d => (
                <div key={d.id} className="flex items-center justify-between p-4 bg-white border border-gray-100 rounded-xl hover:shadow-md transition">
                  <div>
                    <div className="flex items-center gap-2">
                      <p className="font-semibold text-gray-800">{d.diseaseName}</p>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${sevColor(d.severity)}`}>{d.severity}</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{d.diseaseType}{d.doctorNotes ? ` · ${d.doctorNotes}` : ''}</p>
                  </div>
                  <button onClick={() => handleDelete('chronic', d.id)} className="p-2 text-gray-400 hover:text-red-500 transition"><Trash2 className="w-4 h-4" /></button>
                </div>
              ))}
            </div>
          ))}

          {activeTab === 'medications' && (medications.length === 0 ? renderEmpty('No medications') : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {medications.map(m => (
                <div key={m.id} className="p-4 bg-white border border-gray-100 rounded-xl hover:shadow-md transition">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <p className="font-bold text-gray-800 text-lg">{m.medicationName}</p>
                      {m.genericName && <p className="text-xs text-gray-400 font-medium">{m.genericName}</p>}
                    </div>
                    {m.isChronic && <span className="bg-amber-100 text-amber-700 text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">Chronic</span>}
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <div className="w-5 h-5 rounded-md bg-primary-50 flex items-center justify-center text-primary-600"><Heart className="w-3 h-3" /></div>
                      <span>{m.dosage} · {m.form}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <div className="w-5 h-5 rounded-md bg-blue-50 flex items-center justify-center text-blue-600"><Activity className="w-3 h-3" /></div>
                      <span>{m.frequency} · {m.timesPerDay}x/day ({m.doseTimes})</span>
                    </div>
                    {m.instructions && (
                      <p className="text-xs text-gray-500 bg-gray-50 p-2 rounded-lg border border-gray-100 italic">"{m.instructions}"</p>
                    )}
                  </div>
                  <div className="mt-3 pt-3 border-t border-gray-50 flex justify-between items-center">
                    <span className="text-[10px] text-gray-400">Added: {new Date(m.createdAt).toLocaleDateString()}</span>
                    <button onClick={() => handleDelete('medication', m.id)} className="text-gray-400 hover:text-red-500 transition"><Trash2 className="w-4 h-4" /></button>
                  </div>
                </div>
              ))}
            </div>
          ))}

          {activeTab === 'vitals' && (vitals.length === 0 ? renderEmpty('No vitals') : (
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
                    {d.fileUrl && <a href={d.fileUrl} target="_blank" rel="noreferrer" className="text-xs text-primary-600 font-medium hover:underline">View</a>}
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
            <Button variant="outline" onClick={() => setShowForm(false)}>Cancel</Button>
            <Button onClick={handleAdd} disabled={saving} className="bg-primary-600 hover:bg-primary-700 text-white">
              {saving ? <><Loader2 className="w-4 h-4 animate-spin mr-2" />Saving...</> : 'Save'}
            </Button>
          </>
        }
      >
        {renderForm()}
      </Modal>
    </div>
  )
}
