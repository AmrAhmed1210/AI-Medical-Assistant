import { useParams, useNavigate } from 'react-router-dom'
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
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const visitId = Number(id)

  const { summary, isLoading } = useVisitSummary(visitId)

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
        <p className="text-gray-500">Visit summary not found</p>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6 print:p-0">
      {/* Actions bar */}
      <div className="flex items-center justify-between print:hidden">
        <Button variant="ghost" size="sm" onClick={() => navigate('/doctor/today')}>
          <ChevronLeft className="w-4 h-4" />
          Back
        </Button>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handlePrint}>
            <Printer className="w-4 h-4 ml-2" />
            Print
          </Button>
          <Button variant="outline" size="sm" onClick={handleDownloadPdf}>
            <Download className="w-4 h-4 ml-2" />
            Download PDF
          </Button>
        </div>
      </div>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="border-b pb-6"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Visit Medical Summary</h1>
            <div className="flex items-center gap-4 text-sm text-gray-500 mt-2">
              <span className="flex items-center gap-1">
                <Calendar className="w-4 h-4" />
                {summary.visitDate}
              </span>
              <span className="flex items-center gap-1">
                <CheckCircle2 className="w-4 h-4 text-green-500" />
                Closed
              </span>
            </div>
          </div>
          <div className="text-left">
            <p className="text-sm text-gray-500">Visit ID</p>
            <p className="font-bold text-gray-900">#{summary.id}</p>
          </div>
        </div>
      </motion.div>

      {/* Patient Header */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="bg-gray-50">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 bg-primary-100 rounded-full flex items-center justify-center">
              <User className="w-7 h-7 text-primary-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">{summary.patientName}</h2>
              <div className="flex items-center gap-3 text-sm text-gray-500 mt-1">
                <span>{summary.patientAge} yrs</span>
                <span className="flex items-center gap-1">
                  Blood type: <span className="font-medium text-red-600">{summary.bloodType}</span>
                </span>
              </div>
            </div>
          </div>

          {/* Allergies */}
          {summary.allergies?.length > 0 && (
            <div className="mt-4 pt-4 border-t">
              <p className="text-sm text-gray-500 mb-2 flex items-center gap-1">
                <AlertTriangle className="w-4 h-4 text-red-500" />
                Allergies:
              </p>
              <div className="flex flex-wrap gap-2">
                {summary.allergies.map((a, idx) => (
                  <span
                    key={idx}
                    className={`px-3 py-1 rounded-full text-xs font-medium ${a.severity === 'life_threatening'
                      ? 'bg-red-100 text-red-700 border border-red-300'
                      : a.severity === 'severe'
                        ? 'bg-orange-100 text-orange-700'
                        : 'bg-gray-100 text-gray-600'
                      }`}
                  >
                    {a.allergenName} ({a.reaction})
                  </span>
                ))}
              </div>
            </div>
          )}
        </Card>
      </motion.div>

      {/* Visit Details */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="space-y-4"
      >
        {/* Chief Complaint */}
        <Card>
          <h3 className="font-bold text-gray-900 mb-3 flex items-center gap-2">
            <Stethoscope className="w-5 h-5 text-primary-600" />
            Chief Complaint
          </h3>
          <p className="text-gray-700">{summary.chiefComplaint || '—'}</p>
        </Card>

        {/* Examination */}
        {summary.examinationFindings && (
          <Card>
            <h3 className="font-bold text-gray-900 mb-3">Examination Findings</h3>
            <p className="text-gray-700 whitespace-pre-line">{summary.examinationFindings}</p>
          </Card>
        )}

        {/* Assessment */}
        {summary.assessment && (
          <Card>
            <h3 className="font-bold text-gray-900 mb-3">Assessment / Diagnosis</h3>
            <p className="text-gray-700">{summary.assessment}</p>
          </Card>
        )}

        {/* Plan */}
        {summary.plan && (
          <Card>
            <h3 className="font-bold text-gray-900 mb-3">Treatment Plan</h3>
            <p className="text-gray-700">{summary.plan}</p>
          </Card>
        )}

        {/* Vital Signs */}
        {summary.vitalSigns?.length > 0 && (
          <Card>
            <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary-600" />
              Vital Signs
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {summary.vitalSigns.map((vital, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded-lg text-center ${vital.isAbnormal ? 'bg-red-50 border border-red-200' : 'bg-gray-50'
                    }`}
                >
                  <p className="text-xs text-gray-500">{vital.type}</p>
                  <p className={`font-bold text-lg mt-1 ${vital.isAbnormal ? 'text-red-600' : 'text-gray-900'}`}>
                    {vital.value}
                    {vital.value2 && ` / ${vital.value2}`}
                  </p>
                  <p className="text-xs text-gray-400">{vital.unit}</p>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* Symptoms */}
        {summary.symptoms?.length > 0 && (
          <Card>
            <h3 className="font-bold text-gray-900 mb-3">Symptoms</h3>
            <div className="space-y-2">
              {summary.symptoms.map((sym, idx) => (
                <div key={idx} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
                  <span className="font-medium">{sym.name}</span>
                  <div className="flex items-center gap-2 text-sm text-gray-500">
                    <span>{sym.severity === 'severe' ? 'Severe' : sym.severity === 'moderate' ? 'Moderate' : 'Mild'}</span>
                    {sym.location && <span>— {sym.location}</span>}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* Prescriptions */}
        {summary.prescriptions?.length > 0 && (
          <Card>
            <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Pill className="w-5 h-5 text-primary-600" />
              Prescription
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 text-gray-500">
                  <tr>
                    <th className="text-left p-3 rounded-l-lg">Medication</th>
                    <th className="text-left p-3">Dosage</th>
                    <th className="text-left p-3">Frequency</th>
                    <th className="text-left p-3">Duration</th>
                    <th className="text-left p-3 rounded-r-lg">Type</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.prescriptions.map((pres, idx) => (
                    <tr key={idx} className="border-b">
                      <td className="p-3 font-medium">{pres.medicationName}</td>
                      <td className="p-3 text-gray-600">{pres.dosage}</td>
                      <td className="p-3 text-gray-600">{pres.frequency}</td>
                      <td className="p-3 text-gray-600">{pres.duration}</td>
                      <td className="p-3">
                        {pres.isChronic ? (
                          <span className="px-2 py-0.5 bg-teal-100 text-teal-700 rounded-full text-xs font-medium">
                            Chronic
                          </span>
                        ) : (
                          <span className="text-gray-400">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}

        {/* Follow-up */}
        {summary.followUpRequired && (
          <Card className="bg-blue-50 border-blue-200">
            <div className="flex items-start gap-3">
              <Calendar className="w-5 h-5 text-blue-600 mt-0.5" />
              <div>
                <h3 className="font-bold text-blue-900">Follow-up Required</h3>
                <p className="text-blue-700 text-sm mt-1">
                  After {summary.followUpAfterDays} day(s)
                </p>
                {summary.followUpNotes && (
                  <p className="text-blue-600 text-sm mt-1">{summary.followUpNotes}</p>
                )}
              </div>
            </div>
          </Card>
        )}

        {/* Notes */}
        {summary.notes && (
          <Card>
            <h3 className="font-bold text-gray-900 mb-3">Notes</h3>
            <p className="text-gray-700 whitespace-pre-line">{summary.notes}</p>
          </Card>
        )}
      </motion.div>

      {/* Footer */}
      <div className="text-center text-xs text-gray-400 pt-6 border-t print:mt-8">
        <p>MedBook — Electronic Patient Management System</p>
        <p className="mt-1">This summary was auto-generated — not a substitute for professional medical advice</p>
      </div>
    </div>
  )
}
