import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Send, Calendar, FileText, Stethoscope, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { consultApi } from '@/api/consultApi'
import toast from 'react-hot-toast'
import type { PatientSummaryDto } from '@/lib/types'

interface SendConsultationModalProps {
  isOpen: boolean
  onClose: () => void
  patient: PatientSummaryDto | null
  doctorName: string
}

export default function SendConsultationModal({ isOpen, onClose, patient, doctorName }: SendConsultationModalProps) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [date, setDate] = useState('')
  const [time, setTime] = useState('')
  const [isSending, setIsSending] = useState(false)

  const resetForm = () => {
    setTitle('')
    setDescription('')
    setDate('')
    setTime('')
  }

  const handleClose = () => {
    if (!isSending) {
      resetForm()
      onClose()
    }
  }

  const handleSubmit = async () => {
    if (!patient) {
      toast.error('No patient selected')
      return
    }
    if (!title.trim()) {
      toast.error('Please enter a title')
      return
    }
    if (!date || !time) {
      toast.error('Please select a date and time')
      return
    }

    setIsSending(true)
    try {
      const scheduledAt = new Date(`${date}T${time}`).toISOString()
      
      await consultApi.sendConsultation({
        patientId: Number(patient.id),
        title: title.trim(),
        description: description.trim(),
        scheduledAt,
      })

      toast.success('Consultation sent successfully')
      resetForm()
      onClose()
    } catch (err: any) {
      toast.error(err?.response?.data?.message || 'Failed to send consultation')
    } finally {
      setIsSending(false)
    }
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <div className="absolute inset-0 bg-gray-900/40 backdrop-blur-sm" onClick={handleClose} />
          <motion.div
            className="relative bg-white rounded-2xl shadow-xl border border-gray-100 w-full max-w-lg overflow-hidden"
            initial={{ scale: 0.95, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 20 }}
            transition={{ duration: 0.2 }}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 bg-gradient-to-r from-primary-50 to-transparent">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl bg-primary-100 flex items-center justify-center">
                  <Stethoscope size={18} className="text-primary-600" />
                </div>
                <div>
                  <h3 className="text-base font-semibold text-gray-900">Send Consultation</h3>
                  {patient && (
                    <p className="text-xs text-gray-500">To: {patient.fullName}</p>
                  )}
                </div>
              </div>
              <button
                onClick={handleClose}
                disabled={isSending}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
              >
                <X size={18} className="text-gray-500" />
              </button>
            </div>

            {/* Form */}
            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1.5">
                  Title <span className="text-red-500">*</span>
                </label>
                <div className="relative">
                  <FileText size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="e.g., Follow-up Consultation"
                    className="w-full pl-10 pr-4 py-2.5 text-sm border border-gray-200 rounded-xl focus:ring-2 focus:ring-primary-400/30 focus:border-primary-400 outline-none transition-all bg-white"
                    disabled={isSending}
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1.5">Description</label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Consultation details, preparation notes, etc."
                  rows={3}
                  className="w-full px-4 py-2.5 text-sm border border-gray-200 rounded-xl focus:ring-2 focus:ring-primary-400/30 focus:border-primary-400 outline-none transition-all bg-white resize-none"
                  disabled={isSending}
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1.5">
                    Date <span className="text-red-500">*</span>
                  </label>
                  <div className="relative">
                    <Calendar size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                    <input
                      type="date"
                      value={date}
                      onChange={(e) => setDate(e.target.value)}
                      className="w-full pl-10 pr-4 py-2.5 text-sm border border-gray-200 rounded-xl focus:ring-2 focus:ring-primary-400/30 focus:border-primary-400 outline-none transition-all bg-white"
                      disabled={isSending}
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1.5">
                    Time <span className="text-red-500">*</span>
                  </label>
                  <div className="relative">
                    <Calendar size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                    <input
                      type="time"
                      value={time}
                      onChange={(e) => setTime(e.target.value)}
                      className="w-full pl-10 pr-4 py-2.5 text-sm border border-gray-200 rounded-xl focus:ring-2 focus:ring-primary-400/30 focus:border-primary-400 outline-none transition-all bg-white"
                      disabled={isSending}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-100 bg-gray-50/50">
              <Button
                variant="outline"
                onClick={handleClose}
                disabled={isSending}
                size="sm"
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                onClick={handleSubmit}
                disabled={isSending}
                size="sm"
                icon={isSending ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
              >
                {isSending ? 'Sending...' : 'Send Consultation'}
              </Button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
