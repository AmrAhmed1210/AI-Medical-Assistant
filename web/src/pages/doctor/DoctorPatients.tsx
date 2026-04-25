import { useState } from 'react'
import { Search, User } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import { useDoctorPatients } from '@/hooks/useDoctor'
import { Modal } from '@/components/ui/Modal'
import { UrgencyBadge } from '@/components/ui/Badge'
import { Card } from '@/components/ui/Card'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import type { PatientSummaryDto } from '@/lib/types'
import { doctorApi } from '@/api/doctorApi'
import { formatDate } from '@/lib/utils'

export default function DoctorPatients() {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const { patients = [], isLoading } = useDoctorPatients(search)
  const [selected, setSelected] = useState<PatientSummaryDto | null>(null)
  const [targetPatient, setTargetPatient] = useState<PatientSummaryDto | null>(null)
  const [messageText, setMessageText] = useState('')
  const [sendingMessage, setSendingMessage] = useState(false)

  const handleSendMessage = async () => {
    if (!targetPatient?.email || !messageText.trim()) return

    setSendingMessage(true)
    try {
      const res = await doctorApi.messagePatient({
        patientEmail: targetPatient.email,
        message: messageText.trim(),
      })
      toast.success('Message sent')
      setMessageText('')
      setTargetPatient(null)
      // Redirect to chat and auto-open this session
      navigate(`/doctor/chat?sessionId=${res.sessionId}`)
    } catch (error: any) {
      toast.error(error?.response?.data?.message || 'Failed to send message')
    } finally {
      setSendingMessage(false)
    }
  }


  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-bold text-gray-800">Patients</h1>
        <p className="text-sm text-gray-500 mt-0.5">{patients.length} patients registered</p>
      </div>

      <Card>
        <div className="p-4 border-b border-gray-100">
          <div className="relative max-w-sm">
            <Search size={15} className="absolute top-1/2 -translate-y-1/2 left-3 text-gray-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by patient name..."
              className="w-full pl-9 pr-4 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30"
            />
          </div>
        </div>

        {isLoading ? <PageLoader /> : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-100">
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Patient</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Age / Gender</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Last Visit</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Appointments</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Risk Level</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Actions</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-gray-50">
                {patients.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-16 text-center text-gray-400 text-sm">
                      No results found
                    </td>
                  </tr>
                ) : patients.map((p) => {
                  const age = p.dateOfBirth
                    ? Math.floor(
                        (new Date().getTime() - new Date(p.dateOfBirth).getTime()) /
                        (365.25 * 24 * 60 * 60 * 1000)
                      )
                    : undefined

                  return (
                    <tr
                      key={p.id}
                      onClick={() => setSelected(p)}
                      className="hover:bg-gray-50/80 cursor-pointer transition-colors"
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center overflow-hidden">
                            {p.photoUrl ? (
                              <img src={p.photoUrl} alt="" className="w-full h-full object-cover" />
                            ) : (
                              <User size={14} className="text-green-600" />
                            )}
                          </div>
                          <div>
                            <p className="font-medium text-gray-800">{p.fullName}</p>
                            {p.phoneNumber && (
                              <p className="text-xs text-gray-400">{p.phoneNumber}</p>
                            )}
                          </div>
                        </div>
                      </td>

                      <td className="px-4 py-3 text-gray-600">
                        {age ? `${age} years` : '-'} /{' '}
                        {p.gender === 'Male'
                          ? 'Male'
                          : p.gender === 'Female'
                          ? 'Female'
                          : '-'}
                      </td>

                      <td className="px-4 py-3 text-gray-600">
                        {p.lastVisit ? formatDate(p.lastVisit) : '-'}
                      </td>

                      <td className="px-4 py-3 text-gray-600">
                        {p.totalAppointments}
                      </td>

                      <td className="px-4 py-3">
                        <UrgencyBadge level="MEDIUM" />
                      </td>

                      <td className="px-4 py-3">
                        <div className="flex gap-2">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              setTargetPatient(p)
                              setMessageText('')
                            }}
                            className="px-3 py-1.5 text-xs font-medium rounded-lg border border-primary-200 text-primary-700 hover:bg-primary-50"
                          >
                            Send Message
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Modal
        open={!!selected}
        onClose={() => setSelected(null)}
        title="Patient Details"
        size="md"
      >
        {selected && (
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center overflow-hidden">
                {selected.photoUrl ? (
                  <img src={selected.photoUrl} alt="" className="w-full h-full object-cover" />
                ) : (
                  <User size={28} className="text-green-600" />
                )}
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-800">
                  {selected.fullName}
                </h3>
                {selected.email && (
                  <p className="text-sm text-gray-500">{selected.email}</p>
                )}
                {selected.phoneNumber && (
                  <p className="text-sm text-gray-500">{selected.phoneNumber}</p>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 border-t border-gray-100 pt-4">
              <div>
                <p className="text-xs text-gray-400">Age</p>
                <p className="font-medium">
                  {selected.dateOfBirth
                    ? Math.floor(
                        (new Date().getTime() - new Date(selected.dateOfBirth).getTime()) /
                        (365.25 * 24 * 60 * 60 * 1000)
                      ) + ' years'
                    : 'Not set'}
                </p>
              </div>

              <div>
                <p className="text-xs text-gray-400">Gender</p>
                <p className="font-medium">
                  {selected.gender === 'Male'
                    ? 'Male'
                    : selected.gender === 'Female'
                    ? 'Female'
                    : 'Not set'}
                </p>
              </div>

              <div>
                <p className="text-xs text-gray-400">Total Appointments</p>
                <p className="font-medium">{selected.totalAppointments}</p>
              </div>

              <div>
                <p className="text-xs text-gray-400">Last Visit</p>
                <p className="font-medium">
                  {selected.lastVisit ? formatDate(selected.lastVisit) : 'None'}
                </p>
              </div>

              <div>
                <p className="text-xs text-gray-400 mb-1">Blood Type</p>
                <p className="font-medium">{selected.bloodType ?? 'Not set'}</p>
              </div>

              <div>
                <p className="text-xs text-gray-400 mb-1">Allergies</p>
                <p className="font-medium">{selected.allergies ?? 'None'}</p>
              </div>
            </div>
          </div>
        )}
      </Modal>

      <Modal
        open={!!targetPatient}
        onClose={() => {
          if (!sendingMessage) setTargetPatient(null)
        }}
        title="Send Message"
        size="md"
      >
        {targetPatient && (
          <div className="space-y-4">
            <p className="text-sm text-gray-600">
              Send a direct message to <span className="font-semibold text-gray-800">{targetPatient.fullName}</span>
            </p>
            <textarea
              value={messageText}
              onChange={(e) => setMessageText(e.target.value)}
              rows={5}
              placeholder="Write your message..."
              className="w-full rounded-xl border border-gray-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-400/30"
            />
            <div className="flex justify-end gap-2">
              <button
                type="button"
                className="px-4 py-2 text-sm rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-50"
                onClick={() => setTargetPatient(null)}
                disabled={sendingMessage}
              >
                Cancel
              </button>
              <button
                type="button"
                className="px-4 py-2 text-sm rounded-lg bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-60"
                onClick={handleSendMessage}
                disabled={sendingMessage || !messageText.trim()}
              >
                {sendingMessage ? 'Sending...' : 'Send'}
              </button>
            </div>
          </div>
        )}
      </Modal>

    </div>
  )
}
