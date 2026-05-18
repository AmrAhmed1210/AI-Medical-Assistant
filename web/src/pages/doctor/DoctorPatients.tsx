import { useState, useEffect, useMemo } from 'react'
import { Search, User, Calendar, ChevronRight, MessageSquare, FileText, Clock } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import { useDoctorPatients } from '@/hooks/useDoctor'
import { doctorApi } from '@/api/doctorApi'
import { Modal } from '@/components/ui/Modal'
import { UrgencyBadge } from '@/components/ui/Badge'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import type { PatientSummaryDto, AppointmentDto } from '@/lib/types'
import { formatDate, formatDateTime } from '@/lib/utils'

export default function DoctorPatients() {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const { patients = [], isLoading: loadingPatients } = useDoctorPatients(search)
  const [selected, setSelected] = useState<PatientSummaryDto | null>(null)
  const [targetPatient, setTargetPatient] = useState<PatientSummaryDto | null>(null)
  const [messageText, setMessageText] = useState('')
  const [sendingMessage, setSendingMessage] = useState(false)

  // Fetch all appointments for grouping by day
  const [appointments, setAppointments] = useState<AppointmentDto[]>([])
  const [loadingAppts, setLoadingAppts] = useState(true)
  const [viewMode, setViewMode] = useState<'byDay' | 'allPatients'>('byDay')

  useEffect(() => {
    const fetchAppts = async () => {
      setLoadingAppts(true)
      try {
        const data = await doctorApi.getAppointments()
        setAppointments(data)
      } catch {
        setAppointments([])
      } finally {
        setLoadingAppts(false)
      }
    }
    fetchAppts()
  }, [])

  // Group appointments by day
  const toDayKey = (dateStr: string) => {
    const d = new Date(dateStr)
    if (isNaN(d.getTime())) return 'unknown'
    const y = d.getFullYear()
    const m = `${d.getMonth() + 1}`.padStart(2, '0')
    const day = `${d.getDate()}`.padStart(2, '0')
    return `${y}-${m}-${day}`
  }

  const toDayLabel = (key: string) => {
    if (key === 'unknown') return 'Without Day'
    const d = new Date(`${key}T00:00:00`)
    const today = toDayKey(new Date().toISOString())
    if (key === today) return 'Day'

    return d.toLocaleDateString('en-US', {
      weekday: 'long',
      day: 'numeric',
      month: 'long',
      year: 'numeric',
    })
  }

  const filteredAppointments = useMemo(() => {
    if (!search) return appointments
    const q = search.toLowerCase()
    return appointments.filter(a =>
      a.patientName?.toLowerCase().includes(q)
    )
  }, [appointments, search])

  const groupedByDay = useMemo(() => {
    const grouped: Record<string, AppointmentDto[]> = {}
    for (const appt of filteredAppointments) {
      const key = toDayKey(appt.scheduledAt)
      if (!grouped[key]) grouped[key] = []
      grouped[key].push(appt)
    }
    return Object.entries(grouped)
      .sort(([a], [b]) => b.localeCompare(a)) // newest first
      .map(([key, appts]) => ({
        key,
        label: toDayLabel(key),
        appointments: appts,
        isToday: key === toDayKey(new Date().toISOString()),
      }))
  }, [filteredAppointments])

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
      navigate(`/doctor/chat?sessionId=${res.sessionId}`)
    } catch (error: any) {
      toast.error(error?.response?.data?.message || 'Failed to send message')
    } finally {
      setSendingMessage(false)
    }
  }

  const isLoading = loadingPatients || loadingAppts

  const statusColor = (status: string) => {
    switch (status) {
      case 'Confirmed': return 'bg-blue-100 text-blue-700'
      case 'Completed': return 'bg-emerald-100 text-emerald-700'
      case 'Cancelled': return 'bg-red-100 text-red-600'
      case 'Pending': return 'bg-amber-100 text-amber-700'
      default: return 'bg-gray-100 text-gray-600'
    }
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-800 dark:text-white">Patients and Appointments</h1>
          <p className="text-sm text-gray-500 dark:text-slate-400 mt-0.5">
            {patients.length} Patient{patients.length !== 1 ? 's' : ''} registered, {appointments.length} Appointment{appointments.length !== 1 ? 's' : ''}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewMode('byDay')}
            className={`px-3 py-1.5 text-xs font-bold rounded-xl transition-all ${viewMode === 'byDay'
              ? 'bg-primary-600 text-white shadow-lg shadow-primary-500/20'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700'
              }`}
          >
            <Calendar className="w-3 h-3 inline mr-1" />
            Day
          </button>
          <button
            onClick={() => setViewMode('allPatients')}
            className={`px-3 py-1.5 text-xs font-bold rounded-xl transition-all ${viewMode === 'allPatients'
              ? 'bg-primary-600 text-white shadow-lg shadow-primary-500/20'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700'
              }`}
          >
            <User className="w-3 h-3 inline mr-1" />
            All Patients
          </button>
        </div>
      </div>

      {/* Search */}
      <Card>
        <div className="p-4 border-b border-gray-100 dark:border-slate-800">
          <div className="relative max-w-sm">
            <Search size={15} className="absolute top-1/2 -translate-y-1/2 left-3 text-gray-400 dark:text-slate-500" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search patient name..."
              className="w-full pl-9 pr-4 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 dark:bg-slate-900 dark:border-slate-800 dark:text-white"
            />
          </div>
        </div>

        {isLoading ? <PageLoader /> : viewMode === 'byDay' ? (
          /* ===== BY DAY VIEW ===== */
          <div className="p-4 space-y-6">
            {groupedByDay.length === 0 ? (
              <div className="text-center py-16">
                <Calendar className="w-12 h-12 text-gray-200 mx-auto mb-3" />
                <p className="text-gray-400 font-medium">No appointments scheduled</p>
              </div>
            ) : (
              groupedByDay.map((section) => (
                <div key={section.key}>
                  {/* Day Header */}
                  <div className="flex items-center gap-3 mb-3">
                    <span className={`text-xs font-bold px-3 py-1 rounded-full ${section.isToday
                      ? 'bg-primary-600 text-white shadow-md shadow-primary-500/10'
                      : 'bg-gray-200 text-gray-600 dark:bg-slate-800 dark:text-slate-400'
                      }`}>
                      {section.label}
                    </span>
                    <span className="text-[10px] text-gray-400 dark:text-slate-500 font-medium">
                      {section.appointments.length} appointment{section.appointments.length > 1 ? 's' : ''}
                    </span>
                    <div className="h-px bg-gray-100 dark:bg-slate-800 flex-1" />
                  </div>

                  {/* Appointments in this day */}
                  <div className="space-y-2">
                    {section.appointments.map((appt) => (
                      <div
                        key={appt.id}
                        className={`flex items-center gap-4 p-4 rounded-2xl border transition-all hover:shadow-md cursor-pointer ${section.isToday 
                          ? 'bg-white dark:bg-slate-900/60 border-gray-100 dark:border-slate-800/80' 
                          : 'bg-gray-50/50 dark:bg-slate-950/20 border-gray-100 dark:border-slate-800/50'
                          }`}
                        onClick={() => {
                          const p = patients.find(pt => String(pt.id) === String(appt.patientId))
                          if (p) setSelected(p)
                        }}
                      >
                        {/* Avatar */}
                        <div className="w-10 h-10 rounded-xl bg-primary-50 flex items-center justify-center shrink-0">
                          <User size={18} className="text-primary-600" />
                        </div>

                        {/* Info */}
                        <div className="flex-1 min-w-0">
                          <p className="font-semibold text-gray-800 dark:text-slate-200 truncate">{appt.patientName}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="flex items-center gap-1 text-xs text-gray-500 dark:text-slate-400">
                              <Clock className="w-3 h-3" />
                              {new Date(appt.scheduledAt).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}
                            </span>
                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${statusColor(appt.status)}`}>
                              {appt.status}
                            </span>
                          </div>
                        </div>

                        {/* Actions */}
                        <div className="flex items-center gap-2 shrink-0">
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              navigate(`/doctor/patients/${appt.patientId}/records`)
                            }}
                            className="p-2 rounded-xl bg-primary-50 text-primary-600 hover:bg-primary-100 dark:bg-primary-950/30 dark:text-primary-400 dark:hover:bg-primary-900/40 transition"
                            title="View Records"
                          >
                            <FileText size={14} />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              const p = patients.find(pt => String(pt.id) === String(appt.patientId))
                              if (p) {
                                setTargetPatient(p)
                                setMessageText('')
                              }
                            }}
                            className="p-2 rounded-xl bg-blue-50 text-blue-600 hover:bg-blue-100 dark:bg-blue-950/30 dark:text-blue-400 dark:hover:bg-blue-900/40 transition"
                            title="Send Message"
                          >
                            <MessageSquare size={14} />
                          </button>
                          <ChevronRight size={14} className="text-gray-300 dark:text-slate-600" />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))
            )}
          </div>
        ) : (
          /* ===== ALL PATIENTS VIEW ===== */
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left" dir="ltr">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-100 dark:bg-slate-900/40 dark:border-slate-800/80">
                  <th className="px-4 py-3 text-left font-semibold text-gray-600 dark:text-slate-400">Patient name</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600 dark:text-slate-400">Gender and age</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600 dark:text-slate-400">Last visit</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600 dark:text-slate-400">Appointments</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600 dark:text-slate-400">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50 dark:divide-slate-800/50">
                {patients.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-16 text-center text-gray-400 text-sm">
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
                      className="hover:bg-gray-50/80 dark:hover:bg-slate-900/40 cursor-pointer transition-colors"
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-950/30 flex items-center justify-center overflow-hidden">
                            {p.photoUrl ? (
                              <img src={p.photoUrl} alt="" className="w-full h-full object-cover" />
                            ) : (
                              <User size={14} className="text-green-600 dark:text-green-400" />
                            )}
                          </div>
                          <div>
                            <p className="font-medium text-gray-800 dark:text-slate-200">{p.fullName}</p>
                            {p.phoneNumber && (
                              <p className="text-xs text-gray-400 dark:text-slate-500">{p.phoneNumber}</p>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-gray-600 dark:text-slate-300">
                        {age ? `${age} years` : '-'} / {p.gender || '-'}
                      </td>
                      <td className="px-4 py-3 text-gray-600 dark:text-slate-300">
                        {p.lastVisit ? formatDate(p.lastVisit) : '-'}
                      </td>
                      <td className="px-4 py-3 text-gray-600 dark:text-slate-300">{p.totalAppointments}</td>
                      <td className="px-4 py-3">
                        <div className="flex gap-2">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              navigate(`/doctor/patients/${p.id}/records`)
                            }}
                            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-primary-600 text-white hover:bg-primary-700 shadow-md shadow-primary-500/10"
                          >
                            Records
                          </button>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              setTargetPatient(p)
                              setMessageText('')
                            }}
                            className="px-3 py-1.5 text-xs font-medium rounded-lg border border-primary-200 text-primary-700 hover:bg-primary-50 dark:border-primary-800/80 dark:text-primary-400 dark:hover:bg-primary-950/20"
                          >
                            Message
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

      {/* Patient Details Modal */}
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
                <p className="font-medium">{selected.gender || 'Not set'}</p>
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

            {/* Quick Actions */}
            <div className="flex gap-2 pt-2 border-t border-gray-100">
              <Button
                size="sm"
                className="flex-1 bg-primary-600 hover:bg-primary-700 text-white"
                onClick={() => {
                  setSelected(null)
                  navigate(`/doctor/patients/${selected.id}/records`)
                }}
              >
                <FileText className="w-4 h-4 mr-1" />
                View Full Records
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="flex-1"
                onClick={() => {
                  setSelected(null)
                  setTargetPatient(selected)
                  setMessageText('')
                }}
              >
                <MessageSquare className="w-4 h-4 mr-1" />
                Send Message
              </Button>
            </div>
          </div>
        )}
      </Modal>

      {/* Send Message Modal */}
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
