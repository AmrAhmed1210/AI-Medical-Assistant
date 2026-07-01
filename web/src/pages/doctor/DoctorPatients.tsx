import { useState, useEffect, useMemo } from 'react'
import { Search, User, Calendar, ChevronRight, ChevronLeft, MessageSquare, FileText, Clock } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useLanguage } from '@/lib/language'
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
  const { t, isRTL } = useLanguage()
  const [search, setSearch] = useState('')
  const { patients = [], isLoading: loadingPatients } = useDoctorPatients(search)
  const [selected, setSelected] = useState<PatientSummaryDto | null>(null)
  const [targetPatient, setTargetPatient] = useState<PatientSummaryDto | null>(null)
  const [messageText, setMessageText] = useState('')
  const [sendingMessage, setSendingMessage] = useState(false)

  const [appointments, setAppointments] = useState<AppointmentDto[]>([])
  const [loadingAppts, setLoadingAppts] = useState(true)
  const [viewMode, setViewMode] = useState<'byDay' | 'allPatients'>('byDay')
  const [selectedDayKey, setSelectedDayKey] = useState<string>('')

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
    if (key === 'unknown') return 'No Scheduled Date'
    const d = new Date(`${key}T00:00:00`)
    const now = new Date()
    const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
    if (key === today) return 'Today'

    return d.toLocaleDateString('en-US', {
      weekday: 'long',
      day: 'numeric',
      month: 'long',
      year: 'numeric',
    })
  }

  const filteredAppointments = useMemo(() => {
    // Only show Confirmed appointments
    let result = appointments.filter(a => a.status === 'Confirmed')
    
    if (search) {
      const q = search.toLowerCase()
      result = result.filter(a => a.patientName?.toLowerCase().includes(q))
    }
    return result
  }, [appointments, search])

  const groupedByDay = useMemo(() => {
    const grouped: Record<string, AppointmentDto[]> = {}
    for (const appt of filteredAppointments) {
      const key = toDayKey(appt.scheduledAt)
      if (!grouped[key]) grouped[key] = []
      grouped[key].push(appt)
    }
    const now = new Date()
    const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
    return Object.entries(grouped)
      .sort(([a], [b]) => a.localeCompare(b)) // sort chronologically ascending
      .map(([key, appts]) => ({
        key,
        label: toDayLabel(key),
        appointments: appts,
        isToday: key === today,
      }))
  }, [filteredAppointments])

  useEffect(() => {
    if (groupedByDay.length === 0) {
      setSelectedDayKey('')
      return
    }
    const d = new Date()
    const today = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
    const hasToday = groupedByDay.some(g => g.key === today)
    const nextSelected = hasToday ? today : groupedByDay[0].key
    if (!selectedDayKey || !groupedByDay.some(g => g.key === selectedDayKey)) {
      setSelectedDayKey(nextSelected)
    }
  }, [groupedByDay, selectedDayKey])

  const handleSendMessage = async () => {
    if (!targetPatient?.email || !messageText.trim()) return
    setSendingMessage(true)
    try {
      const res = await doctorApi.messagePatient({
        patientEmail: targetPatient.email,
        message: messageText.trim(),
      })
      toast.success(t('msgSent'))
      setMessageText('')
      setTargetPatient(null)
      navigate(`/doctor/chat?sessionId=${res.sessionId}`)
    } catch (error: any) {
      toast.error(error?.response?.data?.message || t('errMsgSent'))
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

  const hasAppointmentToday = (patientId: string | number) => {
    const d = new Date()
    const todayKey = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
    return appointments.some(a => 
      String(a.patientId) === String(patientId) && 
      toDayKey(a.scheduledAt) === todayKey
    )
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-800 dark:text-white">{t('patientsAndAppts')}</h1>
          <p className="text-sm text-gray-500 dark:text-slate-400 mt-0.5">
            {patients.length} {t('registeredPatients')}, {appointments.length} {t('registeredAppts')}
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
              placeholder={t("searchPatientName")}
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
                <p className="text-gray-400 font-medium">{t('noApptsScheduled')}</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Horizontal Tabs */}
                <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
                  {groupedByDay.map(section => (
                    <button
                      key={section.key}
                      onClick={() => setSelectedDayKey(section.key)}
                      className={`whitespace-nowrap px-4 py-2 text-sm font-medium rounded-xl transition-all border ${
                        selectedDayKey === section.key
                          ? 'bg-primary-600 text-white shadow-lg shadow-primary-500/30 border-primary-500'
                          : 'bg-white/50 dark:bg-slate-800/50 border-gray-200/50 dark:border-slate-700/50 text-gray-600 dark:text-slate-300 hover:bg-white dark:hover:bg-slate-800'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span>{section.label}</span>
                        <span className={`px-2 py-0.5 text-[10px] rounded-full ${
                          selectedDayKey === section.key
                            ? 'bg-white/20 text-white'
                            : 'bg-gray-100 dark:bg-slate-800 text-gray-500 dark:text-slate-400'
                        }`}>
                          {section.appointments.length}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>

                {/* Appointments for selected tab */}
                {groupedByDay.filter(s => s.key === selectedDayKey).map((section) => (
                  <div key={section.key}>
                    <div className="space-y-3">
                      {section.appointments.map((appt) => (
                        <div
                          key={appt.id}
                          className={`flex items-center gap-4 p-4 rounded-2xl border transition-all hover:shadow-xl cursor-pointer ${section.isToday 
                            ? 'bg-white/80 dark:bg-slate-800/80 border-primary-100 dark:border-primary-900/30 shadow-sm' 
                            : 'bg-gray-50/50 dark:bg-slate-900/30 border-gray-100 dark:border-slate-800/50'
                            }`}
                          onClick={() => {
                            const p = patients.find(pt => String(pt.id) === String(appt.patientId))
                            if (p) setSelected(p)
                          }}
                        >
                          {/* Avatar */}
                          <div className="w-11 h-11 rounded-xl bg-primary-50 flex items-center justify-center shrink-0">
                            <User size={20} className="text-primary-600" />
                          </div>

                          {/* Info */}
                          <div className="flex-1 min-w-0">
                            <p className="font-semibold text-gray-800 dark:text-slate-200 truncate">{appt.patientName}</p>
                            <div className="flex items-center gap-2 mt-1.5">
                              <span className="flex items-center gap-1 text-xs text-gray-500 dark:text-slate-400">
                                <Clock className="w-3.5 h-3.5" />
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
                                if (hasAppointmentToday(appt.patientId)) {
                                  navigate(`/doctor/patients/${appt.patientId}/records`)
                                } else {
                                  toast.error('Profile access is restricted to the day of visit.')
                                }
                              }}
                              className={`p-2 rounded-xl transition ${hasAppointmentToday(appt.patientId) ? 'bg-primary-50 text-primary-600 hover:bg-primary-100 dark:bg-primary-950/30 dark:text-primary-400 dark:hover:bg-primary-900/40' : 'bg-gray-100 text-gray-400 cursor-not-allowed opacity-50'}`}
                              title={hasAppointmentToday(appt.patientId) ? "View Records" : t("recordsLocked")}
                            >
                              <FileText size={16} />
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
                              title={t('sendMessage')}
                            >
                              <MessageSquare size={16} />
                            </button>
                                      {isRTL ? (
            <ChevronLeft size={16} className="text-gray-300 dark:text-slate-600 ml-1" />
          ) : (
            <ChevronRight size={16} className="text-gray-300 dark:text-slate-600 ml-1" />
          )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          /* ===== ALL PATIENTS VIEW ===== */
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left" dir={isRTL ? "rtl" : "ltr"}>
              <thead>
                <tr className="bg-gray-50/80 border-b border-gray-100 dark:bg-slate-800/50 dark:border-slate-700/50 backdrop-blur-md">
                  <th className="px-4 py-4 text-left font-semibold text-gray-600 dark:text-slate-300">{t('patientName')}</th>
                  <th className="px-4 py-4 text-left font-semibold text-gray-600 dark:text-slate-300">{t('genderAndAge')}</th>
                  <th className="px-4 py-4 text-left font-semibold text-gray-600 dark:text-slate-300">{t('lastVisit')}</th>
                  <th className="px-4 py-4 text-left font-semibold text-gray-600 dark:text-slate-300">{t("appointments")}</th>
                  <th className="px-4 py-4 text-left font-semibold text-gray-600 dark:text-slate-300">{t('actions')}</th>
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
                      className="hover:bg-gray-50 dark:hover:bg-slate-800/40 cursor-pointer transition-colors"
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
                            disabled={!hasAppointmentToday(p.id)}
                            onClick={(e) => {
                              e.stopPropagation()
                              navigate(`/doctor/patients/${p.id}/records`)
                            }}
                            className={`px-3 py-1.5 text-xs font-medium rounded-lg shadow-md transition-all ${
                              hasAppointmentToday(p.id) 
                                ? 'bg-primary-600 text-white hover:bg-primary-700 shadow-primary-500/10' 
                                : 'bg-gray-200 text-gray-400 dark:bg-slate-800 dark:text-slate-500 cursor-not-allowed opacity-60'
                            }`}
                            title={!hasAppointmentToday(p.id) ? "Access restricted to visit day" : ""}
                          >
                            {hasAppointmentToday(p.id) ? 'Records' : 'Locked'}
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
        title={t("patientDetails")}
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
                <p className="text-xs text-gray-400">{t('age')}</p>
                <p className="font-medium">
                  {selected.dateOfBirth
                    ? Math.floor(
                      (new Date().getTime() - new Date(selected.dateOfBirth).getTime()) /
                      (365.25 * 24 * 60 * 60 * 1000)
                    ) + ' ' + t('years')
                    : 'Not set'}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-400">{t('gender')}</p>
                <p className="font-medium">{selected.gender || 'Not set'}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400">{t('totalAppts')}</p>
                <p className="font-medium">{selected.totalAppointments}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400">{t('lastVisit')}</p>
                <p className="font-medium">
                  {selected.lastVisit ? formatDate(selected.lastVisit) : 'None'}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-400 mb-1">{t('bloodType')}</p>
                <p className="font-medium">{selected.bloodType ?? 'Not set'}</p>
              </div>
              <div>
                <p className="text-xs text-gray-400 mb-1">{t('allergies')}</p>
                <p className="font-medium">{selected.allergies ?? 'None'}</p>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="flex gap-2 pt-2 border-t border-gray-100">
              <Button
                size="sm"
                className="flex-1 bg-primary-600 hover:bg-primary-700 text-white"
                disabled={!hasAppointmentToday(selected.id)}
                onClick={() => {
                  setSelected(null)
                  navigate(`/doctor/patients/${selected.id}/records`)
                }}
              >
                <FileText className="w-4 h-4 mr-1" />
                {hasAppointmentToday(selected.id) ? 'View Full Records' : 'Records Locked'}
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
        title={t('sendMessage')}
        size="md"
      >
        {targetPatient && (
          <div className="space-y-4">
            <p className="text-sm text-gray-600">
              {t('sendDirectMsg')} <span className="font-semibold text-gray-800">{targetPatient.fullName}</span>
            </p>
            <textarea
              value={messageText}
              onChange={(e) => setMessageText(e.target.value)}
              rows={5}
              placeholder={t("writeMsg")}
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
