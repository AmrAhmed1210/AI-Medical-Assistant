import { useEffect, useMemo, useState } from 'react'
import { useAppointmentStore } from '@/store/appointmentStore'
import { appointmentApi } from '@/api/appointmentApi'
import { doctorApi } from '@/api/doctorApi'
import { useAuthStore } from '@/store/authStore'
import { useNotificationStore } from '@/store/notificationStore'
import { AppointmentTable } from '@/components/doctor/AppointmentTable'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import type { AppointmentStatus, AppointmentDto } from '@/lib/types'
import toast from 'react-hot-toast'
import { RefreshCw, Calendar, Search, Trash2 } from 'lucide-react'

const STATUS_FILTERS: { label: string; value: AppointmentStatus | '' }[] = [
  { label: 'All', value: '' },
  { label: 'Pending', value: 'Pending' },
  { label: 'Confirmed', value: 'Confirmed' },
  { label: 'Completed', value: 'Completed' },
  { label: 'Cancelled', value: 'Cancelled' },
]

export default function DoctorAppointments() {
  const { appointments, isLoading, fetchAppointments, confirm, cancel, complete, clearHistory, updateLocal } = useAppointmentStore()
  const { addNotification } = useNotificationStore()
  const { token } = useAuthStore()
  
  const [activeTab, setActiveTab] = useState<'active' | 'history'>('active')
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<AppointmentStatus | ''>('')
  const [selectedDayKey, setSelectedDayKey] = useState('')
  const [showClearHistoryModal, setShowClearHistoryModal] = useState(false)

  useEffect(() => {
    fetchAppointments()
  }, [])

  const handleConfirm = async (id: string) => {
    try {
      await confirm(id)
      toast.success('Appointment confirmed')
    } catch { toast.error('Failed to confirm appointment') }
  }

  const handleUnconfirm = async (id: string) => {
    try {
      await appointmentApi.setPending(id)
      updateLocal(id, { status: 'Pending' })
      toast.success('Appointment moved to pending')
    } catch { toast.error('Failed to unconfirm appointment') }
  }

  const handleCancel = async (id: string) => {
    try {
      await cancel(id)
      toast.success('Appointment cancelled')
    } catch { toast.error('Failed to cancel appointment') }
  }

  const handleComplete = async (id: string) => {
    try {
      await complete(id)
      toast.success('Appointment marked as completed')
    } catch { toast.error('Failed to complete operation') }
  }

  const handleDelete = async (id: string) => {
    try {
      await appointmentApi.delete(id)
      updateLocal(id, { status: 'Cancelled' })
      toast.success('Appointment deleted')
    } catch {
      toast.error('Failed to delete appointment')
    }
  }

  const confirmClearHistory = async () => {
    try {
      await clearHistory()
      toast.success('History cleared successfully')
    } catch {
      toast.error('Failed to clear history')
    } finally {
      setShowClearHistoryModal(false)
    }
  }

  const toDayKey = (value: string | Date) => {
    const date = value instanceof Date ? value : new Date(value)
    if (Number.isNaN(date.getTime())) return 'unknown'
    const year = date.getFullYear()
    const month = `${date.getMonth() + 1}`.padStart(2, '0')
    const day = `${date.getDate()}`.padStart(2, '0')
    return `${year}-${month}-${day}`
  }

  const displayAppointments = useMemo(() => {
    let filtered = appointments
    if (activeTab === 'active') {
      // Active: Pending or Confirmed only (NOT Completed or Cancelled)
      filtered = filtered.filter(a => 
        a.status === 'Pending' || a.status === 'Confirmed'
      )
      if (searchQuery) {
        const q = searchQuery.toLowerCase()
        filtered = filtered.filter(a => 
          a.patientName?.toLowerCase().includes(q) ||
          a.scheduledAt?.toLowerCase().includes(q) ||
          a.status?.toLowerCase().includes(q)
        )
      }
    } else {
      // History: Completed or Cancelled
      filtered = filtered.filter(a => 
        a.status === 'Completed' || a.status === 'Cancelled'
      )
    }

    if (statusFilter) {
      filtered = filtered.filter(a => a.status === statusFilter)
    }

    return filtered
  }, [appointments, activeTab, searchQuery, statusFilter])

  const daySections = useMemo(() => {
    const grouped = displayAppointments.reduce<Record<string, AppointmentDto[]>>((acc, appt) => {
      const key = toDayKey(appt.scheduledAt)
      if (!acc[key]) acc[key] = []
      acc[key].push(appt)
      return acc
    }, {})

    return Object.entries(grouped)
      .sort(([a], [b]) => (a > b ? 1 : -1))
      .map(([key, dayAppointments]) => ({
        key,
        label: key === 'unknown'
          ? 'No Date'
          : new Date(`${key}T00:00:00`).toLocaleDateString(undefined, {
            weekday: 'short',
            day: '2-digit',
            month: 'short',
            year: 'numeric',
          }),
        appointments: dayAppointments,
      }))
  }, [displayAppointments])

  useEffect(() => {
    if (daySections.length === 0) {
      setSelectedDayKey('')
      return
    }
    const today = toDayKey(new Date())
    const hasToday = daySections.some((section) => section.key === today)
    const nextSelected = hasToday ? today : daySections[0].key
    if (!selectedDayKey || !daySections.some((section) => section.key === selectedDayKey)) {
      setSelectedDayKey(nextSelected)
    }
  }, [daySections, selectedDayKey])

  const selectedDayAppointments = daySections.find((section) => section.key === selectedDayKey)?.appointments ?? []

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-primary-600 to-primary-400 rounded-xl blur-lg opacity-30" />
            <div className="relative bg-gradient-to-br from-primary-600 to-primary-500 rounded-xl p-3 shadow-lg">
              <Calendar size={28} className="text-white" />
            </div>
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-800">Appointments</h1>
            <p className="text-sm text-gray-500 mt-0.5">Manage patient appointments</p>
          </div>
        </div>
        <Button variant="outline" icon={<RefreshCw size={14} />} onClick={fetchAppointments} size="sm">
          Refresh
        </Button>
      </div>

      <Card>
        <div className="flex flex-col sm:flex-row border-b border-gray-100">
          <div className="flex">
            <button
              onClick={() => setActiveTab('active')}
              className={`px-6 py-4 text-sm font-medium transition-colors border-b-2 ${
                activeTab === 'active'
                  ? 'border-primary-600 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Active
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`px-6 py-4 text-sm font-medium transition-colors border-b-2 ${
                activeTab === 'history'
                  ? 'border-primary-600 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              History
            </button>
          </div>
          
          <div className="flex-1 p-2 flex items-center justify-end gap-3 sm:border-l border-gray-100 bg-gray-50/50">
            {activeTab === 'active' && (
              <div className="relative">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                <input 
                  type="text" 
                  placeholder="Search patient, date, status..." 
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-8 pr-4 py-1.5 text-sm border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none w-64 bg-white"
                />
              </div>
            )}
            
            {activeTab === 'history' && (
              <Button variant="destructive" size="sm" icon={<Trash2 size={14} />} onClick={() => setShowClearHistoryModal(true)}>
                Clear History
              </Button>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 p-4 border-b border-gray-100 flex-wrap bg-white">
          {STATUS_FILTERS.map((f) => (
            <button
              key={f.value}
              onClick={() => setStatusFilter(f.value)}
              className={`px-3 py-1.5 text-xs rounded-xl font-medium transition-colors ${statusFilter === f.value
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        {daySections.length === 0 ? (
          <AppointmentTable
            appointments={[]}
            onConfirm={handleConfirm}
            onUnconfirm={handleUnconfirm}
            onCancel={handleCancel}
            onComplete={handleComplete}
            onDelete={handleDelete}
          />
        ) : (
          <div className="p-4 space-y-4">
            <div className="flex items-center gap-2 flex-wrap">
              {daySections.map((section) => {
                const hasPending = section.appointments.some(a => a.status === 'Pending')
                const allConfirmed = section.appointments.every(a => a.status === 'Confirmed')
                const isSelected = selectedDayKey === section.key
                
                let colorClass = 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                if (isSelected) {
                  colorClass = 'bg-primary-600 text-white shadow-lg shadow-primary-500/25'
                } else if (hasPending) {
                  colorClass = 'bg-amber-50 text-amber-700 border border-amber-200 hover:bg-amber-100'
                } else if (allConfirmed && section.appointments.length > 0) {
                  colorClass = 'bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100'
                }

                return (
                  <button
                    key={section.key}
                    onClick={() => setSelectedDayKey(section.key)}
                    className={`px-4 py-2 text-xs rounded-xl font-semibold transition-all duration-200 flex items-center gap-2 ${colorClass}`}
                  >
                    {hasPending && !isSelected && <div className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />}
                    {section.label}
                    <span className={`text-[10px] px-1.5 py-0.5 rounded-md ${isSelected ? 'bg-white/20' : 'bg-gray-200/50'}`}>
                      {section.appointments.length}
                    </span>
                  </button>
                )
              })}
            </div>

            <div className="rounded-2xl border border-gray-100 overflow-hidden bg-white">
              <div className="px-4 py-3 bg-gray-50 border-b border-gray-100">
                <h3 className="text-sm font-semibold text-gray-700">
                  {daySections.find((section) => section.key === selectedDayKey)?.label ?? 'Appointments'}
                </h3>
              </div>
              <AppointmentTable
                appointments={selectedDayAppointments}
                onConfirm={handleConfirm}
                onUnconfirm={handleUnconfirm}
                onCancel={handleCancel}
                onComplete={handleComplete}
                onDelete={handleDelete}
              />
            </div>
          </div>
        )}
      </Card>

      {showClearHistoryModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900/40 backdrop-blur-sm">
          <div className="bg-white rounded-2xl max-w-sm w-full p-6 shadow-xl border border-gray-100">
            <h3 className="text-lg font-bold text-gray-900 mb-2">Clear History</h3>
            <p className="text-sm text-gray-600 mb-6">Are you sure? This will permanently remove all past appointments from your records. This action cannot be undone.</p>
            <div className="flex justify-end gap-3">
              <Button variant="outline" onClick={() => setShowClearHistoryModal(false)}>Cancel</Button>
              <Button variant="destructive" onClick={confirmClearHistory}>Yes, clear history</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
