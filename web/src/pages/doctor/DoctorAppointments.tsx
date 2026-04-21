import { useEffect, useMemo, useState } from 'react'
import { useAppointmentStore } from '@/store/appointmentStore'
import { appointmentApi } from '@/api/appointmentApi'
import { doctorApi } from '@/api/doctorApi'
import { useAuthStore } from '@/store/authStore'
import { startConnection } from '@/lib/signalr'
import { useNotificationStore } from '@/store/notificationStore'
import { AppointmentTable } from '@/components/doctor/AppointmentTable'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import type { AppointmentStatus, AppointmentDto } from '@/lib/types'
import toast from 'react-hot-toast'
import { RefreshCw, Calendar } from 'lucide-react'

const STATUS_FILTERS: { label: string; value: AppointmentStatus | '' }[] = [
  { label: 'All', value: '' },
  { label: 'Pending', value: 'Pending' },
  { label: 'Confirmed', value: 'Confirmed' },
  { label: 'Completed', value: 'Completed' },
  { label: 'Cancelled', value: 'Cancelled' },
]

export default function DoctorAppointments() {
  const [appointments, setAppointments] = useState<AppointmentDto[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [statusFilter, setStatusFilter] = useState<AppointmentStatus | ''>('')
  const [selectedDayKey, setSelectedDayKey] = useState('')
  const { updateLocal } = useAppointmentStore()
  const { addNotification } = useNotificationStore()
  const { token } = useAuthStore()

  const fetchAppointments = async () => {
    setIsLoading(true)
    try {
      const data = await doctorApi.getAppointments(statusFilter || undefined)
      const normalizeStatus = (status: string): AppointmentStatus => {
        const lowered = (status || '').toLowerCase()
        if (lowered === 'confirmed') return 'Confirmed'
        if (lowered === 'cancelled' || lowered === 'canceled') return 'Cancelled'
        if (lowered === 'completed') return 'Completed'
        return 'Pending'
      }
      const normalized = (data as AppointmentDto[]).map((item: any) => ({
        ...item,
        id: String(item.id),
        patientName: item.patientName ?? 'Unknown',
        doctorName: item.doctorName ?? 'Doctor',
        paymentMethod: item.paymentMethod ?? item.PaymentMethod ?? '',
        status: normalizeStatus(item.status),
        scheduledAt: item.scheduledAt ?? `${item.date ?? ''} ${item.time ?? ''}`.trim(),
      }))
        .sort((a, b) => new Date(a.scheduledAt).getTime() - new Date(b.scheduledAt).getTime())

      setAppointments(normalized)
    } catch (error: any) {
      console.error('Error fetching appointments:', error)
      toast.error('Failed to load appointments')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchAppointments()
  }, [statusFilter])

  useEffect(() => {
    let active = true
    let cleanup: (() => void) | undefined

    const bindRealtime = async () => {
      if (!token) return
      try {
        const conn = await startConnection(token)
        if (!active) return

        const onNotificationReceived = (payload: any) => {
          const category = payload?.category ?? payload?.Category
          const title = payload?.title ?? payload?.Title ?? 'Appointment'
          const message = payload?.message ?? payload?.Message ?? 'Appointments updated'
          if (category === 'new_booking' || category === 'appointment_update' || category === 'schedule_updated') {
            addNotification('info', title, message)
            toast.success(message)
            fetchAppointments().catch(() => undefined)
          }
        }

        const onAppointmentUpdated = (payload: any) => {
          const message = payload?.message ?? 'Appointment updated'
          addNotification('success', 'Appointment Update', message)
          toast.success(message)
          fetchAppointments().catch(() => undefined)
        }

        conn.on('NotificationReceived', onNotificationReceived)
        conn.on('AppointmentUpdated', onAppointmentUpdated)

        cleanup = () => {
          conn.off('NotificationReceived', onNotificationReceived)
          conn.off('AppointmentUpdated', onAppointmentUpdated)
        }
      } catch {
        // realtime is optional
      }
    }

    bindRealtime()
    return () => {
      active = false
      cleanup?.()
    }
  }, [addNotification, token])

  const patchLocalAppointment = (id: string, patch: Partial<AppointmentDto>) => {
    setAppointments((prev) => prev.map((appt) => (appt.id === id ? { ...appt, ...patch } : appt)))
  }

  const handleConfirm = async (id: string) => {
    try {
      await appointmentApi.confirm(id)
      updateLocal(id, { status: 'Confirmed' })
      patchLocalAppointment(id, { status: 'Confirmed' })
      toast.success('Appointment confirmed')
    } catch { toast.error('Failed to confirm appointment') }
  }

  const handleUnconfirm = async (id: string) => {
    try {
      await appointmentApi.setPending(id)
      updateLocal(id, { status: 'Pending' })
      patchLocalAppointment(id, { status: 'Pending' })
      toast.success('Appointment moved to pending')
    } catch { toast.error('Failed to unconfirm appointment') }
  }

  const handleCancel = async (id: string) => {
    try {
      await appointmentApi.cancel(id)
      updateLocal(id, { status: 'Cancelled' })
      patchLocalAppointment(id, { status: 'Cancelled' })
      toast.success('Appointment cancelled')
    } catch { toast.error('Failed to cancel appointment') }
  }

  const handleComplete = async (id: string) => {
    try {
      await appointmentApi.complete(id)
      updateLocal(id, { status: 'Completed' })
      patchLocalAppointment(id, { status: 'Completed' })
      toast.success('Appointment marked as completed')
    } catch { toast.error('Failed to complete operation') }
  }

  const handleDelete = async (id: string) => {
    try {
      await appointmentApi.delete(id)
      updateLocal(id, { status: 'Cancelled' })
      patchLocalAppointment(id, { status: 'Cancelled' })
      toast.success('Appointment deleted')
    } catch {
      toast.error('Failed to delete appointment')
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

  const daySections = useMemo(() => {
    const grouped = appointments.reduce<Record<string, AppointmentDto[]>>((acc, appt) => {
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
  }, [appointments])

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
        <div className="flex items-center gap-2 p-4 border-b border-gray-100 flex-wrap">
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
              {daySections.map((section) => (
                <button
                  key={section.key}
                  onClick={() => setSelectedDayKey(section.key)}
                  className={`px-3 py-1.5 text-xs rounded-xl font-medium transition-colors ${
                    selectedDayKey === section.key
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {section.label} ({section.appointments.length})
                </button>
              ))}
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
    </div>
  )
}
