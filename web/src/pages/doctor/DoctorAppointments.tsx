import { useState } from 'react'
import { useDoctorAppointments } from '@/hooks/useDoctor'
import { useAppointmentStore } from '@/store/appointmentStore'
import { AppointmentTable } from '@/components/doctor/AppointmentTable'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import type { AppointmentStatus } from '@/lib/types'
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
  const [statusFilter, setStatusFilter] = useState<AppointmentStatus | ''>('')
  const { appointments, isLoading, refetch } = useDoctorAppointments(statusFilter || undefined)
  const { confirm, cancel, complete, updateLocal } = useAppointmentStore()

  const handleConfirm = async (id: string) => {
    try {
      await confirm(id)
      updateLocal(id, { status: 'Confirmed' })
      toast.success('Appointment confirmed')
    } catch { toast.error('Failed to confirm appointment') }
  }

  const handleCancel = async (id: string) => {
    try {
      await cancel(id)
      updateLocal(id, { status: 'Cancelled' })
      toast.success('Appointment cancelled')
    } catch { toast.error('Failed to cancel appointment') }
  }

  const handleComplete = async (id: string) => {
    try {
      await complete(id)
      updateLocal(id, { status: 'Completed' })
      toast.success('Appointment marked as completed')
    } catch { toast.error('Failed to complete operation') }
  }

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
        <Button variant="outline" icon={<RefreshCw size={14} />} onClick={refetch} size="sm">
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
        <AppointmentTable
          appointments={appointments}
          onConfirm={handleConfirm}
          onCancel={handleCancel}
          onComplete={handleComplete}
        />
      </Card>
    </div>
  )
}
