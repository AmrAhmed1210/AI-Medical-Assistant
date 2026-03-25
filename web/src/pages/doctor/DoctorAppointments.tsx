import { useState } from 'react'
import { useDoctorAppointments } from '@/hooks/useDoctor'
import { useAppointmentStore } from '@/store/appointmentStore'
import { AppointmentTable } from '@/components/doctor/AppointmentTable'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import type { AppointmentStatus } from '@/lib/types'
import toast from 'react-hot-toast'
import { RefreshCw } from 'lucide-react'

const STATUS_FILTERS: { label: string; value: AppointmentStatus | '' }[] = [
  { label: 'الكل', value: '' },
  { label: 'قيد الانتظار', value: 'Pending' },
  { label: 'مؤكد', value: 'Confirmed' },
  { label: 'مكتمل', value: 'Completed' },
  { label: 'ملغي', value: 'Cancelled' },
]

export default function DoctorAppointments() {
  const [statusFilter, setStatusFilter] = useState<AppointmentStatus | ''>('')
  const { appointments, isLoading, refetch } = useDoctorAppointments(statusFilter || undefined)
  const { confirm, cancel, complete, updateLocal } = useAppointmentStore()

  const handleConfirm = async (id: string) => {
    try {
      await confirm(id)
      updateLocal(id, { status: 'Confirmed' })
      toast.success('تم تأكيد الموعد')
    } catch { toast.error('فشل تأكيد الموعد') }
  }

  const handleCancel = async (id: string) => {
    try {
      await cancel(id)
      updateLocal(id, { status: 'Cancelled' })
      toast.success('تم إلغاء الموعد')
    } catch { toast.error('فشل إلغاء الموعد') }
  }

  const handleComplete = async (id: string) => {
    try {
      await complete(id)
      updateLocal(id, { status: 'Completed' })
      toast.success('تم تسجيل الموعد كمكتمل')
    } catch { toast.error('فشلت العملية') }
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-800">المواعيد</h1>
          <p className="text-sm text-gray-500 mt-0.5">إدارة مواعيد المرضى</p>
        </div>
        <Button variant="outline" icon={<RefreshCw size={14} />} onClick={refetch} size="sm">
          تحديث
        </Button>
      </div>

      <Card padding="none">
        <div className="flex items-center gap-2 p-4 border-b border-gray-100 flex-wrap">
          {STATUS_FILTERS.map((f) => (
            <button
              key={f.value}
              onClick={() => setStatusFilter(f.value)}
              className={`px-3 py-1.5 text-xs rounded-xl font-medium transition-colors ${
                statusFilter === f.value
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
