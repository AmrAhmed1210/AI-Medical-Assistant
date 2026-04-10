import { useDoctorStore } from '@/store/doctorStore'
import { useEffect } from 'react'
import { AvailabilityEditor } from '@/components/doctor/AvailabilityEditor'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import toast from 'react-hot-toast'
import { Clock } from 'lucide-react'

export default function DoctorSchedule() {
  const { availability, fetchAvailability, updateAvailability, isLoadingProfile } = useDoctorStore()

  useEffect(() => { fetchAvailability() }, [fetchAvailability])

  const handleSave = async (data: typeof availability) => {
    try {
      await updateAvailability(data)
      toast.success('Schedule saved successfully')
    } catch {
      toast.error('Failed to save schedule')
    }
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-primary-50 rounded-xl">
          <Clock size={20} className="text-primary-600" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-gray-800">Schedule</h1>
          <p className="text-sm text-gray-500">Manage your weekly reception times</p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Weekly Availability</CardTitle>
        </CardHeader>
        {isLoadingProfile ? <PageLoader /> : (
          <AvailabilityEditor availability={availability} onSave={handleSave} />
        )}
      </Card>
    </div>
  )
}
