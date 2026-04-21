import { useDoctorStore } from '@/store/doctorStore'
import { useEffect, useState } from 'react'
import { AvailabilityEditor } from '@/components/doctor/AvailabilityEditor'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import toast from 'react-hot-toast'
import { Clock, Eye, EyeOff } from 'lucide-react'

export default function DoctorSchedule() {
  const { availability, profile, fetchAvailability, updateAvailability, updateScheduleVisibility, fetchProfile, isLoadingProfile } = useDoctorStore()
  const [isVisible, setIsVisible] = useState(profile?.isScheduleVisible ?? true)

  useEffect(() => { 
    fetchAvailability()
    fetchProfile()
  }, [fetchAvailability, fetchProfile])

  useEffect(() => {
    setIsVisible(profile?.isScheduleVisible ?? true)
  }, [profile?.isScheduleVisible])

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

      <Card className="mb-4">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Weekly Availability</CardTitle>
          <Button
            variant={isVisible ? 'success' : 'outline'}
            size="sm"
            onClick={async () => {
              const newValue = !isVisible
              setIsVisible(newValue)
              await updateScheduleVisibility(newValue)
              toast.success(newValue ? 'Schedule is now visible to patients' : 'Schedule is now hidden from patients')
            }}
            icon={isVisible ? <Eye size={16} /> : <EyeOff size={16} />}
          >
            {isVisible ? 'Visible to Patients' : 'Hidden from Patients'}
          </Button>
        </CardHeader>
        {isLoadingProfile ? <PageLoader /> : (
          <AvailabilityEditor availability={availability} onSave={handleSave} />
        )}
      </Card>
    </div>
  )
}
