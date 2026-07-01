import { useDoctorStore } from '@/store/doctorStore'
import { useLanguage } from '@/lib/language'
import { useEffect, useState } from 'react'
import { AvailabilityEditor } from '@/components/doctor/AvailabilityEditor'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import toast from 'react-hot-toast'
import { Clock, Eye, EyeOff } from 'lucide-react'

export default function DoctorSchedule() {
  const { t, isRTL } = useLanguage()
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
      toast.success(t('scheduleSaved'))
    } catch {
      toast.error(t('errSaveSchedule'))
    }
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-primary-50 rounded-xl">
          <Clock size={20} className="text-primary-600" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-gray-800">{t('schedule')}</h1>
          <p className="text-sm text-gray-500">{t('manageWeeklyTimes')}</p>
        </div>
      </div>

      <Card className="mb-4">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>{t('weeklyAvailability')}</CardTitle>
          <Button
            variant={isVisible ? 'success' : 'outline'}
            size="sm"
            onClick={async () => {
              const newValue = !isVisible
              setIsVisible(newValue)
              await updateScheduleVisibility(newValue)
              toast.success(newValue ? t('scheduleVisible') : t('scheduleHidden'))
            }}
            icon={isVisible ? <Eye size={16} /> : <EyeOff size={16} />}
          >
            {isVisible ? t('visibleToPatients') : t('hiddenFromPatients')}
          </Button>
        </CardHeader>
        {isLoadingProfile ? <PageLoader /> : (
          <AvailabilityEditor availability={availability} onSave={handleSave} />
        )}
      </Card>
    </div>
  )
}
