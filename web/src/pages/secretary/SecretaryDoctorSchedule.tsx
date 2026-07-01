import { useEffect, useState } from 'react'
import { AvailabilityEditor } from '@/components/doctor/AvailabilityEditor'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { secretaryApi } from '@/api/secretaryApi'
import type { AvailabilityDto, DoctorDetailDto } from '@/lib/types'
import toast from 'react-hot-toast'
import { Clock, ArrowLeft, Eye, EyeOff } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useLanguage } from '@/lib/language'

export default function SecretaryDoctorSchedule() {
  const { t, isRTL } = useLanguage()
  const navigate = useNavigate()
  const [availability, setAvailability] = useState<AvailabilityDto[]>([])
  const [myDoctor, setMyDoctor] = useState<DoctorDetailDto | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [isVisible, setIsVisible] = useState(true)

  useEffect(() => {
    setIsLoading(true)
    Promise.all([
      secretaryApi.getMyDoctorAvailability().then(setAvailability),
      secretaryApi.getMyDoctor().then((doc) => {
        setMyDoctor(doc)
        setIsVisible(doc.isScheduleVisible ?? true)
      }),
    ])
      .catch(() => toast.error(t('errLoadSchedule')))
      .finally(() => setIsLoading(false))
  }, [])

  const handleSave = async (data: AvailabilityDto[]) => {
    try {
      setIsSaving(true)
      await secretaryApi.updateMyDoctorAvailability(data)
      toast.success(t('scheduleSavedSuccess'))
    } catch {
      toast.error(t('errSaveScheduleGeneric'))
    } finally {
      setIsSaving(false)
    }
  }

  const toggleVisibility = async () => {
    const newValue = !isVisible
    setIsVisible(newValue)
    try {
      await secretaryApi.updateMyDoctorScheduleVisibility(newValue)
      toast.success(newValue ? t('scheduleVisibleToPatients') : t('scheduleHiddenFromPatients'))
    } catch {
      toast.error(t('errUpdateVisibility'))
      setIsVisible(!newValue)
    }
  }

  return (
    <div className="space-y-5 p-6" dir={isRTL ? 'rtl' : 'ltr'}>
      <div className={`flex items-center gap-3 ${isRTL ? 'flex-row-reverse' : ''}`}>
        <Button variant="ghost" size="sm" icon={<ArrowLeft size={18} />} onClick={() => navigate('/secretary/dashboard')}>
          {t('back')}
        </Button>
        <div className="p-2 bg-primary-50 rounded-xl">
          <Clock size={20} className="text-primary-600" />
        </div>
        <div className={isRTL ? 'text-right' : ''}>
          <h1 className="text-xl font-bold text-gray-800">{t('doctorSchedule')}</h1>
          <p className="text-sm text-gray-500">{t('manageDoctorReception')}</p>
        </div>
      </div>

      <Card className="mb-4">
        <CardHeader className={`flex flex-row items-center justify-between ${isRTL ? 'flex-row-reverse' : ''}`}>
          <CardTitle>{t('weeklyAvailability')}</CardTitle>
          <Button
            variant={isVisible ? 'success' : 'outline'}
            size="sm"
            onClick={toggleVisibility}
            icon={isVisible ? <Eye size={16} /> : <EyeOff size={16} />}
          >
            {isVisible ? t('visibleToPatients') : t('hiddenFromPatients')}
          </Button>
        </CardHeader>
        {isLoading ? <PageLoader /> : (
          <AvailabilityEditor availability={availability} onSave={handleSave} isSaving={isSaving} />
        )}
      </Card>
    </div>
  )
}
