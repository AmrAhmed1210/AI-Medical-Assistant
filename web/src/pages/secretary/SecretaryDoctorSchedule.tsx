import { useEffect, useState } from 'react'
import { AvailabilityEditor } from '@/components/doctor/AvailabilityEditor'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { secretaryApi } from '@/api/secretaryApi'
import type { AvailabilityDto } from '@/lib/types'
import toast from 'react-hot-toast'
import { Clock, ArrowLeft } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

export default function SecretaryDoctorSchedule() {
  const navigate = useNavigate()
  const [availability, setAvailability] = useState<AvailabilityDto[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    setIsLoading(true)
    secretaryApi.getMyDoctorAvailability()
      .then((data) => {
        setAvailability(data)
      })
      .catch(() => toast.error('Failed to load doctor schedule'))
      .finally(() => setIsLoading(false))
  }, [])

  const handleSave = async (data: AvailabilityDto[]) => {
    try {
      setIsSaving(true)
      await secretaryApi.updateMyDoctorAvailability(data)
      toast.success('Doctor schedule saved successfully')
    } catch {
      toast.error('Failed to save schedule')
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <div className="space-y-5 p-6">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="sm" icon={<ArrowLeft size={18} />} onClick={() => navigate('/secretary/dashboard')}>
          Back
        </Button>
        <div className="p-2 bg-primary-50 rounded-xl">
          <Clock size={20} className="text-primary-600" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-gray-800">Doctor Schedule</h1>
          <p className="text-sm text-gray-500">Manage your doctor's weekly reception times</p>
        </div>
      </div>

      <Card className="mb-4">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Weekly Availability</CardTitle>
        </CardHeader>
        {isLoading ? <PageLoader /> : (
          <AvailabilityEditor availability={availability} onSave={handleSave} isSaving={isSaving} />
        )}
      </Card>
    </div>
  )
}
