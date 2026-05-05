import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Calendar,
  Clock,
  User,
  Stethoscope,
  ChevronLeft,
  AlertCircle,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { visitApi } from '@/api/visitApi'
import { useTodayVisits } from '@/hooks/useVisits'
import { Card, Button, SkeletonCard } from '@/components/ui'

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
}

const item = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0 },
}

import type { AppointmentDto } from '@/lib/types'

export default function DoctorToday() {
  const navigate = useNavigate()
  const [startingVisitId, setStartingVisitId] = useState<string | null>(null)

  const { appointments, isLoading } = useTodayVisits()

  const [isCreating, setIsCreating] = useState(false)

  const handleStartVisit = async (appt: AppointmentDto) => {
    setStartingVisitId(appt.id)
    setIsCreating(true)
    try {
      const visit = await visitApi.createVisit({
        patientId: appt.patientId,
        doctorId: 0,
        appointmentId: Number(appt.id),
        chiefComplaint: '',
      })
      toast.success('تم فتح الزيارة')
      navigate(`/doctor/workspace/${visit.id}`)
    } catch {
      toast.error('فشل في فتح الزيارة')
      setStartingVisitId(null)
    } finally {
      setIsCreating(false)
    }
  }

  const today = new Date().toLocaleDateString('ar-EG', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="flex flex-col md:flex-row md:items-center justify-between gap-4"
      >
        <motion.div variants={item}>
          <h1 className="text-2xl font-bold text-gray-900">مواعيد اليوم</h1>
          <p className="text-gray-500 text-sm mt-1 flex items-center gap-2">
            <Calendar className="w-4 h-4" />
            {today}
          </p>
        </motion.div>
        <motion.div variants={item} className="flex items-center gap-3">
          <div className="bg-primary-50 text-primary-700 px-4 py-2 rounded-lg text-sm font-medium">
            {appointments?.length ?? 0} موعد
          </div>
        </motion.div>
      </motion.div>

      {/* Appointments List */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="space-y-4"
      >
        {isLoading ? (
          <SkeletonCard count={4} />
        ) : !appointments || appointments.length === 0 ? (
          <motion.div variants={item}>
            <Card className="text-center py-12">
              <Calendar className="w-12 h-12 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500 font-medium">لا توجد مواعيد لهذا اليوم</p>
              <p className="text-gray-400 text-sm mt-1">
                ستظهر المواعيد الجديدة هنا تلقائياً
              </p>
            </Card>
          </motion.div>
        ) : (
          appointments.map((appt: AppointmentDto) => (
            <motion.div key={appt.id} variants={item}>
              <Card className="hover:shadow-md transition-shadow">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  {/* Patient Info */}
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center">
                      <User className="w-6 h-6 text-primary-600" />
                    </div>
                    <div>
                      <h3 className="font-bold text-gray-900">{appt.patientName}</h3>
                      <div className="flex items-center gap-3 text-sm text-gray-500 mt-1">
                        <span className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          {appt.time || appt.scheduledAt}
                        </span>
                        <span
                          className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                            appt.status === 'Confirmed'
                              ? 'bg-blue-100 text-blue-700'
                              : appt.status === 'Completed'
                              ? 'bg-green-100 text-green-700'
                              : 'bg-gray-100 text-gray-500'
                          }`}
                        >
                          {appt.status === 'Confirmed'
                            ? 'مؤكد'
                            : appt.status === 'Completed'
                            ? 'منتهي'
                            : appt.status === 'Cancelled'
                            ? 'ملغى'
                            : appt.status === 'NoShow'
                            ? 'لم يحضر'
                            : 'معلق'}
                        </span>
                      </div>
                      {appt.notes && (
                        <p className="text-xs text-gray-400 mt-1 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          {appt.notes}
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-3">
                    {appt.status === 'Confirmed' && (
                      <Button
                        onClick={() => handleStartVisit(appt)}
                        disabled={startingVisitId === appt.id}
                        className="bg-primary-600 hover:bg-primary-700"
                      >
                        <Stethoscope className="w-4 h-4 ml-2" />
                        {startingVisitId === appt.id
                          ? 'جاري الفتح...'
                          : 'بدء الزيارة'}
                      </Button>
                    )}
                    {appt.status === 'Completed' && (
                      <Button
                        variant="outline"
                        onClick={() => navigate(`/doctor/visits/${appt.id}/summary`)}
                      >
                        عرض الملخص
                        <ChevronLeft className="w-4 h-4 mr-1" />
                      </Button>
                    )}
                  </div>
                </div>
              </Card>
            </motion.div>
          ))
        )}
      </motion.div>
    </div>
  )
}
