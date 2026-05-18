import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Calendar,
  Clock,
  User,
  Stethoscope,
  ChevronLeft,
  AlertCircle,
  Search,
  History,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { visitApi } from '@/api/visitApi'
import { doctorApi } from '@/api/doctorApi'
import { Card, Button, SkeletonCard } from '@/components/ui'
import type { AppointmentDto } from '@/lib/types'

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
}

const item = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0 },
}

export default function DoctorToday() {
  const navigate = useNavigate()
  const [startingVisitId, setStartingVisitId] = useState<string | null>(null)
  const [appointments, setAppointments] = useState<AppointmentDto[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')

  const fetchAppointments = async () => {
    setIsLoading(true)
    try {
      const data = await doctorApi.getAppointments()
      setAppointments(data)
    } catch {
      toast.error('Failed to load appointments')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchAppointments()
  }, [])

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
      toast.success('Visit started successfully')
      navigate(`/doctor/workspace/${visit.id}`)
    } catch {
      toast.error('Failed to start visit')
      setStartingVisitId(null)
    } finally {
      setIsCreating(false)
    }
  }

  const [isCreating, setIsCreating] = useState(false)

  const toDayKey = (dateStr: string) => {
    const d = new Date(dateStr)
    if (isNaN(d.getTime())) return 'Unknown'
    return d.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    })
  }

  const filteredAppointments = appointments.filter(a => 
    a.patientName.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const groupedAppointments = filteredAppointments.reduce((acc, appt) => {
    const day = toDayKey(appt.scheduledAt)
    if (!acc[day]) acc[day] = []
    acc[day].push(appt)
    return acc
  }, {} as Record<string, AppointmentDto[]>)

  const sortedDays = Object.keys(groupedAppointments).sort((a, b) => {
    // Basic sort (reverse chron for attendance history feel)
    return new Date(groupedAppointments[b][0].scheduledAt).getTime() - new Date(groupedAppointments[a][0].scheduledAt).getTime()
  })

  const todayStr = toDayKey(new Date().toISOString())

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
          <h1 className="text-2xl font-bold text-gray-900">Attendance & Visit Log</h1>
          <p className="text-gray-500 text-sm mt-1 flex items-center gap-2">
            <Calendar className="w-4 h-4" />
            {todayStr}
          </p>
        </motion.div>
        <motion.div variants={item} className="flex items-center gap-3 w-full md:w-auto">
          <div className="relative flex-1 md:w-64">
            <Search className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input 
              type="text"
              placeholder="Search patient..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pr-10 pl-4 py-2 bg-white border border-gray-200 rounded-xl text-sm focus:ring-2 focus:ring-primary-500/20 outline-none transition-all"
            />
          </div>
          <div className="bg-primary-50 text-primary-700 px-4 py-2 rounded-xl text-sm font-bold shadow-sm">
            {appointments?.length ?? 0} appointment{(appointments?.length ?? 0) !== 1 ? 's' : ''}
          </div>
        </motion.div>
      </motion.div>

      {/* Appointments List Grouped by Day */}
      <div className="space-y-8">
        {isLoading ? (
          <SkeletonCard count={4} />
        ) : sortedDays.length === 0 ? (
          <motion.div variants={item}>
            <Card className="text-center py-16 border-dashed border-2">
              <Calendar className="w-16 h-16 text-gray-200 mx-auto mb-4" />
              <p className="text-gray-500 font-bold text-lg">No attendance records found</p>
              <p className="text-gray-400 text-sm mt-1">
                New appointments and visits will appear here
              </p>
            </Card>
          </motion.div>
        ) : (
          sortedDays.map((day) => (
            <div key={day} className="space-y-4">
              <div className="flex items-center gap-4">
                <h2 className={`text-sm font-bold px-3 py-1 rounded-full ${day === todayStr ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-600'}`}>
                  {day === todayStr ? 'Today' : day}
                </h2>
                <div className="h-px bg-gray-200 flex-1" />
              </div>

              <div className="grid gap-4">
                {groupedAppointments[day].map((appt: AppointmentDto) => (
                  <motion.div key={appt.id} variants={item}>
                    <Card className={`hover:shadow-lg transition-all border-l-4 ${
                      appt.status === 'Confirmed' ? 'border-l-blue-500' : 
                      appt.status === 'Completed' ? 'border-l-emerald-500' :
                      appt.status === 'Cancelled' ? 'border-l-red-400' : 'border-l-gray-300'
                    }`}>
                      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                        {/* Patient Info */}
                        <div className="flex items-center gap-4">
                          <div className={`w-14 h-14 rounded-2xl flex items-center justify-center shadow-inner ${
                            appt.status === 'Confirmed' ? 'bg-blue-50 text-blue-600' : 'bg-gray-50 text-gray-400'
                          }`}>
                            <User className="w-7 h-7" />
                          </div>
                          <div>
                            <h3 className="font-bold text-gray-900 text-lg">{appt.patientName}</h3>
                            <div className="flex items-center gap-3 text-sm text-gray-500 mt-1">
                              <span className="flex items-center gap-1 bg-gray-100 px-2 py-0.5 rounded-lg font-medium">
                                <Clock className="w-3.5 h-3.5" />
                                {new Date(appt.scheduledAt).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                              </span>
                              <span
                                className={`px-2.5 py-0.5 rounded-lg text-xs font-bold uppercase tracking-wider ${
                                  appt.status === 'Confirmed'
                                    ? 'bg-blue-100 text-blue-700'
                                    : appt.status === 'Completed'
                                    ? 'bg-emerald-100 text-emerald-700'
                                    : appt.status === 'Cancelled'
                                    ? 'bg-red-100 text-red-700'
                                    : 'bg-gray-100 text-gray-500'
                                }`}
                              >
                                {appt.status === 'Confirmed'
                                  ? 'Confirmed'
                                  : appt.status === 'Completed'
                                  ? 'Completed'
                                  : appt.status === 'Cancelled'
                                  ? 'Cancelled'
                                  : 'Pending'}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Actions */}
                        <div className="flex items-center gap-3">
                          {appt.status === 'Confirmed' ? (
                            <Button
                              onClick={() => handleStartVisit(appt)}
                              disabled={startingVisitId === appt.id}
                              className="bg-primary-600 hover:bg-primary-700 shadow-lg shadow-primary-500/30 px-6"
                            >
                              <Stethoscope className="w-4 h-4 ml-2" />
                              {startingVisitId === appt.id
                                ? 'Starting...'
                                : 'Start Visit'}
                            </Button>
                          ) : appt.status === 'Completed' ? (
                            <Button
                              variant="outline"
                              onClick={() => navigate(`/doctor/visits/${appt.id}/summary`)}
                              className="border-emerald-200 text-emerald-700 hover:bg-emerald-50"
                            >
                              <History className="w-4 h-4 ml-2" />
                              View Record
                              <ChevronLeft className="w-4 h-4 mr-1" />
                            </Button>
                          ) : (
                            <div className="text-xs text-gray-400 font-medium px-4 py-2 bg-gray-50 rounded-lg border">
                              {appt.status === 'Cancelled' ? 'Cannot start a cancelled appointment' : 'Awaiting confirmation'}
                            </div>
                          )}
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
