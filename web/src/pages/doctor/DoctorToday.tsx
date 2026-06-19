import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Calendar,
  Clock,
  User,
  Stethoscope,
  ChevronLeft,
  Search,
  History,
  Sparkles,
  Activity,
  CheckCircle2,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { visitApi } from '@/api/visitApi'
import { doctorApi } from '@/api/doctorApi'
import { Card, Button, SkeletonCard } from '@/components/ui'
import type { AppointmentDto } from '@/lib/types'
import PreVisitSummaryCard from '@/components/doctor/PreVisitSummaryCard'

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } },
}

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } },
}

export default function DoctorToday() {
  const navigate = useNavigate()
  const [startingVisitId, setStartingVisitId] = useState<string | null>(null)
  const [appointments, setAppointments] = useState<AppointmentDto[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [isCreating, setIsCreating] = useState(false)

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

  // Get only today's appointments using local timezone
  const d = new Date()
  const todayKey = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`

  const toDayKey = (dateStr: string) => {
    const date = new Date(dateStr)
    if (isNaN(date.getTime())) return 'unknown'
    const y = date.getFullYear()
    const m = `${date.getMonth() + 1}`.padStart(2, '0')
    const day = `${date.getDate()}`.padStart(2, '0')
    return `${y}-${m}-${day}`
  }
  
  const todayAppointments = appointments.filter(a => {
    // Check if appointment is today AND matches search query
    const isToday = toDayKey(a.scheduledAt) === todayKey
    const matchesSearch = a.patientName.toLowerCase().includes(searchQuery.toLowerCase())
    return isToday && matchesSearch
  }).sort((a, b) => new Date(a.scheduledAt).getTime() - new Date(b.scheduledAt).getTime())

  const formattedTodayDate = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })

  return (
    <div className="max-w-6xl mx-auto p-4 md:p-8 space-y-8 relative">
      {/* Decorative ambient background elements */}
      <div className="absolute top-0 left-10 w-72 h-72 bg-primary-400/10 rounded-full blur-3xl -z-10 animate-pulse pointer-events-none" />
      <div className="absolute bottom-40 right-10 w-96 h-96 bg-purple-400/10 rounded-full blur-3xl -z-10 animate-pulse delay-1000 pointer-events-none" />

      {/* Header Section */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl rounded-3xl p-6 md:p-8 border border-white/50 dark:border-slate-800/80 shadow-[0_8px_30px_rgb(0,0,0,0.04)] relative overflow-hidden"
      >
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 relative z-10">
          <motion.div variants={item} className="space-y-2">
            <h1 className="text-3xl md:text-4xl font-extrabold text-slate-900 dark:text-white tracking-tight">
              Today's <span className="text-primary-600 dark:text-primary-400">Visits</span>
            </h1>
            <p className="text-slate-500 dark:text-slate-400 font-medium flex items-center gap-2">
              <Calendar className="w-4 h-4" />
              {formattedTodayDate}
            </p>
          </motion.div>

          <motion.div variants={item} className="flex flex-col sm:flex-row items-center gap-4 w-full md:w-auto">
            <div className="relative w-full sm:w-72 group">
              <Search className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 w-4 h-4 group-focus-within:text-primary-500 transition-colors" />
              <input 
                type="text"
                placeholder="Search patient..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pr-12 pl-5 py-3 bg-slate-50 dark:bg-slate-900/50 border border-slate-200/60 dark:border-slate-700/60 rounded-2xl text-sm focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none transition-all placeholder:text-slate-400 dark:text-white"
              />
            </div>
            <div className="bg-primary-50 text-primary-700 px-5 py-3 rounded-2xl text-sm font-bold flex items-center gap-2 whitespace-nowrap">
              <Activity className="w-4 h-4" />
              {todayAppointments.length} Today
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* Appointments List */}
      <div className="space-y-6">
        {isLoading ? (
          <div className="space-y-4">
            <SkeletonCard count={3} />
          </div>
        ) : todayAppointments.length === 0 ? (
          <motion.div variants={item} initial="hidden" animate="show">
            <div className="flex flex-col items-center justify-center py-20 text-center bg-white/40 dark:bg-slate-900/40 rounded-3xl border-dashed border-2 border-slate-200 dark:border-slate-800">
              <div className="w-20 h-20 bg-white dark:bg-slate-800 rounded-2xl flex items-center justify-center mb-6 shadow-sm border border-slate-100 dark:border-slate-700">
                <CheckCircle2 className="w-10 h-10 text-emerald-400 dark:text-emerald-500" />
              </div>
              <h3 className="text-xl font-bold text-slate-700 dark:text-slate-300 mb-2">No Appointments Today</h3>
              <p className="text-slate-500 dark:text-slate-500 max-w-sm">
                You have a clear schedule for today.
              </p>
            </div>
          </motion.div>
        ) : (
          <AnimatePresence>
            <motion.div 
              variants={container}
              initial="hidden"
              animate="show"
              className="grid gap-6"
            >
              {todayAppointments.map((appt: AppointmentDto) => {
                const isConfirmed = appt.status === 'Confirmed';
                const isCompleted = appt.status === 'Completed';
                const isCancelled = appt.status === 'Cancelled';
                
                return (
                  <motion.div key={appt.id} variants={item} layout>
                    <div className="bg-white dark:bg-slate-900 rounded-3xl p-6 transition-shadow duration-300 hover:shadow-xl border border-slate-100 dark:border-slate-800">
                      
                      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
                        {/* Patient Info */}
                        <div className="flex items-center gap-5">
                          <div className={`w-14 h-14 rounded-full flex items-center justify-center shrink-0 ${
                            isConfirmed ? 'bg-primary-50 text-primary-600 dark:bg-primary-900/30 dark:text-primary-400' : 
                            'bg-slate-50 dark:bg-slate-800 text-slate-400'
                          }`}>
                            <User className="w-6 h-6" />
                          </div>
                          
                          <div>
                            <h3 className="font-bold text-slate-900 dark:text-white text-lg">
                              {appt.patientName}
                            </h3>
                            <div className="flex items-center gap-3 mt-1.5">
                              <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400 text-sm font-medium">
                                <Clock className="w-4 h-4" />
                                {new Date(appt.scheduledAt).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                              </span>
                              
                              <span className={`flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-bold uppercase ${
                                isConfirmed ? 'text-primary-600 bg-primary-50 dark:text-primary-400 dark:bg-primary-900/20' : 
                                isCompleted ? 'text-emerald-600 bg-emerald-50 dark:text-emerald-400 dark:bg-emerald-900/20' :
                                isCancelled ? 'text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-900/20' : 
                                'text-slate-500 bg-slate-100 dark:text-slate-400 dark:bg-slate-800'
                              }`}>
                                {appt.status}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Actions */}
                        <div className="flex items-center gap-3 w-full lg:w-auto">
                          {isConfirmed ? (
                            <button
                              onClick={() => handleStartVisit(appt)}
                              disabled={startingVisitId === appt.id}
                              className="w-full lg:w-auto flex items-center justify-center gap-2 bg-primary-600 hover:bg-primary-700 text-white font-bold py-3 px-8 rounded-2xl shadow-sm transition-all disabled:opacity-70"
                            >
                              <Stethoscope className="w-5 h-5" />
                              {startingVisitId === appt.id ? 'Starting...' : 'Start Visit'}
                            </button>
                          ) : isCompleted ? (
                            <button
                              onClick={() => navigate(`/doctor/visits/${appt.id}/summary`)}
                              className="w-full lg:w-auto flex items-center justify-center gap-2 bg-slate-50 hover:bg-slate-100 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-bold py-3 px-6 rounded-2xl transition-all"
                            >
                              <History className="w-5 h-5" />
                              View Record
                            </button>
                          ) : (
                            <div className="w-full lg:w-auto text-center text-sm text-slate-500 font-medium px-6 py-3 bg-slate-50 dark:bg-slate-800 rounded-2xl">
                              {isCancelled ? 'Cancelled' : 'Awaiting confirmation'}
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {/* AI Pre-Visit Summary Card - Embedded seamlessly */}
                      <div className="mt-6 pt-6 border-t border-slate-100 dark:border-slate-800">
                        <PreVisitSummaryCard appointment={appt} />
                      </div>
                    </div>
                  </motion.div>
                )
              })}
            </motion.div>
          </AnimatePresence>
        )}
      </div>
    </div>
  )
}
