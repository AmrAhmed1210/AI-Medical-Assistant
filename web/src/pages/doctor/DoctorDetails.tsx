import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  ArrowLeft, Star, MapPin, Calendar, Clock, Phone, Mail, 
  HeartPulse, Award, Languages, CheckCircle, BookOpen
} from 'lucide-react'
import { toast } from 'react-hot-toast'
import { doctorApi } from '@/api/doctorApi'
import { appointmentApi } from '@/api/appointmentApi'
import axiosInstance from '@/api/axiosInstance'
import type { DoctorDetailDto } from '@/lib/types'
import { ReasonForVisitModal } from '@/components/doctor/ReasonForVisitModal'

interface DoctorDetails extends DoctorDetailDto {
  rating?: number
  totalReviews?: number
}

export default function DoctorDetails() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [doctor, setDoctor] = useState<DoctorDetails | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [availableSlots, setAvailableSlots] = useState<string[]>([])
  const [isLoadingSlots, setIsLoadingSlots] = useState(false)
  const [selectedDate, setSelectedDate] = useState('')
  const [selectedTime, setSelectedTime] = useState('')
  const [showReasonModal, setShowReasonModal] = useState(false)

  const fetchDoctorDetails = async () => {
    if (!id) return
    
    setIsLoading(true)
    try {
      const response = await doctorApi.getDoctorById(id)
      
      // Transform API response to include additional UI data
      const transformedDoctor: DoctorDetails = {
        ...response,
        rating: 4.5, // Mock data - should come from API
        totalReviews: 28, // Mock data - should come from API
      }
      
      setDoctor(transformedDoctor)
    } catch (error: any) {
      console.error('Error fetching doctor details:', error)
      toast.error('Failed to load doctor details')
      navigate('/doctors')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchDoctorDetails()
  }, [id])

  useEffect(() => {
    if (!selectedDate || !id) return

    const fetchSlots = async () => {
      setIsLoadingSlots(true)
      try {
        const slots = await appointmentApi.getAvailableSlots(selectedDate, parseInt(id))
        setAvailableSlots(slots.filter(s => s.available).map(s => s.time))
      } catch (error) {
        console.error('Error fetching available slots:', error)
        setAvailableSlots([])
      } finally {
        setIsLoadingSlots(false)
      }
    }

    fetchSlots()
  }, [selectedDate, id])

  const handleBookAppointment = async (reason?: string) => {
    if (!doctor || !selectedDate || !selectedTime) {
      toast.error('Please select a date and time')
      return
    }

    if (!reason) {
      setShowReasonModal(true)
      return
    }

    try {
      const appointmentData = {
        doctorId: doctor.id,
        scheduledAt: `${selectedDate}T${selectedTime}:00`,
        notes: `AI Reason Summary: ${reason}`
      }
      
      await axiosInstance.post('/api/appointments', appointmentData)
      toast.success('Appointment booked successfully!')
      setShowReasonModal(false)
      navigate('/appointments')
    } catch (error: any) {
      console.error('Error booking appointment:', error)
      toast.error(error.response?.data?.message || 'Failed to book appointment')
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500"></div>
      </div>
    )
  }

  if (!doctor) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-950 flex items-center justify-center">
        <div className="text-center">
          <HeartPulse className="mx-auto text-slate-300 dark:text-slate-700 mb-4" size={48} />
          <h3 className="text-xl font-semibold text-slate-600 dark:text-slate-400 mb-2">Doctor not found</h3>
          <button 
            onClick={() => navigate('/doctors')}
            className="text-emerald-600 hover:text-emerald-700 font-medium"
          >
            Back to Doctors List
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-800 dark:text-slate-100" dir="ltr">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-600 to-teal-500 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <button
            onClick={() => navigate('/doctors')}
            className="flex items-center gap-2 text-white/90 hover:text-white mb-6 transition-colors"
          >
            <ArrowLeft size={20} />
            Back to Doctors List
          </button>

          <div className="flex flex-col lg:flex-row items-start gap-8">
            {/* Doctor Photo */}
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="flex-shrink-0"
            >
              <div className="w-32 h-32 glass-panel rounded-full flex items-center justify-center text-white font-bold text-4xl overflow-hidden shadow-2xl shadow-emerald-900/20">
                {doctor.photoUrl ? (
                  <img src={doctor.photoUrl} alt="Doctor" className="w-full h-full object-cover" />
                ) : (
                  doctor.fullName.split(' ').map(n => n[0]).join('').slice(0, 2)
                )}
              </div>
            </motion.div>

            {/* Doctor Info */}
            <motion.div
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.1 }}
              className="flex-1"
            >
              <h1 className="text-2xl font-bold mb-1">{doctor.fullName}</h1>
              <p className="text-lg text-white/90 mb-3">{doctor.specialty}</p>
              
              <div className="flex flex-wrap gap-4 text-sm">
                {doctor.rating && (
                  <div className="flex items-center gap-1">
                    <Star size={16} fill="currentColor" className="text-amber-400" />
                    <span>{doctor.rating}</span>
                    <span className="text-white/70">({doctor.totalReviews} reviews)</span>
                  </div>
                )}
                {doctor.yearsExperience && (
                  <div className="flex items-center gap-1">
                    <Calendar size={16} />
                    <span>{doctor.yearsExperience} yrs experience</span>
                  </div>
                )}
                <div className="flex items-center gap-1">
                  <CheckCircle size={16} />
                  <span>Licensed & Certified</span>
                </div>
              </div>
            </motion.div>

            {/* Quick Actions */}
            <motion.div
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="flex flex-col gap-3"
            >
              {doctor.consultFee && (
                <div className="bg-white/20 backdrop-blur-sm rounded-xl px-6 py-3 text-center">
                  <p className="text-sm text-white/70">Consultation Fee</p>
                  <p className="text-xl font-bold">${doctor.consultFee}</p>
                </div>
              )}
              <button className="bg-white text-emerald-600 hover:bg-white/90 px-6 py-3 rounded-xl font-semibold transition-colors shadow-md shadow-emerald-900/10">
                Book Instant Appointment
              </button>
            </motion.div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-8">
            {/* About */}
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="glass-card dark:bg-slate-900/60 dark:border dark:border-slate-800/80 rounded-2xl shadow-sm p-6"
            >
              <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-3 flex items-center gap-2">
                <BookOpen size={18} className="text-emerald-600 dark:text-emerald-400" />
                About Doctor
              </h2>
              {doctor.bio ? (
                <p className="text-slate-600 dark:text-slate-300 leading-relaxed">{doctor.bio}</p>
              ) : (
                <p className="text-slate-500 dark:text-slate-400">Biography not available at this moment</p>
              )}
            </motion.div>

            {/* Book Appointment */}
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="glass-card dark:bg-slate-950/65 rounded-2xl border border-slate-100 dark:border-slate-800/80 shadow-sm p-6"
            >
              <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-3 flex items-center gap-2">
                <Calendar size={18} className="text-emerald-600" />
                Book Appointment
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Date
                  </label>
                  <input
                    type="date"
                    value={selectedDate}
                    onChange={(e) => setSelectedDate(e.target.value)}
                    min={new Date().toISOString().split('T')[0]}
                    className="w-full px-4 py-2 border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Time
                  </label>
                  <select
                    value={selectedTime}
                    onChange={(e) => setSelectedTime(e.target.value)}
                    disabled={!selectedDate || isLoadingSlots}
                    className="w-full px-4 py-2 border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-850 dark:text-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 disabled:bg-slate-100 dark:disabled:bg-slate-800 disabled:cursor-not-allowed"
                  >
                    <option value="">
                      {isLoadingSlots ? 'Loading slots...' : availableSlots.length === 0 ? 'No available slots' : 'Choose Time'}
                    </option>
                    {availableSlots.map(slot => (
                      <option key={slot} value={slot}>{slot}</option>
                    ))}
                  </select>
                </div>
              </div>

              <button
                onClick={() => handleBookAppointment()}
                disabled={!selectedDate || !selectedTime}
                className="w-full bg-gradient-primary hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed text-white py-3 rounded-xl font-semibold transition-all shadow-lg shadow-emerald-500/20"
              >
                Confirm Appointment
              </button>
            </motion.div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Contact Info */}
            <motion.div
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="bg-white dark:bg-slate-900/60 dark:border dark:border-slate-800/80 rounded-2xl shadow-sm p-6"
            >
              <h3 className="font-bold text-slate-800 dark:text-slate-100 mb-3 text-sm">Contact Information</h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3 text-slate-600 dark:text-slate-300">
                  <Mail size={18} className="text-slate-400 dark:text-slate-500" />
                  <span className="text-sm">{doctor.email}</span>
                </div>
                <div className="flex items-center gap-3 text-slate-600 dark:text-slate-300">
                  <Phone size={18} className="text-slate-400 dark:text-slate-500" />
                  <span className="text-sm">N/A</span>
                </div>
                <div className="flex items-center gap-3 text-slate-600 dark:text-slate-300">
                  <MapPin size={18} className="text-slate-400 dark:text-slate-500" />
                  <span className="text-sm">Riyadh, Saudi Arabia</span>
                </div>
              </div>
            </motion.div>

            {/* Experience */}
            {doctor.yearsExperience && (
              <motion.div
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.6 }}
                className="bg-white dark:bg-slate-900/60 dark:border dark:border-slate-800/80 rounded-2xl shadow-sm p-6"
              >
                <h3 className="font-bold text-slate-800 dark:text-slate-100 mb-3 text-sm">Experience</h3>
                <div className="flex items-center gap-3">
                  <Award size={18} className="text-emerald-600 dark:text-emerald-400" />
                  <span className="text-slate-600 dark:text-slate-300">{doctor.yearsExperience} yrs experience</span>
                </div>
              </motion.div>
            )}

            {/* Languages */}
            <motion.div
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.7 }}
              className="bg-white dark:bg-slate-900/60 dark:border dark:border-slate-800/80 rounded-2xl shadow-sm p-6"
            >
              <h3 className="font-bold text-slate-800 dark:text-slate-100 mb-3 text-sm flex items-center gap-2">
                <Languages size={16} className="text-emerald-600 dark:text-emerald-400" />
                Languages
              </h3>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle size={16} className="text-emerald-600 dark:text-emerald-400" />
                  <span className="text-slate-600 dark:text-slate-300">Arabic</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle size={16} className="text-emerald-600 dark:text-emerald-400" />
                  <span className="text-slate-600 dark:text-slate-300">English</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
      
      <ReasonForVisitModal 
        open={showReasonModal} 
        onClose={() => setShowReasonModal(false)} 
        onComplete={(reason) => handleBookAppointment(reason)} 
      />
    </div>
  )
}
