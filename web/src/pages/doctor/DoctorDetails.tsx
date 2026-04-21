import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  ArrowLeft, Star, MapPin, Calendar, Clock, Phone, Mail, 
  HeartPulse, Award, Languages, CheckCircle, BookOpen
} from 'lucide-react'
import { toast } from 'react-hot-toast'
import { doctorApi } from '@/api/doctorApi'
import axiosInstance from '@/api/axiosInstance'
import type { DoctorDetailDto } from '@/lib/types'

interface DoctorDetails extends DoctorDetailDto {
  rating?: number
  totalReviews?: number
  availableSlots?: string[]
}

export default function DoctorDetails() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [doctor, setDoctor] = useState<DoctorDetails | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [selectedDate, setSelectedDate] = useState('')
  const [selectedTime, setSelectedTime] = useState('')

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
        availableSlots: [
          '09:00', '09:30', '10:00', '10:30', '11:00', '11:30',
          '14:00', '14:30', '15:00', '15:30', '16:00', '16:30'
        ] // Mock data - should come from API
      }
      
      setDoctor(transformedDoctor)
    } catch (error: any) {
      console.error('Error fetching doctor details:', error)
      toast.error('فشل تحميل تفاصيل الطبيب')
      navigate('/doctors')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchDoctorDetails()
  }, [id])

  const handleBookAppointment = async () => {
    if (!doctor || !selectedDate || !selectedTime) {
      toast.error('يرجى اختيار التاريخ والوقت')
      return
    }

    try {
      const appointmentData = {
        doctorId: doctor.id,
        scheduledAt: `${selectedDate}T${selectedTime}:00`,
      }
      
      await axiosInstance.post('/api/appointments', appointmentData)
      toast.success('تم حجز الموعد بنجاح!')
      navigate('/appointments')
    } catch (error: any) {
      console.error('Error booking appointment:', error)
      toast.error(error.response?.data?.message || 'فشل حجز الموعد')
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
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="text-center">
          <HeartPulse className="mx-auto text-slate-300 mb-4" size={48} />
          <h3 className="text-xl font-semibold text-slate-600 mb-2">الطبيب غير موجود</h3>
          <button 
            onClick={() => navigate('/doctors')}
            className="text-emerald-600 hover:text-emerald-700 font-medium"
          >
            العودة لقائمة الأطباء
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-50" dir="rtl">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-600 to-teal-500 text-white">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <button
            onClick={() => navigate('/doctors')}
            className="flex items-center gap-2 text-white/90 hover:text-white mb-6 transition-colors"
          >
            <ArrowLeft size={20} />
            العودة لقائمة الأطباء
          </button>

          <div className="flex flex-col lg:flex-row items-start gap-8">
            {/* Doctor Photo */}
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="flex-shrink-0"
            >
              <div className="w-32 h-32 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center text-white font-bold text-4xl">
                {doctor.fullName.split(' ').map(n => n[0]).join('').slice(0, 2)}
              </div>
            </motion.div>

            {/* Doctor Info */}
            <motion.div
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.1 }}
              className="flex-1"
            >
              <h1 className="text-3xl font-bold mb-2">{doctor.fullName}</h1>
              <p className="text-xl text-white/90 mb-4">{doctor.specialty}</p>
              
              <div className="flex flex-wrap gap-4 text-sm">
                {doctor.rating && (
                  <div className="flex items-center gap-1">
                    <Star size={16} fill="currentColor" />
                    <span>{doctor.rating}</span>
                    <span className="text-white/70">({doctor.totalReviews} تقييم)</span>
                  </div>
                )}
                {doctor.yearsExperience && (
                  <div className="flex items-center gap-1">
                    <Calendar size={16} />
                    <span>{doctor.yearsExperience} سنوات خبرة</span>
                  </div>
                )}
                <div className="flex items-center gap-1">
                  <CheckCircle size={16} />
                  <span>مرخص ومعتمد</span>
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
                  <p className="text-sm text-white/70">رسوم الاستشارة</p>
                  <p className="text-2xl font-bold">{doctor.consultFee} ر.س</p>
                </div>
              )}
              <button className="bg-white text-emerald-600 hover:bg-white/90 px-6 py-3 rounded-xl font-semibold transition-colors">
                حجز موعد فوري
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
              className="bg-white rounded-2xl shadow-sm p-6"
            >
              <h2 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                <BookOpen size={20} className="text-emerald-600" />
                نبذة عن الطبيب
              </h2>
              {doctor.bio ? (
                <p className="text-slate-600 leading-relaxed">{doctor.bio}</p>
              ) : (
                <p className="text-slate-500">لا توجد نبذة متاحة حالياً</p>
              )}
            </motion.div>

            {/* Book Appointment */}
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="bg-white rounded-2xl shadow-sm p-6"
            >
              <h2 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                <Calendar size={20} className="text-emerald-600" />
                حجز موعد
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    التاريخ
                  </label>
                  <input
                    type="date"
                    value={selectedDate}
                    onChange={(e) => setSelectedDate(e.target.value)}
                    min={new Date().toISOString().split('T')[0]}
                    className="w-full px-4 py-2 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    الوقت
                  </label>
                  <select
                    value={selectedTime}
                    onChange={(e) => setSelectedTime(e.target.value)}
                    className="w-full px-4 py-2 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  >
                    <option value="">اختر الوقت</option>
                    {doctor.availableSlots?.map(slot => (
                      <option key={slot} value={slot}>{slot}</option>
                    ))}
                  </select>
                </div>
              </div>

              <button
                onClick={handleBookAppointment}
                disabled={!selectedDate || !selectedTime}
                className="w-full bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white py-3 rounded-xl font-semibold transition-colors"
              >
                تأكيد الحجز
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
              className="bg-white rounded-2xl shadow-sm p-6"
            >
              <h3 className="font-bold text-slate-800 mb-4">معلومات التواصل</h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3 text-slate-600">
                  <Mail size={18} className="text-slate-400" />
                  <span className="text-sm">{doctor.email}</span>
                </div>
                <div className="flex items-center gap-3 text-slate-600">
                  <Phone size={18} className="text-slate-400" />
                  <span className="text-sm">غير متوفر</span>
                </div>
                <div className="flex items-center gap-3 text-slate-600">
                  <MapPin size={18} className="text-slate-400" />
                  <span className="text-sm">الرياض، السعودية</span>
                </div>
              </div>
            </motion.div>

            {/* Experience */}
            {doctor.yearsExperience && (
              <motion.div
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.6 }}
                className="bg-white rounded-2xl shadow-sm p-6"
              >
                <h3 className="font-bold text-slate-800 mb-4">الخبرات</h3>
                <div className="flex items-center gap-3">
                  <Award size={18} className="text-emerald-600" />
                  <span className="text-slate-600">{doctor.yearsExperience} سنوات خبرة</span>
                </div>
              </motion.div>
            )}

            {/* Languages */}
            <motion.div
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.7 }}
              className="bg-white rounded-2xl shadow-sm p-6"
            >
              <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
                <Languages size={18} className="text-emerald-600" />
                اللغات
              </h3>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle size={16} className="text-emerald-600" />
                  <span className="text-slate-600">العربية</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle size={16} className="text-emerald-600" />
                  <span className="text-slate-600">الإنجليزية</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  )
}
