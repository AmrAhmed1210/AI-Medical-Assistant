import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Search, Filter, Star, MapPin, Calendar, HeartPulse } from 'lucide-react'
import { toast } from 'react-hot-toast'
import { doctorApi } from '@/api/doctorApi'
import type { DoctorDetailDto } from '@/lib/types'

interface DoctorListItem {
  id: string
  fullName: string
  email: string
  specialty: string
  bio: string | null
  photoUrl: string | null
  consultFee: number | null
  yearsExperience: number | null
  rating?: number
  totalReviews?: number
}

export default function DoctorsList() {
  const navigate = useNavigate()
  const [doctors, setDoctors] = useState<DoctorListItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedSpecialty, setSelectedSpecialty] = useState('')

  const specialties = [
    'Cardiology', 'Dermatology', 'Pediatrics', 'Orthopedics', 
    'Neurology', 'Psychiatry', 'General Practice', 'Internal Medicine'
  ]

  const fetchDoctors = async () => {
    setIsLoading(true)
    try {
      const response = await doctorApi.getAllDoctors()
      
      // Transform API response to our format
      const transformedDoctors: DoctorListItem[] = response.map(doctor => ({
        id: doctor.id,
        fullName: doctor.fullName,
        email: doctor.email,
        specialty: doctor.specialty,
        bio: doctor.bio,
        photoUrl: doctor.photoUrl,
        consultFee: doctor.consultFee,
        yearsExperience: doctor.yearsExperience,
        rating: 4.5, // Mock data - should come from API
        totalReviews: 12 // Mock data - should come from API
      }))
      
      setDoctors(transformedDoctors)
    } catch (error: any) {
      console.error('Error fetching doctors:', error)
      toast.error('فشل تحميل قائمة الأطباء')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchDoctors()
  }, [])

  const filteredDoctors = doctors.filter(doctor =>
    doctor.fullName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doctor.specialty.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="min-h-screen bg-slate-50 p-6" dir="rtl">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-slate-800 mb-2">قائمة الأطباء</h1>
          <p className="text-slate-600">ابحث عن الطبيب المناسب لك</p>
        </motion.div>

        {/* Search and Filters */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-2xl shadow-sm p-6 mb-8"
        >
          <div className="flex flex-col lg:flex-row gap-4">
            {/* Search */}
            <div className="flex-1 relative">
              <Search className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400" size={20} />
              <input
                type="text"
                placeholder="ابحث عن طبيب بالاسم أو التخصص..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pr-12 pl-4 py-3 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              />
            </div>

            {/* Specialty Filter */}
            <div className="lg:w-64">
              <select
                value={selectedSpecialty}
                onChange={(e) => setSelectedSpecialty(e.target.value)}
                className="w-full px-4 py-3 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              >
                <option value="">جميع التخصصات</option>
                {specialties.map(specialty => (
                  <option key={specialty} value={specialty}>{specialty}</option>
                ))}
              </select>
            </div>
          </div>
        </motion.div>

        {/* Doctors Grid */}
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500"></div>
          </div>
        ) : filteredDoctors.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-16"
          >
            <HeartPulse className="mx-auto text-slate-300 mb-4" size={48} />
            <h3 className="text-xl font-semibold text-slate-600 mb-2">لا يوجد أطباء</h3>
            <p className="text-slate-500">لم يتم العثور على أطباء مطابقين لبحثك</p>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredDoctors.map((doctor, index) => (
              <motion.div
                key={doctor.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="bg-white rounded-2xl shadow-sm hover:shadow-lg transition-shadow p-6"
              >
                {/* Doctor Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-4">
                    <div className="w-16 h-16 bg-gradient-to-br from-emerald-400 to-teal-500 rounded-full flex items-center justify-center text-white font-bold text-xl">
                      {doctor.fullName.split(' ').map(n => n[0]).join('').slice(0, 2)}
                    </div>
                    <div>
                      <h3 className="font-semibold text-slate-800">{doctor.fullName}</h3>
                      <p className="text-sm text-slate-600">{doctor.specialty}</p>
                    </div>
                  </div>
                  {doctor.rating && (
                    <div className="flex items-center gap-1 text-amber-500">
                      <Star size={16} fill="currentColor" />
                      <span className="text-sm font-medium">{doctor.rating}</span>
                    </div>
                  )}
                </div>

                {/* Doctor Info */}
                <div className="space-y-3 mb-4">
                  {doctor.yearsExperience && (
                    <div className="flex items-center gap-2 text-sm text-slate-600">
                      <Calendar size={16} />
                      <span>{doctor.yearsExperience} سنوات خبرة</span>
                    </div>
                  )}
                  {doctor.consultFee && (
                    <div className="flex items-center gap-2 text-sm text-slate-600">
                      <span className="font-medium text-emerald-600">{doctor.consultFee} ر.س</span>
                      <span>رسوم الاستشارة</span>
                    </div>
                  )}
                  {doctor.bio && (
                    <p className="text-sm text-slate-600 line-clamp-2">{doctor.bio}</p>
                  )}
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  <button className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white py-2 px-4 rounded-xl text-sm font-medium transition-colors">
                    حجز موعد
                  </button>
                  <button 
                    onClick={() => navigate(`/doctor/${doctor.id}`)}
                    className="flex-1 border border-slate-200 hover:bg-slate-50 text-slate-700 py-2 px-4 rounded-xl text-sm font-medium transition-colors"
                  >
                    التفاصيل
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
