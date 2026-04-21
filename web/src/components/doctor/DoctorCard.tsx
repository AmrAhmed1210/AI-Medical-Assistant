import { Star, Clock, DollarSign } from 'lucide-react'
import type { DoctorDetailDto } from '@/lib/types'
import { cn } from '@/lib/utils'

interface DoctorCardProps {
  doctor: DoctorDetailDto
  onClick?: () => void
}

export function DoctorCard({ doctor, onClick }: DoctorCardProps) {
  return (
    <div
      onClick={onClick}
      className={cn(
        'bg-white rounded-xl border border-gray-100 p-5 shadow-sm',
        onClick && 'cursor-pointer hover:shadow-md transition-shadow'
      )}
    >
      <div className="flex items-start gap-4">
        <div className="w-14 h-14 rounded-full bg-primary-100 flex items-center justify-center flex-shrink-0 overflow-hidden">
          {doctor.photoUrl ? (
            <img src={doctor.photoUrl} alt={doctor.fullName} className="w-full h-full object-cover" />
          ) : (
            <span className="text-primary-700 text-xl font-bold">{doctor.fullName.charAt(0)}</span>
          )}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-gray-800">{doctor.fullName}</h3>
          <p className="text-sm text-primary-600">{doctor.specialityNameAr}</p>
          <div className="flex items-center gap-1 mt-1">
            <Star size={13} className="text-amber-400 fill-amber-400" />
            <span className="text-xs text-gray-600">{(doctor.rating ?? 0).toFixed(1)}</span>
          </div>
        </div>
        <div className={cn(
          'w-2 h-2 rounded-full mt-1',
          doctor.isAvailable ? 'bg-green-500' : 'bg-gray-300'
        )} />
      </div>
      <div className="flex gap-4 mt-4 pt-4 border-t border-gray-50">
        <div className="flex items-center gap-1.5 text-xs text-gray-500">
          <Clock size={12} />
          {doctor.yearsExperience} سنة خبرة
        </div>
        <div className="flex items-center gap-1.5 text-xs text-gray-500">
          <DollarSign size={12} />
          {doctor.consultFee} ج.م
        </div>
      </div>
    </div>
  )
}
