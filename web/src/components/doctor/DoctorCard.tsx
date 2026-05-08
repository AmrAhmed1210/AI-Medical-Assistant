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
        'bg-white rounded-2xl border border-slate-100 p-5 shadow-sm',
        onClick && 'cursor-pointer hover:shadow-md transition-shadow'
      )}
    >
      <div className="flex items-start gap-4">
        <div className="w-14 h-14 rounded-full bg-emerald-50 flex items-center justify-center flex-shrink-0 overflow-hidden">
          {doctor.photoUrl ? (
            <img src={doctor.photoUrl} alt={doctor.fullName} className="w-full h-full object-cover" />
          ) : (
            <span className="text-emerald-700 text-xl font-bold">{doctor.fullName.charAt(0)}</span>
          )}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-slate-800">{doctor.fullName}</h3>
          <p className="text-sm text-emerald-600">{doctor.specialityNameAr}</p>
          <div className="flex items-center gap-1 mt-1">
            <Star size={13} className="text-amber-400 fill-amber-400" />
            <span className="text-xs text-slate-600">{(doctor.rating ?? 0).toFixed(1)}</span>
          </div>
        </div>
        <div className={cn(
          'w-2 h-2 rounded-full mt-1',
          doctor.isAvailable ? 'bg-green-500' : 'bg-slate-300'
        )} />
      </div>
      <div className="flex gap-4 mt-4 pt-4 border-t border-slate-100">
        <div className="flex items-center gap-1.5 text-xs text-slate-500">
          <Clock size={12} />
          {doctor.yearsExperience} سنة خبرة
        </div>
        <div className="flex items-center gap-1.5 text-xs text-slate-500">
          <DollarSign size={12} />
          {doctor.consultFee} ر.س
        </div>
      </div>
    </div>
  )
}
