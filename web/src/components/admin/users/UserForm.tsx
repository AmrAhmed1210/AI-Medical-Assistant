import { useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertCircle, User, Mail, Lock, Stethoscope, MapPin } from 'lucide-react'
import type { UserRole, CreateUserRequest } from '@/lib/types'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import axiosInstance from '@/api/axiosInstance'

interface SpecialtyOption {
  id: number
  name: string
  nameAr?: string | null
}

interface UserFormProps {
  form: CreateUserRequest
  errors: Partial<Record<keyof CreateUserRequest, string>>
  onChange: (key: keyof CreateUserRequest, value: unknown) => void
}

export const UserForm = ({ form, errors, onChange }: UserFormProps) => {
  const [specialties, setSpecialties] = useState<SpecialtyOption[]>([])
  const [isWritingSpecialty, setIsWritingSpecialty] = useState(false)

  useEffect(() => {
    axiosInstance.get<SpecialtyOption[]>('/api/specialties')
      .then((res) => setSpecialties(res.data ?? []))
      .catch(() => setSpecialties([]))
  }, [])

  const selectedSpecialtyValue = useMemo(() => {
    if (isWritingSpecialty) return '__custom'
    const selected = specialties.find(s => s.name.toLowerCase() === (form.specialityName ?? '').toLowerCase())
    if (selected) return String(selected.id)
    return form.specialityName ? '__custom' : ''
  }, [form.specialityName, isWritingSpecialty, specialties])

  const isCustomSpecialty = selectedSpecialtyValue === '__custom'

  const handleSpecialtyChange = (value: string) => {
    if (value === '__custom') {
      setIsWritingSpecialty(true)
      onChange('specialityName', '')
      onChange('specialityNameAr', '')
      return
    }

    const selected = specialties.find(s => String(s.id) === value)
    setIsWritingSpecialty(false)
    onChange('specialityName', selected?.name ?? '')
    onChange('specialityNameAr', selected?.nameAr || selected?.name || '')
  }

  return (
    <div className="space-y-5">
      {/* Info Notice */}
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-slate-950 border border-blue-200 dark:border-slate-800 rounded-2xl p-4 text-sm text-blue-800 dark:text-blue-450 flex gap-3"
      >
        <div className="flex-shrink-0 p-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-xl">
          <AlertCircle className="w-4 h-4" />
        </div>
        <p>Only system administrators can add new users. Access credentials will be automatically sent to the user's email address.</p>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 dark:text-slate-350 mb-2">Full Name *</label>
          <Input
            type="text"
            value={form.fullName}
            onChange={(e) => onChange('fullName', e.target.value)}
            placeholder="Dr. John Doe"
            icon={<User className="w-4 h-4" />}
            error={errors.fullName}
          />
        </div>
        <div>
          <label className="block text-sm font-semibold text-gray-700 dark:text-slate-350 mb-2">Email *</label>
          <Input
            type="email"
            value={form.email}
            onChange={(e) => onChange('email', e.target.value)}
            placeholder="user@medbook.com"
            icon={<Mail className="w-4 h-4" />}
            error={errors.email}
          />
        </div>
        <div>
          <label className="block text-sm font-semibold text-gray-700 dark:text-slate-350 mb-2">Password *</label>
          <Input
            type="password"
            value={form.password}
            onChange={(e) => onChange('password', e.target.value)}
            placeholder="•••••••• (At least 8 characters)"
            icon={<Lock className="w-4 h-4" />}
            error={errors.password}
          />
        </div>
        <div>
          <label className="block text-sm font-semibold text-gray-700 dark:text-slate-350 mb-2">Role *</label>
          <Select
            value={form.role}
            onChange={(e) => onChange('role', e.target.value as UserRole)}
          >
            <option value="Patient">👤 Patient</option>
            <option value="Doctor">🩺 Doctor</option>
            <option value="Admin">👑 Admin</option>
          </Select>
        </div>
      </div>

      {/* Doctor-specific Fields */}
      <AnimatePresence>
        {form.role === 'Doctor' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-gray-100 dark:border-slate-800/80 pt-5 space-y-4"
          >
            <div className="flex items-center gap-2 pb-2">
              <Stethoscope className="w-5 h-5 text-emerald-500" />
              <p className="text-sm font-bold text-gray-800 dark:text-white">Doctor Professional Info</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="md:col-span-2">
                <label className="block text-sm font-semibold text-gray-700 dark:text-slate-350 mb-2">Specialty *</label>
                <Select value={selectedSpecialtyValue} onChange={(e) => handleSpecialtyChange(e.target.value)}>
                  <option value="">Select specialty</option>
                  {specialties.map((specialty) => (
                    <option key={specialty.id} value={specialty.id}>
                      {specialty.name}{specialty.nameAr ? ` - ${specialty.nameAr}` : ''}
                    </option>
                  ))}
                  <option value="__custom">Not listed, write it manually</option>
                </Select>
                {errors.specialityName && (
                  <p className="mt-1 text-sm text-red-500">{errors.specialityName}</p>
                )}
              </div>
              {isCustomSpecialty && (
                <>
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 dark:text-slate-350 mb-2">Specialty (EN) *</label>
                    <Input
                      type="text"
                      value={form.specialityName ?? ''}
                      onChange={(e) => onChange('specialityName', e.target.value)}
                      placeholder="Cardiology"
                      error={errors.specialityName}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 dark:text-slate-350 mb-2">Specialty (AR) *</label>
                    <Input
                      type="text"
                      value={form.specialityNameAr ?? ''}
                      onChange={(e) => onChange('specialityNameAr', e.target.value)}
                      placeholder="Cardiology & Vascular Medicine"
                      error={errors.specialityNameAr}
                    />
                  </div>
                </>
              )}
              <div className="md:col-span-2">
                <label className="block text-sm font-semibold text-gray-700 dark:text-slate-350 mb-2">Clinic Address *</label>
                <Input
                  type="text"
                  value={form.location ?? ''}
                  onChange={(e) => onChange('location', e.target.value)}
                  placeholder="Cairo, Helwan, main street"
                  icon={<MapPin className="w-4 h-4" />}
                  error={errors.location}
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 dark:text-slate-350 mb-2">Years of Experience</label>
                <Input
                  type="number"
                  min={0}
                  max={60}
                  value={form.yearsExperience === undefined ? '' : form.yearsExperience}
                  onChange={(e) => onChange('yearsExperience', Number(e.target.value))}
                  placeholder="e.g. 10"
                  error={errors.yearsExperience}
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 dark:text-slate-350 mb-2">Consultation Fee (USD)</label>
                <Input
                  type="number"
                  min={0}
                  value={form.consultationFee === undefined ? '' : form.consultationFee}
                  onChange={(e) => onChange('consultationFee', Number(e.target.value))}
                  placeholder="e.g. 150"
                  error={errors.consultationFee}
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-semibold text-gray-700 dark:text-slate-350 mb-2">Bio</label>
              <textarea
                rows={3}
                value={form.bio}
                onChange={(e) => onChange('bio', e.target.value)}
                placeholder="Write a brief summary of the doctor's experience, achievements, and credentials..."
                className="w-full px-4 py-3 text-sm bg-white dark:bg-slate-900 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-slate-500 border border-gray-200 dark:border-slate-800 rounded-2xl focus:outline-none focus:ring-4 focus:ring-blue-200 focus:border-blue-400 shadow-sm transition-all duration-200 resize-none"
              />
              {errors.bio && (
                <p className="mt-1 text-sm text-red-500">{errors.bio}</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
