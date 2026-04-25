import { motion, AnimatePresence } from 'framer-motion'
import { AlertCircle, User, Mail, Lock, Stethoscope } from 'lucide-react'
import type { UserRole, CreateUserRequest } from '@/lib/types'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'

interface UserFormProps {
  form: CreateUserRequest
  errors: Partial<Record<keyof CreateUserRequest, string>>
  onChange: (key: keyof CreateUserRequest, value: unknown) => void
}

export const UserForm = ({ form, errors, onChange }: UserFormProps) => {
  return (
    <div className="space-y-5">
      {/* Info Notice */}
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-2xl p-4 text-sm text-blue-800 flex gap-3"
      >
        <div className="flex-shrink-0 p-1.5 bg-blue-100 rounded-xl">
          <AlertCircle className="w-4 h-4" />
        </div>
        <p>فقط مدير النظام يمكنه إضافة مستخدمين جدد. سيتم إرسال بيانات الدخول إلى البريد الإلكتروني للمستخدم.</p>
      </motion.div>

      {/* Basic Fields */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Full Name / الاسم الكامل *</label>
          <Input
            type="text"
            value={form.fullName}
            onChange={(e) => onChange('fullName', e.target.value)}
            placeholder="د. أحمد محمد علي"
            icon={<User className="w-4 h-4" />}
            error={errors.fullName}
          />
        </div>
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Email / البريد الإلكتروني *</label>
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
          <label className="block text-sm font-semibold text-gray-700 mb-2">Password / كلمة المرور *</label>
          <Input
            type="password"
            value={form.password}
            onChange={(e) => onChange('password', e.target.value)}
            placeholder="•••••••• (8 أحرف على الأقل)"
            icon={<Lock className="w-4 h-4" />}
            error={errors.password}
          />
        </div>
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Role / الدور *</label>
          <Select
            value={form.role}
            onChange={(e) => onChange('role', e.target.value as UserRole)}
          >
            <option value="Patient">👤 Patient / مريض</option>
            <option value="Doctor">🩺 Doctor / طبيب</option>
            <option value="Admin">👑 Admin / مدير نظام</option>
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
            className="border-t border-gray-100 pt-5 space-y-4"
          >
            <div className="flex items-center gap-2 pb-2">
              <Stethoscope className="w-5 h-5 text-emerald-500" />
              <p className="text-sm font-bold text-gray-800">Doctor Professional Info / بيانات الطبيب المهنية</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Specialty (EN) / التخصص بالإنجليزي *</label>
                <Input
                  type="text"
                  value={form.specialityName}
                  onChange={(e) => onChange('specialityName', e.target.value)}
                  placeholder="Cardiology"
                  error={errors.specialityName}
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Specialty (AR) / التخصص بالعربي *</label>
                <Input
                  type="text"
                  value={form.specialityNameAr}
                  onChange={(e) => onChange('specialityNameAr', e.target.value)}
                  placeholder="أمراض القلب والأوعية الدموية"
                  error={errors.specialityNameAr}
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Years of Experience / سنوات الخبرة</label>
                <Input
                  type="number"
                  min={0}
                  max={60}
                  value={form.yearsExperience === undefined ? '' : form.yearsExperience}
                  onChange={(e) => onChange('yearsExperience', Number(e.target.value))}
                  placeholder="مثال: 10"
                  error={errors.yearsExperience}
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Consultation Fee (EGP) / رسوم الاستشارة</label>
                <Input
                  type="number"
                  min={0}
                  value={form.consultationFee === undefined ? '' : form.consultationFee}
                  onChange={(e) => onChange('consultationFee', Number(e.target.value))}
                  placeholder="مثال: 500"
                  error={errors.consultationFee}
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">Bio / نبذة تعريفية</label>
              <textarea
                rows={3}
                value={form.bio}
                onChange={(e) => onChange('bio', e.target.value)}
                placeholder="اكتب نبذة مختصرة عن خبرة الطبيب وإنجازاته..."
                className="w-full px-4 py-3 text-sm bg-white text-gray-900 placeholder-gray-400 border border-gray-200 rounded-2xl focus:outline-none focus:ring-4 focus:ring-blue-200 focus:border-blue-400 shadow-sm transition-all duration-200 resize-none"
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