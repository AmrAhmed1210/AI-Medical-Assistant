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
          <label className="block text-sm font-semibold text-gray-700 mb-2">الاسم الكامل *</label>
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
          <label className="block text-sm font-semibold text-gray-700 mb-2">البريد الإلكتروني *</label>
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
          <label className="block text-sm font-semibold text-gray-700 mb-2">كلمة المرور *</label>
          <Input
            type="password"
            value={form.passwordHash}
            onChange={(e) => onChange('passwordHash', e.target.value)}
            placeholder="•••••••• (8 أحرف على الأقل)"
            icon={<Lock className="w-4 h-4" />}
            error={errors.passwordHash}
          />
        </div>
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">الدور *</label>
          <Select
            value={form.role}
            onChange={(e) => onChange('role', e.target.value as UserRole)}
          >
            <option value="Patient">👤 مريض</option>
            <option value="Doctor">🩺 طبيب</option>
            <option value="Admin">👑 مدير نظام</option>
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
              <p className="text-sm font-bold text-gray-800">بيانات الطبيب المهنية</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">التخصص (إنجليزي) *</label>
                <Input
                  type="text"
                  value={form.specialityName}
                  onChange={(e) => onChange('specialityName', e.target.value)}
                  placeholder="Cardiology"
                  error={errors.specialityName}
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">التخصص (عربي) *</label>
                <Input
                  type="text"
                  value={form.specialityNameAr}
                  onChange={(e) => onChange('specialityNameAr', e.target.value)}
                  placeholder="أمراض القلب والأوعية الدموية"
                  error={errors.specialityNameAr}
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">سنوات الخبرة</label>
                <Input
                  type="number"
                  min={0}
                  max={50}
                  value={form.yearsExperience || ''}
                  onChange={(e) => onChange('yearsExperience', Number(e.target.value) || 0)}
                  placeholder="مثال: 10"
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">رسوم الاستشارة (ج.م)</label>
                <Input
                  type="number"
                  min={0}
                  value={form.consultationFee || ''}
                  onChange={(e) => onChange('consultationFee', Number(e.target.value) || 0)}
                  placeholder="مثال: 500"
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">نبذة تعريفية</label>
              <textarea
                rows={3}
                value={form.bio}
                onChange={(e) => onChange('bio', e.target.value)}
                placeholder="اكتب نبذة مختصرة عن خبرة الطبيب وإنجازاته..."
                className="w-full px-4 py-3 text-sm border border-gray-200 rounded-2xl bg-white/50 focus:outline-none focus:ring-4 focus:ring-blue-200 focus:border-blue-400 transition-all duration-200 resize-none"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}