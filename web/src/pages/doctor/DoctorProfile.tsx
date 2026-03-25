import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { Camera, Save, Star, Clock, DollarSign } from 'lucide-react'
import { useDoctorProfile } from '@/hooks/useDoctor'
import { useDoctorStore } from '@/store/doctorStore'
import { doctorApi } from '@/api/doctorApi'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import toast from 'react-hot-toast'

export default function DoctorProfile() {
  const { profile, isLoading } = useDoctorProfile()
  const { updateProfile } = useDoctorStore()
  const fileRef = useRef<HTMLInputElement>(null)

  const [form, setForm] = useState({
    fullName: profile?.fullName ?? '',
    bio: profile?.bio ?? '',
    yearsExperience: profile?.yearsExperience ?? 0,
    consultationFee: profile?.consultationFee ?? 0,
    isAvailable: profile?.isAvailable ?? true,
    phone: profile?.phone ?? '',
  })
  const [saving, setSaving] = useState(false)
  const [photoUploading, setPhotoUploading] = useState(false)

  // Sync form with profile
  useState(() => {
    if (profile) {
      setForm({
        fullName: profile.fullName,
        bio: profile.bio,
        yearsExperience: profile.yearsExperience,
        consultationFee: profile.consultationFee,
        isAvailable: profile.isAvailable,
        phone: profile.phone ?? '',
      })
    }
  })

  if (isLoading) return <PageLoader />

  const handleSave = async () => {
    setSaving(true)
    try {
      await updateProfile(form)
      toast.success('تم حفظ الملف الشخصي')
    } catch {
      toast.error('فشل حفظ البيانات')
    } finally {
      setSaving(false)
    }
  }

  const handlePhotoChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setPhotoUploading(true)
    try {
      await doctorApi.uploadPhoto(file)
      toast.success('تم رفع الصورة')
    } catch {
      toast.error('فشل رفع الصورة')
    } finally {
      setPhotoUploading(false)
    }
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-bold text-gray-800">الملف الشخصي</h1>
        <p className="text-sm text-gray-500 mt-0.5">إدارة بياناتك الشخصية والمهنية</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Profile Preview */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="text-center">
            <div className="relative inline-block mb-4">
              <div className="w-24 h-24 rounded-full bg-primary-100 flex items-center justify-center mx-auto overflow-hidden">
                {profile?.photoUrl ? (
                  <img src={profile.photoUrl} alt={profile.fullName} className="w-full h-full object-cover" />
                ) : (
                  <span className="text-primary-700 text-3xl font-bold">{form.fullName.charAt(0)}</span>
                )}
              </div>
              <button
                onClick={() => fileRef.current?.click()}
                disabled={photoUploading}
                className="absolute bottom-0 left-0 w-7 h-7 bg-primary-600 rounded-full flex items-center justify-center text-white hover:bg-primary-700 transition-colors"
              >
                <Camera size={13} />
              </button>
              <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handlePhotoChange} />
            </div>
            <h3 className="font-semibold text-gray-800">{form.fullName || 'اسم الطبيب'}</h3>
            <p className="text-sm text-primary-600 mt-0.5">{profile?.specialityNameAr}</p>
            <div className="flex items-center justify-center gap-1 mt-2">
              <Star size={13} className="text-amber-400 fill-amber-400" />
              <span className="text-sm text-gray-600">{profile?.rating.toFixed(1)}</span>
            </div>
            <div className="flex items-center justify-center gap-4 mt-4 pt-4 border-t border-gray-50 text-xs text-gray-500">
              <div className="flex items-center gap-1">
                <Clock size={11} />
                {form.yearsExperience} سنة
              </div>
              <div className="flex items-center gap-1">
                <DollarSign size={11} />
                {form.consultationFee} ج.م
              </div>
            </div>
            <div className="mt-3">
              <label className="relative inline-flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={form.isAvailable}
                  onChange={(e) => setForm((p) => ({ ...p, isAvailable: e.target.checked }))}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-gray-200 rounded-full peer peer-checked:bg-green-500 after:content-[''] after:absolute after:top-0.5 after:right-0.5 after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:after:translate-x-[-16px]" />
                <span className="text-xs text-gray-600">{form.isAvailable ? 'متاح الآن' : 'غير متاح'}</span>
              </label>
            </div>
          </Card>
        </motion.div>

        {/* Edit Form */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-2"
        >
          <Card>
            <CardHeader>
              <CardTitle>تعديل البيانات</CardTitle>
            </CardHeader>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">الاسم الكامل</label>
                  <input
                    type="text"
                    value={form.fullName}
                    onChange={(e) => setForm((p) => ({ ...p, fullName: e.target.value }))}
                    className="w-full px-3 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">رقم الهاتف</label>
                  <input
                    type="tel"
                    value={form.phone}
                    onChange={(e) => setForm((p) => ({ ...p, phone: e.target.value }))}
                    className="w-full px-3 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">سنوات الخبرة</label>
                  <input
                    type="number"
                    min={0}
                    value={form.yearsExperience}
                    onChange={(e) => setForm((p) => ({ ...p, yearsExperience: Number(e.target.value) }))}
                    className="w-full px-3 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">رسوم الاستشارة (ج.م)</label>
                  <input
                    type="number"
                    min={0}
                    value={form.consultationFee}
                    onChange={(e) => setForm((p) => ({ ...p, consultationFee: Number(e.target.value) }))}
                    className="w-full px-3 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">نبذة تعريفية</label>
                <textarea
                  rows={4}
                  value={form.bio}
                  onChange={(e) => setForm((p) => ({ ...p, bio: e.target.value }))}
                  placeholder="اكتب نبذة عن تخصصك وخبراتك..."
                  className="w-full px-3 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 resize-none"
                />
              </div>
              <div className="flex justify-end">
                <Button variant="primary" onClick={handleSave} loading={saving} icon={<Save size={15} />}>
                  حفظ التغييرات
                </Button>
              </div>
            </div>
          </Card>
        </motion.div>
      </div>
    </div>
  )
}
