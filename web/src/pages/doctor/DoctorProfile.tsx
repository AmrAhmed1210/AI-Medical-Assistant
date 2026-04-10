import { useState, useRef, useEffect } from 'react'
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
    consultationFee: profile?.consultFee ?? 0,
    isAvailable: true, // Default to available since it's not in the DTO
  })
  const [saving, setSaving] = useState(false)
  const [photoUploading, setPhotoUploading] = useState(false)

  // Sync form with profile
  useEffect(() => {
    if (profile) {
      setForm({
        fullName: profile.fullName,
        bio: profile.bio ?? '',
        yearsExperience: profile.yearsExperience ?? 0,
        consultationFee: profile.consultFee ?? 0,
        isAvailable: true, // Default to available since it's not in the DTO
      })
    }
  }, [profile])

  if (isLoading) return <PageLoader />

  const handleSave = async () => {
    setSaving(true)
    try {
      await updateProfile(form)
      toast.success('Profile saved successfully')
    } catch {
      toast.error('Failed to save data')
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
      toast.success('Photo uploaded successfully')
    } catch {
      toast.error('Failed to upload photo')
    } finally {
      setPhotoUploading(false)
    }
  }

  return (
    <div className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center gap-4 pb-6 border-b border-gray-100"
      >
        <div className="p-3 bg-gradient-to-br from-primary-500 to-primary-600 rounded-2xl shadow-lg shadow-primary-500/25">
          <Camera size={24} className="text-white" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Doctor Profile</h1>
          <p className="text-gray-600 mt-1">Manage your personal and professional profile information</p>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Profile Preview */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="relative overflow-hidden border-0 shadow-xl hover:shadow-2xl transition-all duration-300 bg-gradient-to-br from-white via-gray-50/50 to-white">
            {/* Decorative background elements */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-primary-100/30 to-transparent rounded-full -translate-y-16 translate-x-16" />
            <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-blue-100/20 to-transparent rounded-full translate-y-12 -translate-x-12" />

            <div className="relative text-center p-8">
              <div className="relative inline-block mb-6">
                <div className="w-28 h-28 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 p-1 shadow-2xl shadow-primary-500/25">
                  <div className="w-full h-full rounded-full bg-white flex items-center justify-center overflow-hidden">
                    {profile?.photoUrl ? (
                      <img src={profile.photoUrl} alt={profile.fullName} className="w-full h-full object-cover" />
                    ) : (
                      <span className="text-primary-600 text-4xl font-bold">{form.fullName.charAt(0)}</span>
                    )}
                  </div>
                </div>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => fileRef.current?.click()}
                  disabled={photoUploading}
                  className="absolute -bottom-1 -right-1 w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-600 rounded-full flex items-center justify-center text-white shadow-lg hover:shadow-xl transition-all duration-200 border-2 border-white"
                >
                  <Camera size={14} />
                </motion.button>
                <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handlePhotoChange} />
              </div>

              <div className="space-y-3">
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-1">{form.fullName || 'Doctor Name'}</h3>
                  <p className="text-primary-600 font-medium text-sm">{profile?.specialty || 'General Practice'}</p>
                </div>

                <div className="flex items-center justify-center gap-1">
                  <div className="flex items-center gap-1 bg-amber-50 px-3 py-1.5 rounded-full border border-amber-200">
                    <Star size={14} className="text-amber-500 fill-amber-500" />
                    <span className="text-sm font-semibold text-amber-700">{((profile as any)?.rating ?? 4.5).toFixed(1)}</span>
                  </div>
                </div>

                <div className="flex items-center justify-center gap-6 pt-2">
                  <div className="flex items-center gap-2 text-gray-600">
                    <div className="p-1.5 bg-gray-100 rounded-lg">
                      <Clock size={12} />
                    </div>
                    <span className="text-xs font-medium">{form.yearsExperience} Years</span>
                  </div>
                  <div className="flex items-center gap-2 text-gray-600">
                    <div className="p-1.5 bg-gray-100 rounded-lg">
                      <DollarSign size={12} />
                    </div>
                    <span className="text-xs font-medium">${form.consultationFee}</span>
                  </div>
                </div>

                <div className="pt-4 border-t border-gray-100">
                  <label className="relative inline-flex items-center gap-3 cursor-pointer group">
                    <input
                      type="checkbox"
                      checked={form.isAvailable}
                      onChange={(e) => setForm((p) => ({ ...p, isAvailable: e.target.checked }))}
                      className="sr-only peer"
                    />
                    <div className="w-12 h-6 bg-gray-200 rounded-full peer peer-checked:bg-gradient-to-r peer-checked:from-green-400 peer-checked:to-green-500 transition-all duration-300 shadow-inner peer-checked:shadow-green-500/25 after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all after:shadow-sm peer-checked:after:translate-x-[24px] peer-checked:after:shadow-green-500/50" />
                    <span className={`text-sm font-medium transition-colors ${form.isAvailable ? 'text-green-600' : 'text-gray-500'}`}>
                      {form.isAvailable ? 'Available Now' : 'Not Available'}
                    </span>
                  </label>
                </div>
              </div>
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
          <Card className="border-0 shadow-xl hover:shadow-2xl transition-all duration-300 bg-gradient-to-br from-white via-gray-50/30 to-white">
            <CardHeader className="pb-6">
              <CardTitle className="text-xl font-bold text-gray-900 flex items-center gap-3">
                <div className="p-2 bg-primary-50 rounded-xl">
                  <Save size={20} className="text-primary-600" />
                </div>
                Edit Profile
              </CardTitle>
            </CardHeader>
            <div className="px-6 pb-6">
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Full Name</label>
                    <input
                      type="text"
                      value={form.fullName}
                      onChange={(e) => setForm((p) => ({ ...p, fullName: e.target.value }))}
                      className="w-full px-4 py-3 text-sm border-2 border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 focus:border-primary-400 transition-all duration-200 bg-white shadow-sm hover:border-gray-300"
                      placeholder="Enter your full name"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Years of Experience</label>
                    <input
                      type="number"
                      min={0}
                      value={form.yearsExperience}
                      onChange={(e) => setForm((p) => ({ ...p, yearsExperience: Number(e.target.value) }))}
                      className="w-full px-4 py-3 text-sm border-2 border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 focus:border-primary-400 transition-all duration-200 bg-white shadow-sm hover:border-gray-300"
                      placeholder="0"
                    />
                  </div>
                  <div className="space-y-2 md:col-span-2">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Consultation Fee ($)</label>
                    <input
                      type="number"
                      min={0}
                      value={form.consultationFee}
                      onChange={(e) => setForm((p) => ({ ...p, consultationFee: Number(e.target.value) }))}
                      className="w-full px-4 py-3 text-sm border-2 border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 focus:border-primary-400 transition-all duration-200 bg-white shadow-sm hover:border-gray-300"
                      placeholder="0"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Doctor Biography</label>
                  <textarea
                    rows={5}
                    value={form.bio}
                    onChange={(e) => setForm((p) => ({ ...p, bio: e.target.value }))}
                    placeholder="Write a brief description about your specialization, experience, and approach to patient care..."
                    className="w-full px-4 py-3 text-sm border-2 border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 focus:border-primary-400 transition-all duration-200 bg-white shadow-sm hover:border-gray-300 resize-none"
                  />
                </div>

                <div className="flex justify-end pt-4 border-t border-gray-100">
                  <Button
                    variant="primary"
                    onClick={handleSave}
                    loading={saving}
                    icon={<Save size={16} />}
                    className="px-8 py-3 shadow-lg hover:shadow-xl transition-all duration-200"
                  >
                    Save Changes
                  </Button>
                </div>
              </div>
            </div>
          </Card>
        </motion.div>
      </div>
    </div>
  )
}
