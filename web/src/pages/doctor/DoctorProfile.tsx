import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Camera, Save, Star, Clock, DollarSign, MessageSquare, ThumbsUp } from 'lucide-react'
import { useDoctorProfile } from '@/hooks/useDoctor'
import { useDoctorStore } from '@/store/doctorStore'
import { useLanguage } from '@/lib/language'
import { doctorApi } from '@/api/doctorApi'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import type { ReviewDto } from '@/lib/types'
import toast from 'react-hot-toast'

export default function DoctorProfile() {
  const { profile, isLoading, refresh } = useDoctorProfile()
  const { t, isRTL } = useLanguage()
  const { updateProfile } = useDoctorStore()
  const fileRef = useRef<HTMLInputElement>(null)

  const [form, setForm] = useState({
    fullName: profile?.fullName ?? '',
    bio: profile?.bio ?? '',
    yearsExperience: profile?.yearsExperience ?? 0,
    consultationFee: profile?.consultFee ?? 0,
    isAvailable: profile?.isAvailable ?? false,
  })
  const [saving, setSaving] = useState(false)
  const [photoUploading, setPhotoUploading] = useState(false)
  const [reviews, setReviews] = useState<ReviewDto[]>([])
  const [reviewsLoading, setReviewsLoading] = useState(false)

  // Sync form with profile
  useEffect(() => {
    if (profile) {
      setForm({
        fullName: profile.fullName,
        bio: profile.bio ?? '',
        yearsExperience: profile.yearsExperience ?? 0,
        consultationFee: profile.consultFee ?? 0,
        isAvailable: profile.isAvailable ?? false,
      })
    }
  }, [profile])

  // Fetch reviews
  useEffect(() => {
    const fetchReviews = async () => {
      setReviewsLoading(true)
      try {
        const data = await doctorApi.getReviews()
        setReviews(data)
      } catch {
        // silently fail
      } finally {
        setReviewsLoading(false)
      }
    }
    fetchReviews()
  }, [])

  if (isLoading) return <PageLoader />

  const handleSave = async () => {
    setSaving(true)
    try {
      await updateProfile(form)
      toast.success(t('profileSaved'))
    } catch {
      toast.error(t('errSaveProfile'))
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
      await refresh()
      toast.success(t('photoUploaded'))
    } catch {
      toast.error(t('errUploadPhoto'))
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
          <h1 className="text-xl font-bold text-gray-900">{t('doctorProfile')}</h1>
          <p className="text-sm text-gray-500 mt-0.5">{t('manageProfileDesc')}</p>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Profile Preview */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="relative overflow-hidden border-0 shadow-xl hover:shadow-2xl transition-all duration-300 bg-gradient-to-br from-white via-gray-50/50 to-white">
            {/* Decorative background elements */}
            <div className="absolute top-0 w-32 h-32 bg-gradient-to-br from-primary-100/30 to-transparent rounded-full -translate-y-16 rtl:-translate-x-16 ltr:translate-x-16" />
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
                  className={`absolute -bottom-1 ${isRTL ? '-left-1' : '-right-1'} w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-600 rounded-full flex items-center justify-center text-white shadow-lg hover:shadow-xl transition-all duration-200 border-2 border-white`}
                >
                  <Camera size={14} />
                </motion.button>
                <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handlePhotoChange} />
              </div>

              <div className="space-y-3">
                <div>
                  <h3 className="text-lg font-bold text-gray-900 mb-0.5">{form.fullName || t('doctorName')}</h3>
                  <p className="text-primary-600 font-medium text-xs">{profile?.specialty || t('generalPractice')}</p>
                </div>

                <div className="flex items-center justify-center gap-1">
                  <div className="flex items-center gap-1 bg-amber-50 px-3 py-1.5 rounded-full border border-amber-200">
                    <Star size={14} className="text-amber-500 fill-amber-500" />
                    <span className="text-sm font-semibold text-amber-700">
                      {reviews.length > 0 
                        ? (reviews.reduce((acc, r) => acc + r.rating, 0) / reviews.length).toFixed(1)
                        : ((profile as any)?.rating ?? 4.5).toFixed(1)}
                    </span>
                    <span className="text-xs text-amber-600">({reviews.length} {t('reviewsWord')})</span>
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
                      {form.isAvailable ? t('availableNow') : t('notAvailable')}
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
              <CardTitle className="text-lg font-bold text-gray-900 flex items-center gap-3">
                <div className="p-2 bg-primary-50 rounded-xl">
                  <Save size={20} className="text-primary-600" />
                </div>
                {t('editProfile')}
              </CardTitle>
            </CardHeader>
            <div className="px-6 pb-6">
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">{t('fullName')}</label>
                    <input
                      type="text"
                      value={form.fullName}
                      onChange={(e) => setForm((p) => ({ ...p, fullName: e.target.value }))}
                      className="w-full px-4 py-3 text-sm border-2 border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 focus:border-primary-400 transition-all duration-200 bg-white shadow-sm hover:border-gray-300"
                      placeholder={t("enterFullName")}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">{t('yearsExp')}</label>
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
                    <label className="block text-sm font-semibold text-gray-700 mb-2">{t('consultFee')}</label>
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
                  <label className="block text-sm font-semibold text-gray-700 mb-2">{t('doctorBio')}</label>
                  <textarea
                    rows={5}
                    value={form.bio}
                    onChange={(e) => setForm((p) => ({ ...p, bio: e.target.value }))}
                    placeholder={t("bioPlaceholder")}
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

      {/* Reviews Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mt-8"
      >
        <Card className="border-0 shadow-xl bg-gradient-to-br from-white via-gray-50/30 to-white">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg font-bold text-gray-900 flex items-center gap-3">
              <div className="p-2 bg-amber-50 rounded-xl">
                <MessageSquare size={20} className="text-amber-600" />
              </div>
              {t('patientReviews')}
              {reviews.length > 0 && (
                <span className="text-sm font-normal text-gray-500 ml-2">
                  ({reviews.length} {t('totalWord')})
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <div className="px-6 pb-6">
            {reviewsLoading ? (
              <div className="flex justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
              </div>
            ) : reviews.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <ThumbsUp size={48} className="mx-auto mb-3 text-gray-300" />
                <p>{t('noReviewsYet')}</p>
              </div>
            ) : (
              <div className="space-y-4">
                {reviews.map((review) => (
                  <motion.div
                    key={review.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="p-4 bg-white rounded-xl border border-gray-100 shadow-sm"
                  >
                    <div className="flex items-start gap-3">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-100 to-primary-200 flex items-center justify-center text-primary-700 font-semibold">
                        {review.patientName?.charAt(0) ?? 'P'}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-semibold text-gray-900">
                            {review.patientName ?? t('anonymous')}
                          </span>
                          <div className="flex items-center gap-0.5">
                            {Array.from({ length: 5 }).map((_, i) => (
                              <Star
                                key={i}
                                size={12}
                                className={i < review.rating ? 'text-amber-500 fill-amber-500' : 'text-gray-300'}
                              />
                            ))}
                          </div>
                        </div>
                        <p className="text-gray-700 text-sm">{review.comment}</p>
                        <p className="text-gray-400 text-xs mt-2">
                          {new Date(review.createdAt).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </Card>
      </motion.div>
    </div>
  )
}
