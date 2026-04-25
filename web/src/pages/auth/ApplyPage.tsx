import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import axiosInstance from '@/api/axiosInstance'
import { toast } from 'react-hot-toast'
import {
  User, Mail, Phone, Briefcase, FileText, Upload, ChevronDown,
  CheckCircle2, HeartPulse, X, AlertCircle
} from 'lucide-react'

interface Specialty { id: number; name: string }

const MAX_FILE_SIZE = 5 * 1024 * 1024 // 5 MB
const ALLOWED_TYPES = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg']

export default function ApplyPage() {
  const [step, setStep] = useState<'form' | 'success'>('form')
  const [isLoading, setIsLoading] = useState(false)
  const [specialties, setSpecialties] = useState<Specialty[]>([])
  const [fileError, setFileError] = useState('')
  const [formError, setFormError] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  const [form, setForm] = useState({
    name: '',
    email: '',
    phone: '',
    specialtyId: 0,
    experience: '',
    licenseNumber: '',
    bio: '',
    message: '',
  })
  const [file, setFile] = useState<File | null>(null)
  const [photoFile, setPhotoFile] = useState<File | null>(null)
  const [photoPreview, setPhotoPreview] = useState<string | null>(null)

  // Fetch specialties
  useEffect(() => {
    axiosInstance.get<Specialty[]>('/api/specialties')
      .then(r => setSpecialties(r.data))
      .catch(() => {/* silently ignore if endpoint not available */})
  }, [])

  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setForm(prev => ({ ...prev, [k]: e.target.value }))
    setFormError('')
  }

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    setFileError('')
    if (!f) return
    if (!ALLOWED_TYPES.includes(f.type)) {
      setFileError('Only PDF, JPG, or PNG files are allowed.')
      return
    }
    if (f.size > MAX_FILE_SIZE) {
      setFileError('File must be less than 5 MB.')
      return
    }
    setFile(f)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setFormError('')

    if (!form.specialtyId || form.specialtyId === 0) {
      setFormError('Please select a specialty.')
      return
    }
    if (!file) {
      setFormError('Please upload your license/credential document.')
      return
    }
    if (!form.licenseNumber.trim()) {
      setFormError('License / National ID is required.')
      return
    }

    setIsLoading(true)
    try {
      // 1. Upload CV to Cloudinary
      const formData = new FormData()
      formData.append('file', file)
      
      const uploadRes = await axiosInstance.post('/api/doctors/apply/upload-cv', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      
      const cvUrl = uploadRes.data.url

      // 2. Upload Photo to Cloudinary (if provided)
      let photoUrl = ''
      if (photoFile) {
        const photoFormData = new FormData()
        photoFormData.append('file', photoFile)
        const photoRes = await axiosInstance.post('/api/doctors/apply/upload-cv', photoFormData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        photoUrl = photoRes.data.url
      }

      // 3. Submit application with the URLs
      await axiosInstance.post('/api/doctors/apply', {
        name: form.name.trim(),
        email: form.email.trim(),
        phone: form.phone.trim(),
        specialtyId: Number(form.specialtyId),
        experience: Number(form.experience) || 0,
        licenseNumber: form.licenseNumber.trim(),
        bio: form.bio.trim(),
        message: form.message.trim(),
        documentUrl: cvUrl,
        photoUrl: photoUrl
      })
      setStep('success')
    } catch (err: any) {
      const msg = err?.response?.data?.message || 'Failed to submit application. Please try again.'
      setFormError(msg)
    } finally {
      setIsLoading(false)
    }
  }

  if (step === 'success') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 flex items-center justify-center p-4" dir="ltr">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: 'spring', stiffness: 200, damping: 20 }}
          className="bg-white/5 backdrop-blur-2xl border border-white/10 rounded-3xl p-10 text-center max-w-md w-full shadow-2xl"
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
            className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-emerald-500 to-teal-400 rounded-full mb-6 shadow-xl shadow-emerald-500/30"
          >
            <CheckCircle2 size={40} className="text-white" />
          </motion.div>
          <h1 className="text-2xl font-bold text-white mb-3">Application Received!</h1>
          <p className="text-blue-200/80 text-sm leading-relaxed mb-8">
            Your application has been received. We will contact you soon at <span className="text-emerald-400 font-semibold">{form.email}</span> after reviewing your documents.
          </p>
          <button
            onClick={() => navigate('/login')}
            className="w-full py-3 bg-gradient-to-r from-emerald-500 to-teal-500 text-white font-bold rounded-xl text-sm hover:opacity-90 transition-opacity"
          >
            Back to Login
          </button>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 py-10 px-4" dir="ltr">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <div className="inline-flex items-center justify-center w-14 h-14 bg-gradient-to-tr from-emerald-500 to-teal-400 rounded-2xl shadow-xl shadow-emerald-500/30 mb-4">
            <HeartPulse size={28} className="text-white" />
          </div>
          <h1 className="text-3xl font-extrabold text-white">Apply as a Doctor</h1>
          <p className="text-blue-300/60 mt-1.5 text-sm">Join our network of healthcare professionals</p>
        </motion.div>

        <motion.form
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          onSubmit={handleSubmit}
          className="bg-white/5 backdrop-blur-2xl border border-white/10 rounded-3xl p-8 shadow-2xl space-y-5"
        >
          {/* Name + Email */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Field label="Full Name" required>
              <InputWithIcon icon={<User size={15} />} type="text" required placeholder="Dr. Jane Smith" value={form.name} onChange={set('name')} />
            </Field>
            <Field label="Email Address" required>
              <InputWithIcon icon={<Mail size={15} />} type="email" required placeholder="you@example.com" value={form.email} onChange={set('email')} />
            </Field>
          </div>

          {/* Phone + Experience */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Field label="Phone Number" required>
              <InputWithIcon icon={<Phone size={15} />} type="tel" required placeholder="+1 234 567 8900" value={form.phone} onChange={set('phone')} />
            </Field>
            <Field label="Years of Experience" required>
              <InputWithIcon icon={<Briefcase size={15} />} type="number" min="0" required placeholder="5" value={form.experience} onChange={set('experience')} />
            </Field>
          </div>

          {/* Specialty */}
          <Field label="Specialty" required>
            <div className="relative">
              <select
                required
                value={form.specialtyId}
                onChange={e => { setForm(p => ({ ...p, specialtyId: Number(e.target.value) })); setFormError('') }}
                className="w-full px-4 py-3 text-sm bg-white border border-gray-200 rounded-xl text-gray-900 appearance-none focus:outline-none focus:ring-2 focus:ring-emerald-500/50 shadow-sm transition-all"
              >
                <option value={0}>-- Select Specialty --</option>
                {specialties.length > 0
                  ? specialties.map(s => (
                    <option key={s.id} value={s.id}>{s.name}</option>
                  ))
                  : [
                    { id: 1, name: 'General Practice' },
                    { id: 2, name: 'Cardiology' },
                    { id: 3, name: 'Dermatology' },
                    { id: 4, name: 'Neurology' },
                    { id: 5, name: 'Orthopedics' },
                    { id: 6, name: 'Pediatrics' },
                    { id: 7, name: 'Psychiatry' },
                    { id: 8, name: 'Surgery' },
                  ].map(s => (
                    <option key={s.id} value={s.id}>{s.name}</option>
                  ))
                }
              </select>
              <ChevronDown size={15} className="absolute right-3.5 top-1/2 -translate-y-1/2 text-blue-400/50 pointer-events-none" />
            </div>
          </Field>

          {/* License Number */}
          <Field label="License / National ID" required>
            <InputWithIcon icon={<FileText size={15} />} type="text" required placeholder="LIC-123456789" value={form.licenseNumber} onChange={set('licenseNumber')} />
          </Field>

          {/* Bio */}
          <Field label="Professional Bio" required>
            <textarea
              required
              rows={3}
              placeholder="Brief description of your experience and qualifications..."
              value={form.bio}
              onChange={set('bio')}
              className="w-full px-4 py-3 text-sm bg-white border border-gray-200 rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 shadow-sm transition-all resize-none"
            />
          </Field>

          {/* Message to Admin */}
          <Field label="Message to Admin (optional)">
            <textarea
              rows={2}
              placeholder="Any additional information you'd like to share..."
              value={form.message}
              onChange={set('message')}
              className="w-full px-4 py-3 text-sm bg-white border border-gray-200 rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 shadow-sm transition-all resize-none"
            />
          </Field>

          {/* Profile Photo */}
          <Field label="Profile Photo (Professional Headshot)">
            <div className="flex items-center gap-4">
              <div 
                onClick={() => document.getElementById('photo-upload')?.click()}
                className="w-20 h-20 rounded-2xl border-2 border-dashed border-white/20 bg-white/5 flex items-center justify-center cursor-pointer overflow-hidden hover:border-emerald-500/40 transition-all"
              >
                {photoPreview ? (
                  <img src={photoPreview} alt="Preview" className="w-full h-full object-cover" />
                ) : (
                  <User size={24} className="text-white/20" />
                )}
                <input 
                  id="photo-upload" 
                  type="file" 
                  accept="image/*" 
                  className="hidden" 
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) {
                      setPhotoFile(f);
                      const reader = new FileReader();
                      reader.onloadend = () => setPhotoPreview(reader.result as string);
                      reader.readAsDataURL(f);
                    }
                  }} 
                />
              </div>
              <div className="flex-1">
                <p className="text-sm text-blue-200/80">Upload a professional photo for your profile.</p>
                <p className="text-xs text-blue-300/40 mt-1">Recommended: Square image, max 2MB</p>
              </div>
            </div>
          </Field>

          {/* Document Upload */}
          <Field label="Documents (PDF, JPG, PNG — max 5 MB)" required>
            <div
              onClick={() => fileRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                fileError ? 'border-red-500/50 bg-red-500/5' :
                file ? 'border-emerald-500/50 bg-emerald-500/5' :
                'border-white/15 hover:border-emerald-500/40 hover:bg-white/5'
              }`}
            >
              <input ref={fileRef} type="file" accept=".pdf,.jpg,.jpeg,.png" onChange={handleFile} className="hidden" />
              {file ? (
                <div className="flex items-center justify-center gap-3">
                  <FileText size={20} className="text-emerald-400" />
                  <span className="text-sm text-emerald-300 font-medium">{file.name}</span>
                  <button type="button" onClick={e => { e.stopPropagation(); setFile(null); setFileError('') }}>
                    <X size={16} className="text-red-400 hover:text-red-300" />
                  </button>
                </div>
              ) : (
                <>
                  <Upload size={24} className="text-blue-400/50 mx-auto mb-2" />
                  <p className="text-sm text-blue-300/60">Click to upload or drag and drop</p>
                  <p className="text-xs text-blue-300/40 mt-1">PDF, JPG, PNG up to 5 MB</p>
                </>
              )}
            </div>
            {fileError && (
              <p className="text-xs text-red-400 mt-1 flex items-center gap-1"><AlertCircle size={12} />{fileError}</p>
            )}
          </Field>

          {/* Form error */}
          {formError && (
            <motion.div
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3 flex items-center gap-2"
            >
              <AlertCircle size={16} /> {formError}
            </motion.div>
          )}

          {/* Submit */}
          <motion.button
            type="submit"
            disabled={isLoading}
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.98 }}
            className="w-full py-3.5 bg-gradient-to-r from-emerald-500 to-teal-500 text-white font-bold rounded-xl text-sm shadow-lg shadow-emerald-500/25 hover:shadow-emerald-500/40 disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
          >
            {isLoading ? (
              <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Submitting...</>
            ) : (
              <><Upload size={16} /> Submit Application</>
            )}
          </motion.button>

          <button
            type="button"
            onClick={() => navigate('/login')}
            className="w-full py-2 text-blue-300/60 text-sm hover:text-blue-200 transition-colors"
          >
            ← Back to Login
          </button>
        </motion.form>
      </div>
    </div>
  )
}

function Field({ label, required, children }: { label: string; required?: boolean; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-sm font-medium text-blue-200 mb-1.5">
        {label}{required && <span className="text-red-400 ml-0.5">*</span>}
      </label>
      {children}
    </div>
  )
}

function InputWithIcon({ icon, ...props }: { icon: React.ReactNode } & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div className="relative">
      <span className="absolute top-1/2 -translate-y-1/2 left-3.5 text-blue-400/50">{icon}</span>
      <input
        {...props}
        className="w-full pl-10 pr-4 py-3 text-sm bg-white border border-gray-200 rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 shadow-sm transition-all"
      />
    </div>
  )
}
