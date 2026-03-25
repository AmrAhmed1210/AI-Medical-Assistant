import { useState } from 'react'
import { motion } from 'framer-motion'
import { Eye, EyeOff, Mail, Lock, HeartPulse } from 'lucide-react'
import { useAuth } from '@/hooks/useAuth'

export default function LoginPage() {
  const { login, isLoading } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [errors, setErrors] = useState<{ email?: string; password?: string }>({})

  const validate = () => {
    const e: typeof errors = {}
    if (!email) e.email = 'البريد الإلكتروني مطلوب'
    else if (!/\S+@\S+\.\S+/.test(email)) e.email = 'بريد إلكتروني غير صحيح'
    if (!password) e.password = 'كلمة المرور مطلوبة'
    else if (password.length < 6) e.password = 'كلمة المرور يجب أن تكون 6 أحرف على الأقل'
    setErrors(e)
    return Object.keys(e).length === 0
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!validate()) return
    await login({ email, password })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-blue-50 flex items-center justify-center p-4 font-tajawal" dir="rtl">
      <div className="w-full max-w-sm">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        >
          {/* Header */}
          <div className="text-center mb-8">
            <motion.div
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.1, type: 'spring' }}
              className="inline-flex items-center justify-center w-16 h-16 bg-primary-600 rounded-2xl shadow-lg shadow-primary-200 mb-4"
            >
              <HeartPulse size={28} className="text-white" />
            </motion.div>
            <h1 className="text-2xl font-bold text-gray-800">مرحباً بك في MedBook</h1>
            <p className="text-sm text-gray-500 mt-1">منصة الاستشارات الطبية الذكية</p>
          </div>

          {/* Card */}
          <div className="bg-white rounded-2xl shadow-xl shadow-gray-100 border border-gray-100 p-8">
            <h2 className="text-lg font-semibold text-gray-800 mb-6">تسجيل الدخول</h2>

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Email */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1.5">
                  البريد الإلكتروني
                </label>
                <div className="relative">
                  <Mail size={16} className="absolute top-1/2 -translate-y-1/2 right-3 text-gray-400" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => { setEmail(e.target.value); setErrors((p) => ({ ...p, email: undefined })) }}
                    placeholder="example@email.com"
                    className={`w-full pr-10 pl-4 py-2.5 text-sm border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 transition-colors ${
                      errors.email ? 'border-red-300 bg-red-50' : 'border-gray-200 focus:border-primary-400'
                    }`}
                  />
                </div>
                {errors.email && <p className="mt-1 text-xs text-red-500">{errors.email}</p>}
              </div>

              {/* Password */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1.5">
                  كلمة المرور
                </label>
                <div className="relative">
                  <Lock size={16} className="absolute top-1/2 -translate-y-1/2 right-3 text-gray-400" />
                  <input
                    type={showPw ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => { setPassword(e.target.value); setErrors((p) => ({ ...p, password: undefined })) }}
                    placeholder="••••••••"
                    className={`w-full pr-10 pl-10 py-2.5 text-sm border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 transition-colors ${
                      errors.password ? 'border-red-300 bg-red-50' : 'border-gray-200 focus:border-primary-400'
                    }`}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPw(!showPw)}
                    className="absolute top-1/2 -translate-y-1/2 left-3 text-gray-400 hover:text-gray-600"
                  >
                    {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
                {errors.password && <p className="mt-1 text-xs text-red-500">{errors.password}</p>}
              </div>

              <motion.button
                type="submit"
                disabled={isLoading}
                whileTap={{ scale: 0.98 }}
                className="w-full py-2.5 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-semibold text-sm transition-colors shadow-sm shadow-primary-200 mt-2 disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    جاري الدخول...
                  </>
                ) : 'تسجيل الدخول'}
              </motion.button>
            </form>
          </div>

          <p className="text-center text-xs text-gray-400 mt-6">
            للحصول على حساب، تواصل مع مدير النظام
          </p>
        </motion.div>
      </div>
    </div>
  )
}
