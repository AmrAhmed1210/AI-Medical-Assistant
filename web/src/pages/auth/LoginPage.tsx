import { useState } from 'react'
import { motion } from 'framer-motion'
import { Eye, EyeOff, Mail, Lock, HeartPulse } from 'lucide-react'
import { useAuthStore } from '@/store/authStore'
import { useNavigate } from 'react-router-dom'

export default function LoginPage() {
  const [isLoading, setIsLoading] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [errors, setErrors] = useState<{ email?: string; password?: string }>({})

  const setAuth = useAuthStore(state => state.setAuth)
  const navigate = useNavigate()

  const login = async (_: { email: string; password: string }) => {
    setIsLoading(true)
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setIsLoading(false)
    setAuth({
      id: 'test-id',
      firstName: 'Doctor',
      lastName: 'User',
      email: email || 'doctor@medbook.com',
      role: 'Doctor'
    } as any, 'fake-auth-token')
    navigate('/')
  }

  const validate = () => {
    setErrors({})
    return true
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!validate()) return
    await login({ email, password })
  }

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4 relative overflow-hidden" dir="ltr">
      {/* Decorative Background Elements */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0 pointer-events-none">
        <div className="absolute -top-[10%] -right-[10%] w-[50%] h-[50%] rounded-full bg-gradient-to-br from-primary-400/20 to-blue-500/20 blur-3xl" />
        <div className="absolute -bottom-[10%] -left-[10%] w-[50%] h-[50%] rounded-full bg-gradient-to-tr from-purple-400/20 to-primary-500/20 blur-3xl" />
      </div>

      <div className="w-full max-w-md z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        >
          {/* Header */}
          <div className="text-center mb-10">
            <motion.div
              initial={{ scale: 0.5, rotate: -15, opacity: 0 }}
              animate={{ scale: 1, rotate: 0, opacity: 1 }}
              transition={{ delay: 0.2, type: 'spring', stiffness: 200, damping: 15 }}
              className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-tr from-primary-600 to-primary-400 rounded-3xl shadow-xl shadow-primary-500/30 mb-6"
            >
              <HeartPulse size={36} className="text-white" />
            </motion.div>
            <h1 className="text-3xl font-extrabold text-slate-800 tracking-tight">Welcome to MedBook</h1>
            <p className="text-sm text-slate-500 mt-2 font-medium">Smart Medical Consultation Platform</p>
          </div>

          {/* Card */}
          <div className="bg-white/80 backdrop-blur-xl rounded-3xl shadow-2xl shadow-slate-200/50 border border-white p-8">
            <h2 className="text-xl font-bold text-slate-800 mb-6">Sign In</h2>

            <form onSubmit={handleSubmit} className="space-y-5">
              {/* Email */}
              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                  Email Address
                </label>
                <div className="relative group">
                  <Mail size={18} className="absolute top-1/2 -translate-y-1/2 left-4 text-slate-400 group-focus-within:text-primary-500 transition-colors" />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => { setEmail(e.target.value); setErrors((p) => ({ ...p, email: undefined })) }}
                    placeholder="doctor@medbook.com"
                    className={`w-full pl-11 pr-4 py-3 text-sm bg-slate-50 border rounded-2xl focus:outline-none focus:ring-4 focus:ring-primary-500/20 focus:bg-white transition-all ${
                      errors.email ? 'border-red-300 bg-red-50 focus:border-red-400 focus:ring-red-500/20' : 'border-slate-200 focus:border-primary-500 hover:border-slate-300'
                    }`}
                  />
                </div>
                {errors.email && (
                  <motion.p initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }} className="mt-2 text-xs font-medium text-red-500">
                    {errors.email}
                  </motion.p>
                )}
              </div>

              {/* Password */}
              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                  Password
                </label>
                <div className="relative group">
                  <Lock size={18} className="absolute top-1/2 -translate-y-1/2 left-4 text-slate-400 group-focus-within:text-primary-500 transition-colors" />
                  <input
                    type={showPw ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => { setPassword(e.target.value); setErrors((p) => ({ ...p, password: undefined })) }}
                    placeholder="••••••••"
                    className={`w-full pl-11 pr-12 py-3 text-sm bg-slate-50 border rounded-2xl focus:outline-none focus:ring-4 focus:ring-primary-500/20 focus:bg-white transition-all ${
                      errors.password ? 'border-red-300 bg-red-50 focus:border-red-400 focus:ring-red-500/20' : 'border-slate-200 focus:border-primary-500 hover:border-slate-300'
                    }`}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPw(!showPw)}
                    className="absolute top-1/2 -translate-y-1/2 right-4 text-slate-400 hover:text-slate-600 transition-colors p-1"
                  >
                    {showPw ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
                {errors.password && (
                  <motion.p initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }} className="mt-2 text-xs font-medium text-red-500">
                    {errors.password}
                  </motion.p>
                )}
              </div>

              <motion.button
                type="submit"
                disabled={isLoading}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.98 }}
                className="w-full py-3.5 bg-primary-600 hover:bg-primary-700 text-white rounded-2xl font-bold text-sm transition-all shadow-lg shadow-primary-500/30 mt-4 disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-3 group"
              >
                {isLoading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Signing in...
                  </>
                ) : (
                  <>
                    Sign In
                    <motion.div
                      initial={{ x: -5, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      transition={{ delay: 0.1 }}
                      className="group-hover:translate-x-1 transition-transform"
                    >
                      &rarr;
                    </motion.div>
                  </>
                )}
              </motion.button>
            </form>
          </div>

          <p className="text-center text-sm font-medium text-slate-400 mt-8">
            To get an account, please contact the system administrator
          </p>
        </motion.div>
      </div>
    </div>
  )
}
