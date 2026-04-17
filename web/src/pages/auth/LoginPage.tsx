import { useState } from 'react'
import { motion } from 'framer-motion'
import { Eye, EyeOff, Mail, Lock, HeartPulse } from 'lucide-react'
import { useAuthStore } from '@/store/authStore'
import { useNavigate } from 'react-router-dom'
import { authApi } from '@/api/authApi'
import { toast } from 'react-hot-toast'

export default function LoginPage() {
  const [isLoading, setIsLoading] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [loginRole, setLoginRole] = useState<'Admin' | 'Doctor'>('Doctor')
  const [errors, setErrors] = useState<{ email?: string; password?: string }>({})

  const setAuth = useAuthStore(state => state.setAuth)
  const navigate = useNavigate()

  const handleQuickLogin = (role: 'Admin' | 'Doctor') => {
    setLoginRole(role)
    if (role === 'Admin') {
      setEmail('hassanmohamed5065@gmail.com')
      setPassword('123456789')
    } else {
      setEmail('doctor@medbook.com')
      setPassword('123456789')
    }
  }

  const login = async (credentials: { email: string; password: string }) => {
    setIsLoading(true)
    // Simulated delay for realistic feel
    await new Promise(resolve => setTimeout(resolve, 800))
    
    try {
      // ── Open Access Bypass (Development Mode) ──────────────────────────
      // This allows entering any credentials and getting the selected role
      const mockName = loginRole === 'Admin' ? 'Hassan Mohamed' : 'Dr. User'
      const mockToken = 'dev-token-' + Math.random().toString(36).substr(2)
      
      setAuth({
        id: 'dev-user-id',
        firstName: mockName.split(' ')[0],
        lastName: mockName.split(' ')[1] || '',
        email: credentials.email || (loginRole === 'Admin' ? 'admin@medbook.com' : 'doctor@medbook.com'),
        role: loginRole as any
      }, mockToken)

      toast.success(`Welcome to Dev Mode, ${mockName}!`)

      if (loginRole === 'Admin') {
        navigate('/admin/dashboard')
      } else {
        navigate('/')
      }
    } catch (err: any) {
      toast.error('Something went wrong')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
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
              className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-tr from-emerald-600 to-teal-500 rounded-3xl shadow-xl shadow-emerald-500/30 mb-6"
            >
              <HeartPulse size={36} className="text-white" />
            </motion.div>
            <h1 className="text-3xl font-extrabold text-slate-800 tracking-tight">MedBook Portal</h1>
            <p className="text-sm text-slate-500 mt-2 font-medium">Please choose your access level</p>
          </div>

          {/* Role Selection Tabs */}
          <div className="flex p-1 bg-slate-200/50 backdrop-blur-md rounded-2xl mb-6">
            <button
              onClick={() => handleQuickLogin('Doctor')}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-sm font-bold rounded-xl transition-all ${
                loginRole === 'Doctor' ? 'bg-white text-emerald-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              <HeartPulse size={16} />
              Doctor
            </button>
            <button
              onClick={() => handleQuickLogin('Admin')}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-sm font-bold rounded-xl transition-all ${
                loginRole === 'Admin' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              <Lock size={16} />
              Admin
            </button>
          </div>

          {/* Card */}
          <div className="bg-white/80 backdrop-blur-xl rounded-3xl shadow-2xl shadow-slate-200/50 border border-white p-8">
            <h2 className="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
              Sign In as <span className={loginRole === 'Admin' ? 'text-indigo-600' : 'text-emerald-600'}>{loginRole}</span>
            </h2>

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
                    placeholder={loginRole === 'Admin' ? 'admin@medbook.com' : 'doctor@medbook.com'}
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
              </div>

              <motion.button
                type="submit"
                disabled={isLoading}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.98 }}
                className={`w-full py-3.5 text-white rounded-2xl font-bold text-sm transition-all shadow-lg mt-4 disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-3 group ${
                  loginRole === 'Admin' ? 'bg-indigo-600 hover:bg-indigo-700 shadow-indigo-500/30' : 'bg-emerald-600 hover:bg-emerald-700 shadow-emerald-500/30'
                }`}
              >
                {isLoading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Signing in...
                  </>
                ) : (
                  <>
                    Sign In as {loginRole}
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
