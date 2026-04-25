import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Eye, EyeOff, Mail, Lock, HeartPulse } from 'lucide-react'
import { useAuthStore } from '@/store/authStore'
import { useNavigate } from 'react-router-dom'
import { authApi } from '@/api/authApi'
import toast from 'react-hot-toast'

export default function LoginPage() {
  const [isLoading, setIsLoading] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [error, setError] = useState('')

  const logout = useAuthStore(state => state.logout)
  const setAuth = useAuthStore(state => state.setAuth)
  const navigate = useNavigate()

  // Force logout on mount to clear stale state
  useEffect(() => {
    logout()
  }, [logout])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsLoading(true)

    try {
      const response = await authApi.login({ email, password })

      const user = {
        id: parseInt(response.user.id, 10),
        name: response.user.fullName,
        email: response.user.email,
        role: response.user.role,
        isActive: true,
      }

      setAuth(user, response.accessToken)

      const role = response.user.role
      if (role === 'Admin') {
        navigate('/admin/dashboard')
      } else if (role === 'Doctor') {
        navigate('/doctor/dashboard')
      } else {
        setError('Access denied. Please use the mobile application.')
        useAuthStore.getState().logout?.()
      }
    } catch (err: any) {
      const msg = err?.response?.data?.message || 'Invalid email or password. Please try again.'
      setError(msg)
      toast.error(msg)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 flex items-center justify-center p-4 relative overflow-hidden" dir="ltr">
      {/* Animated background orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          animate={{ x: [0, 30, 0], y: [0, -20, 0] }}
          transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
          className="absolute top-1/4 -left-20 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"
        />
        <motion.div
          animate={{ x: [0, -25, 0], y: [0, 25, 0] }}
          transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
          className="absolute bottom-1/4 -right-20 w-80 h-80 bg-emerald-500/10 rounded-full blur-3xl"
        />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 24, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="relative w-full max-w-md"
      >
        {/* Logo */}
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.1, type: 'spring', stiffness: 200 }}
            className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-tr from-emerald-500 to-teal-400 rounded-2xl shadow-2xl shadow-emerald-500/30 mb-4"
          >
            <HeartPulse size={32} className="text-white" />
          </motion.div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">MedBook Portal</h1>
          <p className="text-blue-300/70 mt-1 text-sm">Secure staff access</p>
        </div>

        {/* Card */}
        <div className="bg-white/5 backdrop-blur-2xl border border-white/10 rounded-3xl p-8 shadow-2xl">
          <h2 className="text-lg font-semibold text-white mb-6">Sign in to your account</h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-blue-200 mb-1.5">Email address</label>
              <div className="relative">
                <Mail size={16} className="absolute top-1/2 -translate-y-1/2 left-3.5 text-blue-400/60" />
                <input
                  id="login-email"
                  type="email"
                  required
                  autoComplete="off"
                  value={email}
                  onChange={e => { setEmail(e.target.value); setError('') }}
                  placeholder="you@example.com"
                  className="w-full pl-10 pr-4 py-3 text-sm bg-white border border-gray-200 rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 shadow-sm transition-all"
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-blue-200 mb-1.5">Password</label>
              <div className="relative">
                <Lock size={16} className="absolute top-1/2 -translate-y-1/2 left-3.5 text-blue-400/60" />
                <input
                  id="login-password"
                  type={showPw ? 'text' : 'password'}
                  required
                  autoComplete="new-password"
                  value={password}
                  onChange={e => { setPassword(e.target.value); setError('') }}
                  placeholder="••••••••"
                  className="w-full pl-10 pr-11 py-3 text-sm bg-white border border-gray-200 rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 shadow-sm transition-all"
                />
                <button
                  type="button"
                  onClick={() => setShowPw(!showPw)}
                  className="absolute top-1/2 -translate-y-1/2 right-3.5 text-blue-400/50 hover:text-blue-300 transition-colors"
                >
                  {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-2.5"
              >
                {error}
              </motion.div>
            )}

            {/* Submit */}
            <motion.button
              type="submit"
              disabled={isLoading}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.98 }}
              className="w-full py-3.5 mt-2 bg-gradient-to-r from-emerald-500 to-teal-500 text-white font-bold rounded-xl text-sm shadow-lg shadow-emerald-500/25 hover:shadow-emerald-500/40 disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Signing in...
                </>
              ) : 'Sign In'}
            </motion.button>
          </form>
        </div>

        {/* Apply link */}
        <p className="text-center text-sm text-blue-300/60 mt-6">
          Are you a doctor?{' '}
          <button
            onClick={() => navigate('/apply')}
            className="text-emerald-400 hover:text-emerald-300 font-semibold hover:underline transition-colors"
          >
            Apply to join our platform
          </button>
        </p>
      </motion.div>
    </div>
  )
}
