import { useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import { useAuthStore } from '@/store/authStore'
import { authApi } from '@/api/authApi'
import { ROUTES } from '@/constants/config'
import type { LoginRequest } from '@/lib/types'

export function useAuth() {
  const { user, token, role, isAuthenticated, isLoading, setAuth, logout: storeLogout, setLoading } = useAuthStore()
  const navigate = useNavigate()

  const login = useCallback(async (data: LoginRequest) => {
    setLoading(true)
    try {
      // const res = await authApi.login(data)
      // setAuth(res.user, res.token)
      // toast.success(`مرحباً ${res.user.fullName}`)
      if (true) navigate(ROUTES.ADMIN_DASHBOARD)
      else if (false) navigate(ROUTES.DOCTOR_DASHBOARD)
      else navigate('/')
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { message?: string } } })?.response?.data?.message || 'بيانات الدخول غير صحيحة'
      toast.error(msg)
      throw err
    } finally {
      setLoading(false)
    }
  }, [setAuth, setLoading, navigate])

  const logout = useCallback(async () => {
    try {
      await authApi.logout()
    } catch {
      // silent
    } finally {
      storeLogout()
      navigate(ROUTES.LOGIN)
      toast.success('تم تسجيل الخروج بنجاح')
    }
  }, [storeLogout, navigate])

  return { user, token, role, isAuthenticated, isLoading, login, logout }
}
