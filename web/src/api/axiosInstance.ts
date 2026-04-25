import axios from 'axios'
import { API_BASE_URL, TOKEN_KEY } from '@/constants/config'
import { useAuthStore } from '@/store/authStore'

const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Accept-Language': 'ar',
  },
  timeout: 30000,
})

// Request interceptor - add JWT token
axiosInstance.interceptors.request.use(
  (config) => {
    const token = useAuthStore.getState().token
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor - handle 401
axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    const isLoginRequest = error.config?.url?.includes('/api/auth/login')
    const isAlreadyOnLoginPage = window.location.pathname === '/login'

    if (error.response?.status === 401 && !isLoginRequest && !isAlreadyOnLoginPage) {
      useAuthStore.getState().logout()
      // Full reload to clear any stale state and redirect to login
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default axiosInstance
