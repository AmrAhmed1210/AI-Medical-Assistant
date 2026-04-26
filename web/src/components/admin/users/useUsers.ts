import { useState, useCallback, useMemo, useEffect } from 'react'
import { useAuthStore } from '@/store/authStore'
import { adminApi } from '@/api/adminApi'
import type { UserDto, UserRole } from '@/lib/types'
import toast from 'react-hot-toast'
import { debounce } from '@/lib/utils/debounce'
import { MOCK_USERS, PAGE_SIZE } from './constants'
import { startConnection } from '@/lib/signalr'

interface UseUsersOptions {
  initialSearch?: string
  initialRole?: UserRole | ''
}

export function useUsers(
  { initialSearch = '', initialRole = '' }: UseUsersOptions = {}
) {

  const [users, setUsers] = useState<UserDto[]>([])
  const [total, setTotal] = useState(0)

  const [page, setPage] = useState(1)
  const [search, setSearch] = useState(initialSearch)
  const [roleFilter, setRoleFilter] = useState<UserRole | ''>(initialRole)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [connectionStatus, setConnectionStatus] =
    useState<'checking' | 'connected' | 'error'>('checking')
  const [newUserNotification, setNewUserNotification] = useState<{
    userId: number
    name: string
    email: string
    role: string
    message: string
    timestamp: string
  } | null>(null)

  /* ---------------- fetch users ---------------- */

  const fetchUsers = useCallback(async (
    p = page,
    q = search,
    role = roleFilter,
    useMock = false
  ) => {
    setLoading(true)
    setError(null)

    try {
      if (useMock) {
        await new Promise(r => setTimeout(r, 500))

        setUsers(MOCK_USERS)
        setTotal(MOCK_USERS.length)
        setConnectionStatus('connected')

        toast.success('تم تحميل البيانات التجريبية')
        return
      }

      const res = await adminApi.getUsers({
        page: p,
        pageSize: PAGE_SIZE,
        search: q || undefined,
        role: role || undefined
      })

      if (res?.items) {
        setUsers(res.items)
        setTotal(res.total ?? res.items.length)
      } else {
        setUsers([])
        setTotal(0)
      }

      setConnectionStatus('connected')

    } catch (err: any) {
      console.warn('Network error in useUsers, falling back to mock data.', err)

      // ── Automatic Mock Fallback ──────────────────────────────────────
      setUsers(MOCK_USERS)
      setTotal(MOCK_USERS.length)
      setConnectionStatus('error')
      setError('Server unreachable. Running in Demo Mode with mock data.')
      
      // We don't toast error here to avoid annoying the user since we have a fallback
    }
    finally {
      setLoading(false)
    }

  }, [page, search, roleFilter])


  /* ---------------- search debounce ---------------- */

  const debouncedSearch = useMemo(() =>
    debounce((val: string) => {
      setPage(1)
      fetchUsers(1, val, roleFilter)
    }, 400)
  , [fetchUsers, roleFilter])


  const handleSearch = useCallback((value: string) => {
    setSearch(value)
    debouncedSearch(value)
  }, [debouncedSearch])


  const handleRoleFilter = useCallback((role: UserRole | '') => {
    setRoleFilter(role)
    setPage(1)
    fetchUsers(1, search, role)
  }, [search, fetchUsers])


  /* ---------------- toggle ---------------- */

  const handleToggle = useCallback(async (id: number) => {
    try {
      await adminApi.toggleUser(id)

      setUsers(prev =>
        prev.map(u =>
          u.id === id
            ? { ...u, isActive: !u.isActive }
            : u
        )
      )

      toast.success('تم تحديث الحالة')
    } catch {
      toast.error('فشلت العملية')
    }
  }, [])


  /* ---------------- delete ---------------- */

  const handleDelete = useCallback(async (id: number, role: string) => {

    if (!confirm('هل أنت متأكد؟')) return

    try {
      await adminApi.deleteUser(id, role)

      setUsers(prev => prev.filter(u => u.id !== id))
      setTotal(t => t - 1)

      toast.success('تم الحذف')
    } catch {
      toast.error('فشل الحذف')
    }

  }, [])


  const handleRetry = useCallback(() => {
    setError(null)
    fetchUsers()
  }, [fetchUsers])


  const handleUseMockData = useCallback(() => {
    fetchUsers(1, search, roleFilter, true)
  }, [fetchUsers, search, roleFilter])


  const clearFilters = useCallback(() => {
    setSearch('')
    setRoleFilter('')
    setPage(1)
    fetchUsers(1, '', '')
  }, [fetchUsers])


  /* ✅ FIX مهم */
  useEffect(() => {
    fetchUsers(1, search, roleFilter)
  }, []) // أول تحميل

  /* SignalR: Listen for new user registrations */
  useEffect(() => {
    const token = useAuthStore.getState().token
    if (!token) return

    let cleanup: (() => void) | undefined

    startConnection(token).then((conn) => {
      conn.on('NewUserRegistered', (data) => {
        // Add new user to the top of the list
        setUsers(prev => [{
          id: data.userId,
          name: data.name,
          email: data.email,
          role: data.role,
          isActive: true,
          createdAt: data.timestamp,
        }, ...prev])

        setTotal(prev => prev + 1)
        setNewUserNotification(data)

        // Show toast notification
        toast.success(
          `New ${data.role} registered: ${data.name}`,
          {
            icon: '👤',
            duration: 5000
          }
        )
      })
      
      conn.on('DoctorUpdated', (data) => {
        setUsers(prev => prev.map(u => 
          (u.id === data.doctorId || u.id === data.userId)
            ? { ...u, photoUrl: data.photoUrl || u.photoUrl, name: data.doctorName || u.name } 
            : u
        ))
      })

      cleanup = () => {
        conn.off('NewUserRegistered')
        conn.off('DoctorUpdated')
      }
    }).catch(() => {
      // SignalR optional - app works without it
    })

    return () => cleanup?.()
  }, [])

  useEffect(() => {
    fetchUsers(page, search, roleFilter)
  }, [page])


  const handleAddUser = useCallback(async (data: {
    fullName: string
    email: string
    password: string
    role: UserRole
    specialityName?: string
    specialityNameAr?: string
    yearsExperience?: number
    consultationFee?: number
    bio?: string
  }) => {
    try {
      setLoading(true)

      const newUser = await adminApi.createUser({
        fullName: data.fullName,
        email: data.email,
        password: data.password,
        role: data.role,
        specialityName: data.specialityName,
        specialityNameAr: data.specialityNameAr,
        yearsExperience: data.yearsExperience,
        consultationFee: data.consultationFee,
        bio: data.bio
      })

      // تحديث الليستة (أفضل UX)
      setUsers(prev => [newUser, ...prev])
      setTotal(prev => prev + 1)

      toast.success('تم إنشاء المستخدم بنجاح')

    } catch (err: any) {
      const msg =
        err?.response?.data?.message ||
        err?.message ||
        'فشل إنشاء المستخدم'

      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }, [])

  return {
    users,
    total,
    page,
    search,
    roleFilter,
    loading,
    error,
    connectionStatus,
    fetchUsers,
    handleSearch,
    handleRoleFilter,
    handleToggle,
    handleDelete,
    handleRetry,
    handleUseMockData,
    clearFilters,
    handleAddUser,
    setPage,
    PAGE_SIZE,
    newUserNotification,
    setNewUserNotification,
  }
}
