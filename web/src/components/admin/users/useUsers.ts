import { useState, useCallback, useMemo, useEffect } from 'react'
import { adminApi } from '@/api/adminApi'
import type { UserDto, UserRole } from '@/lib/types'
import toast from 'react-hot-toast'
import { debounce } from '@/lib/utils/debounce'
import { MOCK_USERS, PAGE_SIZE } from './constants'

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

      if (Array.isArray(res)) {

        let mapped = res.map((u: any) => ({
          id: u.id,
          name: u.name,
          email: u.email,
          role: u.role,
          isActive: u.isActive
        }))

        // ✅ search
        if (q?.trim()) {
          const lower = q.toLowerCase()
          mapped = mapped.filter(u =>
            u.name?.toLowerCase().includes(lower) ||
            u.email?.toLowerCase().includes(lower)
          )
        }

        // ✅ filter
        if (role) {
          mapped = mapped.filter(u => u.role === role)
        }

        const start = (p - 1) * PAGE_SIZE
        const paginated = mapped.slice(start, start + PAGE_SIZE)

        setUsers(paginated)
        setTotal(mapped.length)
      }

      else if (res?.items) {
        setUsers(res.items)
        setTotal(res.total ?? res.items.length)
      }

      else {
        setUsers([])
        setTotal(0)
      }

      setConnectionStatus('connected')

    } catch (err: any) {
      console.error(err)

      const msg =
        err?.response?.data?.message ||
        err?.message ||
        'فشل الاتصال بالخادم'

      setError(msg)
      setConnectionStatus('error')

      toast.error(msg)

      setUsers([])
      setTotal(0)
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
      await adminApi.toggleUser(id as unknown as string)

      setUsers(prev =>
        prev.map(u =>
          u.id === (id as unknown as string)
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

  const handleDelete = useCallback(async (id: number) => {

    if (!confirm('هل أنت متأكد؟')) return

    try {
      await adminApi.deleteUser(id as unknown as string)

      setUsers(prev => prev.filter(u => u.id !== (id as unknown as string)))
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


  useEffect(() => {
    fetchUsers(page, search, roleFilter)
  }, [page])


 const handleAddUser = useCallback(async (data: {
  fullName: string
  email: string
  passwordHash: string
  role: UserRole
}) => {
  try {
    setLoading(true)

    const newUser = await adminApi.createUser({
      fullName : data.fullName,
      email: data.email,
      passwordHash: data.passwordHash,
      role: data.role
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
  }
}