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

        let filtered = res as UserDto[]

        if (q?.trim()) {

          const lower = q.toLowerCase()

          filtered = filtered.filter(u =>
            u.fullName?.toLowerCase().includes(lower) ||
            u.email?.toLowerCase().includes(lower)
          )

        }

        if (role) {
          filtered = filtered.filter(u => u.role === role)
        }

        const start = (p - 1) * PAGE_SIZE

        const paginated =
          filtered.slice(start, start + PAGE_SIZE)

        setUsers(paginated)

        setTotal(filtered.length)

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

    }

    catch (err: any) {

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


  /* ---------------- search handler ---------------- */

  const handleSearch = useCallback((value: string) => {

    setSearch(value)

    debouncedSearch(value)

  }, [debouncedSearch])


  /* ---------------- role filter ---------------- */

  const handleRoleFilter = useCallback((role: UserRole | '') => {

    setRoleFilter(role)

    setPage(1)

    fetchUsers(1, search, role)

  }, [search, fetchUsers])


  /* ---------------- toggle user ---------------- */

  const handleToggle = useCallback(async (id: string) => {

    try {

      await adminApi.toggleUser(id)

      setUsers(prev =>
        prev.map(u =>
          u.userId === id
            ? { ...u, isActive: !u.isActive }
            : u
        )
      )

      toast.success('تم تحديث حالة المستخدم')

    }

    catch {

      toast.error('فشلت العملية')

    }

  }, [])


  /* ---------------- delete user ---------------- */

  const handleDelete = useCallback(async (id: string) => {

    if (!confirm('هل أنت متأكد من حذف هذا المستخدم؟'))
      return

    try {

      await adminApi.deleteUser(id)

      setUsers(prev =>
        prev.filter(u => u.userId !== id)
      )

      setTotal(t => t - 1)

      toast.success('تم حذف المستخدم')

    }

    catch {

      toast.error('فشل حذف المستخدم')

    }

  }, [])


  /* ---------------- retry ---------------- */

  const handleRetry = useCallback(() => {

    setError(null)

    fetchUsers()

  }, [fetchUsers])


  /* ---------------- mock data ---------------- */

  const handleUseMockData = useCallback(() => {

    fetchUsers(1, search, roleFilter, true)

  }, [fetchUsers, search, roleFilter])


  /* ---------------- clear filters ---------------- */

  const clearFilters = useCallback(() => {

    setSearch('')
    setRoleFilter('')

    setPage(1)

    fetchUsers(1, '', '')

  }, [fetchUsers])


  /* ---------------- page change ---------------- */

  useEffect(() => {

    fetchUsers(page, search, roleFilter)

  }, [page])

  const  handleAddUser = () => {
  toast.error('هذه الميزة غير متاحة حالياً')
}
  /* ---------------- return ---------------- */

  return {

    users,
    total,

    setUsers,
    setTotal,

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

