import { useEffect, useState, useCallback, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Search, UserPlus, Filter, Trash2, ToggleLeft, ToggleRight,
  X, Check, AlertCircle, Loader2, ChevronLeft, ChevronRight,
  Mail, Lock, User, Stethoscope, Crown, Activity, RefreshCw,
  Database, Server, WifiOff
} from 'lucide-react'
import { adminApi } from '@/api/adminApi'
import type { UserDto, UserRole, CreateUserRequest } from '@/lib/types'
import toast from 'react-hot-toast'

// ============================================================================
// 🧩 مكونات الواجهة
// ============================================================================

const Card = ({ className = '', children, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`rounded-3xl border border-white/20 bg-white/80 backdrop-blur-xl shadow-xl ${className}`} {...props}>
    {children}
  </div>
)

const CardHeader = ({ className = '', ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`flex flex-col space-y-2 p-6 ${className}`} {...props} />
)

const CardTitle = ({ className = '', ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
  <h3 className={`text-xl font-bold tracking-tight text-gray-900 ${className}`} {...props} />
)

const CardDescription = ({ className = '', ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
  <p className={`text-sm text-gray-500 ${className}`} {...props} />
)

const CardContent = ({ className = '', ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`p-6 pt-0 ${className}`} {...props} />
)

const Button = ({ 
  variant = 'primary', 
  size = 'md', 
  className = '', 
  loading = false,
  disabled = false,
  icon,
  children,
  ...props 
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { 
  variant?: 'primary' | 'outline' | 'ghost' | 'destructive' | 'glass' | 'success'; 
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
}) => {
  const base = 'inline-flex items-center justify-center font-semibold rounded-2xl transition-all duration-300 focus:outline-none focus:ring-4 disabled:opacity-50 disabled:cursor-not-allowed'
  
  const variants = {
    primary: 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40 hover:-translate-y-0.5',
    outline: 'border-2 border-gray-200 bg-white/80 text-gray-700 hover:bg-white hover:border-gray-300 hover:shadow-lg',
    ghost: 'text-gray-600 hover:text-gray-900 hover:bg-gray-100/80',
    destructive: 'bg-gradient-to-r from-red-500 to-rose-600 text-white hover:from-red-600 hover:to-rose-700 shadow-lg shadow-red-500/30',
    glass: 'bg-white/20 backdrop-blur-md border border-white/30 text-gray-700 hover:bg-white/30 hover:shadow-xl',
    success: 'bg-gradient-to-r from-emerald-500 to-teal-600 text-white hover:from-emerald-600 hover:to-teal-700 shadow-lg shadow-emerald-500/30',
  }
  
  const sizes = {
    sm: 'h-9 px-4 text-xs gap-1.5',
    md: 'h-11 px-5 text-sm gap-2',
    lg: 'h-12 px-6 text-base gap-2.5',
  }
  
  return (
    <button 
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`} 
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Loader2 className="w-4 h-4 animate-spin" />}
      {!loading && icon && <span className="flex-shrink-0">{icon}</span>}
      <span className={loading ? 'mr-2' : ''}>{children}</span>
    </button>
  )
}

const Badge = ({ 
  variant = 'default', 
  className = '', 
  children,
  ...props 
}: React.HTMLAttributes<HTMLSpanElement> & { variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'admin' }) => {
  const variants = {
    default: 'bg-gray-100/80 text-gray-700 border border-gray-200',
    success: 'bg-gradient-to-r from-green-400 to-emerald-500 text-white shadow-md shadow-green-500/20',
    warning: 'bg-gradient-to-r from-amber-400 to-orange-500 text-white shadow-md shadow-amber-500/20',
    danger: 'bg-gradient-to-r from-red-400 to-rose-500 text-white shadow-md shadow-red-500/20',
    info: 'bg-gradient-to-r from-blue-400 to-indigo-500 text-white shadow-md shadow-blue-500/20',
    admin: 'bg-gradient-to-r from-violet-500 to-purple-600 text-white shadow-md shadow-purple-500/20',
  }
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${variants[variant]} ${className}`} {...props}>
      {children}
    </span>
  )
}

const Input = ({ 
  className = '', 
  error,
  icon,
  ...props 
}: React.InputHTMLAttributes<HTMLInputElement> & { error?: string; icon?: React.ReactNode }) => (
  <div className="relative">
    {icon && (
      <div className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400">
        {icon}
      </div>
    )}
    <input
      className={`w-full ${icon ? 'pr-10' : 'pr-4'} pl-4 py-3 text-sm border rounded-2xl focus:outline-none focus:ring-4 transition-all duration-200 ${
        error 
          ? 'border-red-300 bg-red-50/50 focus:ring-red-200 focus:border-red-400' 
          : 'border-gray-200 bg-white/50 focus:ring-blue-200 focus:border-blue-400 hover:border-gray-300'
      } ${className}`}
      {...props}
    />
    {error && <p className="mt-1.5 text-xs text-red-500 font-medium flex items-center gap-1"><AlertCircle className="w-3 h-3" />{error}</p>}
  </div>
)

const Select = ({ 
  className = '', 
  error,
  children,
  ...props 
}: React.SelectHTMLAttributes<HTMLSelectElement> & { error?: string }) => (
  <div className="relative">
    <select
      className={`w-full px-4 py-3 text-sm border rounded-2xl focus:outline-none focus:ring-4 transition-all duration-200 appearance-none bg-white/50 ${
        error 
          ? 'border-red-300 bg-red-50/50 focus:ring-red-200 focus:border-red-400' 
          : 'border-gray-200 focus:ring-blue-200 focus:border-blue-400 hover:border-gray-300'
      } ${className}`}
      {...props}
    >
      {children}
    </select>
    <ChevronLeft className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
    {error && <p className="mt-1.5 text-xs text-red-500 font-medium">{error}</p>}
  </div>
)

const Modal = ({ 
  open, 
  onClose, 
  title, 
  children, 
  footer,
  size = 'md'
}: { 
  open: boolean; 
  onClose: () => void; 
  title: string; 
  children: React.ReactNode;
  footer?: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}) => {
  const sizes = { sm: 'max-w-md', md: 'max-w-lg', lg: 'max-w-2xl', xl: 'max-w-4xl' }
  
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50"
          />
          <div className="fixed inset-0 flex items-center justify-center p-4 z-50 pointer-events-none">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              onClick={(e) => e.stopPropagation()}
              className={`w-full ${sizes[size]} pointer-events-auto`}
            >
              <Card className="border-0 shadow-2xl">
                <div className="flex items-center justify-between p-5 border-b border-gray-100">
                  <h3 className="text-lg font-bold text-gray-900">{title}</h3>
                  <button onClick={onClose} className="p-2 rounded-xl hover:bg-gray-100 transition-colors text-gray-400 hover:text-gray-600">
                    <X className="w-5 h-5" />
                  </button>
                </div>
                <div className="p-5 max-h-[70vh] overflow-y-auto">{children}</div>
                {footer && (
                  <div className="flex items-center justify-end gap-3 p-5 border-t border-gray-100 bg-gray-50/50 rounded-b-3xl">
                    {footer}
                  </div>
                )}
              </Card>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  )
}

const Skeleton = ({ className = '', ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`animate-pulse rounded-xl bg-gray-200/80 ${className}`} {...props} />
)

const Pagination = ({ 
  total, 
  page, 
  pageSize, 
  onChange 
}: { 
  total: number; 
  page: number; 
  pageSize: number; 
  onChange: (page: number) => void;
}) => {
  const totalPages = Math.ceil(total / pageSize)
  if (totalPages <= 1) return null
  
  const pages = useMemo(() => {
    const items: (number | string)[] = []
    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) items.push(i)
    } else {
      items.push(1)
      if (page > 3) items.push('...')
      for (let i = Math.max(2, page - 1); i <= Math.min(totalPages - 1, page + 1); i++) items.push(i)
      if (page < totalPages - 2) items.push('...')
      items.push(totalPages)
    }
    return items
  }, [page, totalPages])
  
  return (
    <div className="flex items-center justify-center gap-1.5 p-4">
      <Button variant="outline" size="sm" onClick={() => onChange(page - 1)} disabled={page === 1} className="px-3">
        <ChevronRight className="w-4 h-4" />
      </Button>
      {pages.map((p, i) => (
        p === '...' ? (
          <span key={`ellipsis-${i}`} className="px-3 py-2 text-sm text-gray-400">...</span>
        ) : (
          <button
            key={p}
            onClick={() => onChange(p as number)}
            className={`min-w-[40px] h-10 px-3 py-2 text-sm font-medium rounded-xl transition-all duration-200 ${
              page === p
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/30'
                : 'text-gray-600 hover:bg-gray-100/80'
            }`}
          >
            {p}
          </button>
        )
      ))}
      <Button variant="outline" size="sm" onClick={() => onChange(page + 1)} disabled={page === totalPages} className="px-3">
        <ChevronLeft className="w-4 h-4" />
      </Button>
    </div>
  )
}

const Table = ({ className = '', ...props }: React.HTMLAttributes<HTMLTableElement>) => (
  <div className="overflow-x-auto"><table className={`w-full ${className}`} {...props} /></div>
)
const TableHeader = ({ className = '', ...props }: React.HTMLAttributes<HTMLTableSectionElement>) => (
  <thead className={`bg-gray-50/80 ${className}`} {...props} />
)
const TableRow = ({ className = '', ...props }: React.HTMLAttributes<HTMLTableRowElement>) => (
  <tr className={`border-b border-gray-100 hover:bg-gray-50/50 transition-colors ${className}`} {...props} />
)
const TableHead = ({ className = '', ...props }: React.HTMLAttributes<HTMLTableCellElement>) => (
  <th className={`px-4 py-4 text-right text-xs font-semibold text-gray-500 uppercase tracking-wider ${className}`} {...props} />
)
const TableCell = ({ className = '', ...props }: React.HTMLAttributes<HTMLTableCellElement>) => (
  <td className={`px-4 py-4 text-sm text-gray-700 ${className}`} {...props} />
)

const UserTable = ({ users, onToggle, onDelete }: { users: UserDto[]; onToggle: (id: string) => void; onDelete: (id: string) => void }) => {
  const getRoleBadge = (role: UserRole) => {
    const config = {
      Admin: { variant: 'admin' as const, icon: <Crown className="w-3 h-3" />, label: 'مدير' },
      Doctor: { variant: 'success' as const, icon: <Stethoscope className="w-3 h-3" />, label: 'طبيب' },
      Patient: { variant: 'info' as const, icon: <User className="w-3 h-3" />, label: 'مريض' },
    }
    const c = config[role]
    return <Badge variant={c.variant}>{c.icon}{c.label}</Badge>
  }
  
  const getStatusBadge = (isActive: boolean) => (
    <Badge variant={isActive ? 'success' : 'danger'}>
      {isActive ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
      {isActive ? 'نشط' : 'معطل'}
    </Badge>
  )
  
  if (!users || users.length === 0) {
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center py-16 text-center">
        <div className="p-4 rounded-2xl bg-gray-100/80 mb-4"><User className="w-8 h-8 text-gray-400" /></div>
        <p className="text-sm text-gray-500 font-medium">لا توجد مستخدمين لعرضهم</p>
      </motion.div>
    )
  }
  
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="rounded-r-2xl">المستخدم</TableHead>
          <TableHead>البريد الإلكتروني</TableHead>
          <TableHead>الدور</TableHead>
          <TableHead>الحالة</TableHead>
          <TableHead>تاريخ الإنشاء</TableHead>
          <TableHead className="rounded-l-2xl text-center">الإجراءات</TableHead>
        </TableRow>
      </TableHeader>
      <tbody>
        <AnimatePresence mode="popLayout">
          {users.map((user, index) => (
            <motion.tr
              key={user.userId}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ delay: index * 0.05 }}
              className="group"
            >
              <TableCell className="font-medium">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-bold shadow-lg shadow-blue-500/20">
                    {user.fullName.charAt(0)}
                  </div>
                  <span className="text-gray-900 font-semibold">{user.fullName}</span>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2 text-gray-600">
                  <Mail className="w-4 h-4 text-gray-400" />
                  {user.email}
                </div>
              </TableCell>
              <TableCell>{getRoleBadge(user.role)}</TableCell>
              <TableCell>{getStatusBadge(user.isActive)}</TableCell>
              <TableCell className="text-gray-500">{new Date(user.createdAt).toLocaleDateString('ar-EG')}</TableCell>
              <TableCell>
                <div className="flex items-center justify-center gap-2">
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => onToggle(user.userId)}
                    className={`p-2.5 rounded-xl transition-all duration-200 ${user.isActive ? 'bg-amber-100 text-amber-600 hover:bg-amber-200' : 'bg-green-100 text-green-600 hover:bg-green-200'}`}
                    title={user.isActive ? 'تعطيل المستخدم' : 'تفعيل المستخدم'}
                  >
                    {user.isActive ? <ToggleRight className="w-5 h-5" /> : <ToggleLeft className="w-5 h-5" />}
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => { if(confirm('هل أنت متأكد من حذف هذا المستخدم؟')) onDelete(user.userId) }}
                    className="p-2.5 rounded-xl bg-red-100 text-red-600 hover:bg-red-200 transition-all duration-200"
                    title="حذف المستخدم"
                  >
                    <Trash2 className="w-5 h-5" />
                  </motion.button>
                </div>
              </TableCell>
            </motion.tr>
          ))}
        </AnimatePresence>
      </tbody>
    </Table>
  )
}

function debounce<T extends (...args: any[]) => any>(func: T, wait: number) {
  let timeout: ReturnType<typeof setTimeout> | null = null
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}

// ============================================================================
// 📊 بيانات تجريبية (Mock Data) للاختبار
// ============================================================================

const MOCK_USERS: UserDto[] = [
  {
    userId: '1',
    fullName: 'د. أحمد محمد علي',
    email: 'ahmed@medbook.com',
    role: 'Doctor',
    isActive: true,
    createdAt: new Date().toISOString(),
  },
  {
    userId: '2',
    fullName: 'منى إبراهيم',
    email: 'mona@medbook.com',
    role: 'Patient',
    isActive: true,
    createdAt: new Date().toISOString(),
  },
  {
    userId: '3',
    fullName: 'خالد محمود',
    email: 'khaled@medbook.com',
    role: 'Admin',
    isActive: true,
    createdAt: new Date().toISOString(),
  },
]

// ============================================================================
// 🎯 المكون الرئيسي: AdminUsers
// ============================================================================

const PAGE_SIZE = 10

export default function AdminUsers() {
  const [users, setUsers] = useState<UserDto[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [roleFilter, setRoleFilter] = useState<UserRole | ''>('')
  const [loading, setLoading] = useState(false)
  const [showAddModal, setShowAddModal] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'checking' | 'connected' | 'error'>('checking')

  const [form, setForm] = useState<CreateUserRequest>({
    fullName: '', email: '', password: '', role: 'Patient',
    specialityName: '', specialityNameAr: '', consultationFee: 0, yearsExperience: 0, bio: '',
  })
  const [formErrors, setFormErrors] = useState<Partial<Record<keyof CreateUserRequest, string>>>({})
  const [addLoading, setAddLoading] = useState(false)

const fetchUsers = useCallback(async (p = 1, q = search, role = roleFilter, useMock = false) => {
  setLoading(true)
  setError(null)
  
  try {
    console.log('🔄 Fetching users... Page:', p, 'Search:', q, 'Role:', role)
    
    // ✅ بيانات تجريبية للاختبار
    if (useMock) {
      await new Promise(resolve => setTimeout(resolve, 500))
      setUsers(MOCK_USERS)
      setTotal(MOCK_USERS.length)
      setConnectionStatus('connected')
      toast.success('تم تحميل البيانات التجريبية')
      return
    }
    
    console.log('🌐 Calling adminApi.getUsers...')
    const res = await adminApi.getUsers({ 
      page: p, 
      pageSize: PAGE_SIZE, 
      search: q || undefined, 
      role: role || undefined 
    })
    
    console.log('✅ Raw API Response:', res)
    console.log('✅ Response type:', Array.isArray(res) ? 'ARRAY ✓' : 'OBJECT')
    
    // ✅ التعامل مع تنسيق المصفوفة المباشرة
    if (Array.isArray(res)) {
      // 🔹 الحالة: الـ API بيرجع مصفوفة مباشرة [...]
      console.log('📦 Processing array response...')
      
      // فلترة البحث (على المستوى المحلي لأن الـ backend مش بيدعمه بعد)
      let filtered = res
      if (q && q.trim()) {
        const lowerQ = q.toLowerCase()
        filtered = filtered.filter((u: any) => 
          u.name?.toLowerCase().includes(lowerQ) || 
          u.email?.toLowerCase().includes(lowerQ)
        )
        console.log(`🔍 Filtered by search: ${filtered.length} results`)
      }
      
      // فلترة الدور (على المستوى المحلي)
      if (role && role.trim()) {
        filtered = filtered.filter((u: any) => u.role === role)
        console.log(`🔍 Filtered by role: ${filtered.length} results`)
      }
      
      // تطبيق Pagination محلياً
      const startIndex = (p - 1) * PAGE_SIZE
      const paginated = filtered.slice(startIndex, startIndex + PAGE_SIZE)
      
      console.log(`✅ Displaying page ${p}: ${paginated.length} of ${filtered.length} users`)
      
      setUsers(paginated)
      setTotal(filtered.length) // ✅ الإجمالي بعد الفلترة
      setConnectionStatus('connected')
      
      if (filtered.length === 0) {
        toast.warning('لا توجد نتائج مطابقة')
      }
      
    } else if (res?.items && Array.isArray(res.items)) {
      // 🔹 الحالة: الـ API بيرجع { items: [], total: 123 }
      console.log('📦 Processing paginated object response...')
      setUsers(res.items)
      setTotal(res.total || res.items.length)
      setConnectionStatus('connected')
    } else {
      // 🔹 حالة غير متوقعة
      console.warn('⚠️ Unexpected response format:', res)
      setUsers([])
      setTotal(0)
      setConnectionStatus('connected')
      toast.warning('تنسيق استجابة غير متوقع')
    }
    
  } catch (err: any) {
    console.error('❌ Failed to fetch users:', err)
    console.error('Error details:', err?.response?.data || err?.message || err)
    
    const errorMsg = err?.response?.data?.message || err?.message || 'فشل الاتصال بالخادم'
    setError(errorMsg)
    setConnectionStatus('error')
    toast.error(`خطأ: ${errorMsg}`)
    
    setUsers([])
    setTotal(0)
  } finally {
    setLoading(false)
  }
}, [search, roleFilter])

  const debouncedSearch = useMemo(
    () => debounce((val: string) => { setPage(1); fetchUsers(1, val, roleFilter) }, 400),
    [roleFilter, fetchUsers]
  )

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearch(e.target.value)
    debouncedSearch(e.target.value)
  }

  const handleRoleFilter = (role: UserRole | '') => { setRoleFilter(role); setPage(1); fetchUsers(1, search, role) }
  
  const handleToggle = async (id: string) => {
    try {
      await adminApi.toggleUser(id)
      setUsers(prev => prev.map(u => u.userId === id ? { ...u, isActive: !u.isActive } : u))
      toast.success('تم تحديث حالة المستخدم')
    } catch { toast.error('فشلت العملية') }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('هل أنت متأكد من حذف هذا المستخدم؟')) return
    try {
      await adminApi.deleteUser(id)
      setUsers(prev => prev.filter(u => u.userId !== id))
      setTotal(t => t - 1)
      toast.success('تم حذف المستخدم بنجاح')
    } catch { toast.error('فشل حذف المستخدم') }
  }

  const validateForm = (): boolean => {
    const errors: typeof formErrors = {}
    if (!form.fullName.trim()) errors.fullName = 'الاسم الكامل مطلوب'
    if (!form.email.trim()) errors.email = 'البريد الإلكتروني مطلوب'
    else if (!/\S+@\S+\.\S+/.test(form.email)) errors.email = 'بريد غير صحيح'
    if (!form.password || form.password.length < 8) errors.password = '8 أحرف على الأقل'
    if (form.role === 'Doctor') {
      if (!form.specialityName?.trim()) errors.specialityName = 'التخصص مطلوب'
      if (!form.specialityNameAr?.trim()) errors.specialityNameAr = 'التخصص بالعربية مطلوب'
    }
    setFormErrors(errors)
    return Object.keys(errors).length === 0
  }

  const handleAddUser = async () => {
    if (!validateForm()) return
    setAddLoading(true)
    try {
      const newUser = await adminApi.createUser(form)
      setUsers(prev => [newUser, ...prev]); setTotal(t => t + 1)
      toast.success(`✅ تم إنشاء حساب "${newUser.fullName}"`)
      setShowAddModal(false); setForm({ fullName: '', email: '', password: '', role: 'Patient' }); setFormErrors({})
    } catch (err: any) { toast.error(err?.response?.data?.message || 'فشل الإنشاء') }
    finally { setAddLoading(false) }
  }

  const setField = (key: keyof CreateUserRequest, val: unknown) => {
    setForm(p => ({ ...p, [key]: val })); setFormErrors(p => ({ ...p, [key]: undefined }))
  }

  const handleRetry = () => {
    setError(null)
    fetchUsers()
  }

  const handleUseMockData = () => {
    fetchUsers(1, search, roleFilter, true)
  }

  // ── حالة الاتصال بالخادم ─────────────────────────────────────────────
  if (connectionStatus === 'error' && users.length === 0 && !loading) {
    return (
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-red-50/30 p-8 flex items-center justify-center"
        dir="rtl"
      >
        <Card className="max-w-2xl w-full border-0 shadow-2xl">
          <CardContent className="p-8 text-center space-y-6">
            <motion.div 
              className="p-6 rounded-3xl bg-gradient-to-br from-red-50 to-rose-100 border-2 border-red-200 shadow-xl shadow-red-500/20 inline-block"
              animate={{ rotate: [0, -5, 5, 0] }}
              transition={{ duration: 0.5 }}
            >
              <WifiOff className="w-16 h-16 text-red-500" />
            </motion.div>
            
            <div className="space-y-3">
              <h3 className="text-2xl font-black text-gray-900">تعذر الاتصال بالخادم</h3>
              <p className="text-gray-600">{error}</p>
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-800">
                <p className="font-semibold mb-1">💡 الحلول المقترحة:</p>
                <ul className="text-right space-y-1 list-disc list-inside">
                  <li>تأكد أن الـ Backend شغال على http://localhost:5194</li>
                  <li>تحقق من CORS settings</li>
                  <li>تأكد من صحة الـ API endpoint</li>
                </ul>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-3 justify-center pt-4">
              <Button onClick={handleRetry} variant="primary" size="lg" className="gap-2">
                <RefreshCw className="w-5 h-5" />
                إعادة المحاولة
              </Button>
              <Button onClick={handleUseMockData} variant="success" size="lg" className="gap-2">
                <Database className="w-5 h-5" />
                استخدام بيانات تجريبية
              </Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    )
  }

  // ── الواجهة الرئيسية ──────────────────────────────────────────────────
  return (
    <motion.div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-blue-50/30 p-4 md:p-6 lg:p-8" initial={{opacity:0}} animate={{opacity:1}} dir="rtl">
      
      {/* الهيدر */}
      <motion.div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 mb-6 border-b-2 border-gray-100" initial={{y:-20,opacity:0}} animate={{y:0,opacity:1}}>
        <div className="space-y-1">
          <motion.h1 className="text-3xl font-black bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent" initial={{opacity:0}} animate={{opacity:1}} transition={{delay:0.1}}>
            إدارة المستخدمين
          </motion.h1>
          <motion.p className="text-sm text-gray-500 font-medium flex items-center gap-2" initial={{opacity:0}} animate={{opacity:1}} transition={{delay:0.2}}>
            <Activity className="w-4 h-4" />
            إجمالي <span className="font-bold text-gray-900">{total.toLocaleString('ar-EG')}</span> مستخدم
            {connectionStatus === 'connected' && <Check className="w-4 h-4 text-green-500 mr-1" />}
            {connectionStatus === 'error' && <AlertCircle className="w-4 h-4 text-red-500 mr-1" />}
          </motion.p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleUseMockData} className="gap-2">
            <Database className="w-4 h-4" />
            بيانات تجريبية
          </Button>
          <Button variant="primary" size="md" icon={<UserPlus className="w-4 h-4" />} onClick={() => setShowAddModal(true)} className="shadow-xl hover:shadow-2xl">
            إضافة مستخدم
          </Button>
        </div>
      </motion.div>

      {/* البطاقة الرئيسية */}
      <motion.div initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} transition={{delay:0.2}}>
        <Card className="border-0 shadow-2xl hover:shadow-3xl transition-shadow duration-300">
          
          {/* شريط حالة الاتصال */}
          <div className={`px-5 py-3 border-b border-gray-100 flex items-center gap-2 ${
            connectionStatus === 'connected' ? 'bg-green-50' : 
            connectionStatus === 'error' ? 'bg-red-50' : 'bg-amber-50'
          }`}>
            {connectionStatus === 'connected' ? (
              <><Check className="w-4 h-4 text-green-600" /><span className="text-sm text-green-700">متصل بالخادم</span></>
            ) : connectionStatus === 'error' ? (
              <><AlertCircle className="w-4 h-4 text-red-600" /><span className="text-sm text-red-700">غير متصل - {error}</span></>
            ) : (
              <><Loader2 className="w-4 h-4 text-amber-600 animate-spin" /><span className="text-sm text-amber-700">جاري الاتصال...</span></>
            )}
          </div>

          {/* الفلاتر */}
          <div className="flex flex-wrap items-center gap-3 p-5 border-b border-gray-100 bg-white/50 rounded-t-3xl">
            <div className="relative flex-1 min-w-48 max-w-md">
              <Search className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={search}
                onChange={handleSearch}
                placeholder="🔍 بحث بالاسم أو البريد الإلكتروني..."
                className="w-full pr-10 pl-4 py-3 text-sm border border-gray-200 rounded-2xl bg-white/80 focus:outline-none focus:ring-4 focus:ring-blue-200 focus:border-blue-400 transition-all duration-200"
              />
            </div>
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-gray-400" />
              <div className="flex items-center gap-1.5 bg-gray-100/80 rounded-2xl p-1">
                {(['', 'Admin', 'Doctor', 'Patient'] as const).map(role => (
                  <motion.button key={role} whileHover={{scale:1.05}} whileTap={{scale:0.95}} onClick={() => handleRoleFilter(role)} className={`px-4 py-2 text-xs font-semibold rounded-xl transition-all duration-200 ${roleFilter===role?'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/30':'text-gray-600 hover:bg-white hover:shadow-md'}`}>
                    {role===''?'📋 الكل':role==='Admin'?'👑 مدراء':role==='Doctor'?'🩺 أطباء':'👤 مرضى'}
                  </motion.button>
                ))}
              </div>
            </div>
            {(search || roleFilter) && <Button variant="ghost" size="sm" onClick={()=>{setSearch('');setRoleFilter('');fetchUsers(1,'','')}} className="text-gray-500 hover:text-gray-700"><X className="w-4 h-4" /> مسح</Button>}
          </div>

          {/* الجدول */}
          <CardContent className="p-0">
            {loading ? (
              <div className="p-6"><div className="space-y-4">{[...Array(5)].map((_,i)=><div key={i} className="flex items-center gap-4 py-3"><Skeleton className="w-10 h-10 rounded-2xl" /><div className="flex-1 space-y-2"><Skeleton className="h-4 w-32" /><Skeleton className="h-3 w-48" /></div><Skeleton className="h-6 w-16 rounded-full" /><Skeleton className="h-6 w-16 rounded-full" /></div>)}</div></div>
            ) : <UserTable users={users} onToggle={handleToggle} onDelete={handleDelete} />}
          </CardContent>

          {/* الترقيم */}
          {!loading && total > PAGE_SIZE && <div className="border-t border-gray-100 bg-gray-50/50 rounded-b-3xl"><Pagination total={total} page={page} pageSize={PAGE_SIZE} onChange={(p)=>{setPage(p);fetchUsers(p)}} /></div>}
        </Card>
      </motion.div>

      {/* مودال إضافة مستخدم */}
      <Modal open={showAddModal} onClose={()=>{setShowAddModal(false);setFormErrors({})}} title="✨ إضافة مستخدم جديد" size="lg" footer={<>
        <Button variant="outline" onClick={()=>{setShowAddModal(false);setFormErrors({})}}>إلغاء</Button>
        <Button variant="primary" onClick={handleAddUser} loading={addLoading}><Check className="w-4 h-4" /> إنشاء الحساب</Button>
      </>}>
        <div className="space-y-5">
          <motion.div initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-2xl p-4 text-sm text-blue-800 flex gap-3">
            <div className="flex-shrink-0 p-1.5 bg-blue-100 rounded-xl"><AlertCircle className="w-4 h-4" /></div>
            <p>فقط مدير النظام يمكنه إضافة مستخدمين جدد. سيتم إرسال بيانات الدخول إلى البريد الإلكتروني للمستخدم.</p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">الاسم الكامل *</label>
              <Input type="text" value={form.fullName} onChange={(e)=>setField('fullName',e.target.value)} placeholder="د. أحمد محمد علي" icon={<User className="w-4 h-4" />} error={formErrors.fullName} />
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">البريد الإلكتروني *</label>
              <Input type="email" value={form.email} onChange={(e)=>setField('email',e.target.value)} placeholder="user@medbook.com" icon={<Mail className="w-4 h-4" />} error={formErrors.email} />
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">كلمة المرور *</label>
              <Input type="password" value={form.password} onChange={(e)=>setField('password',e.target.value)} placeholder="•••••••• (8 أحرف على الأقل)" icon={<Lock className="w-4 h-4" />} error={formErrors.password} />
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">الدور *</label>
              <Select value={form.role} onChange={(e)=>setField('role',e.target.value as UserRole)}>
                <option value="Patient">👤 مريض</option>
                <option value="Doctor">🩺 طبيب</option>
                <option value="Admin">👑 مدير نظام</option>
              </Select>
            </div>
          </div>

          <AnimatePresence>
            {form.role === 'Doctor' && (
              <motion.div initial={{opacity:0,height:0}} animate={{opacity:1,height:'auto'}} exit={{opacity:0,height:0}} className="border-t border-gray-100 pt-5 space-y-4">
                <div className="flex items-center gap-2 pb-2"><Stethoscope className="w-5 h-5 text-emerald-500" /><p className="text-sm font-bold text-gray-800">بيانات الطبيب المهنية</p></div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div><label className="block text-sm font-semibold text-gray-700 mb-2">التخصص (إنجليزي) *</label><Input type="text" value={form.specialityName} onChange={(e)=>setField('specialityName',e.target.value)} placeholder="Cardiology" error={formErrors.specialityName} /></div>
                  <div><label className="block text-sm font-semibold text-gray-700 mb-2">التخصص (عربي) *</label><Input type="text" value={form.specialityNameAr} onChange={(e)=>setField('specialityNameAr',e.target.value)} placeholder="أمراض القلب والأوعية الدموية" error={formErrors.specialityNameAr} /></div>
                  <div><label className="block text-sm font-semibold text-gray-700 mb-2">سنوات الخبرة</label><Input type="number" min={0} max={50} value={form.yearsExperience||''} onChange={(e)=>setField('yearsExperience',Number(e.target.value)||0)} placeholder="مثال: 10" /></div>
                  <div><label className="block text-sm font-semibold text-gray-700 mb-2">رسوم الاستشارة (ج.م)</label><Input type="number" min={0} value={form.consultationFee||''} onChange={(e)=>setField('consultationFee',Number(e.target.value)||0)} placeholder="مثال: 500" /></div>
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">نبذة تعريفية</label>
                  <textarea rows={3} value={form.bio} onChange={(e)=>setField('bio',e.target.value)} placeholder="اكتب نبذة مختصرة عن خبرة الطبيب وإنجازاته..." className="w-full px-4 py-3 text-sm border border-gray-200 rounded-2xl bg-white/50 focus:outline-none focus:ring-4 focus:ring-blue-200 focus:border-blue-400 transition-all duration-200 resize-none" />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </Modal>
    </motion.div>
  )
}