import { useEffect, useState, useCallback, useMemo, use } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Search, UserPlus, Filter, Trash2, ToggleLeft, ToggleRight,
  X, Check, AlertCircle, Loader2, ChevronLeft, ChevronRight,
  Mail, Lock, User, Stethoscope, Crown, Activity, RefreshCw,
  Database, Server, WifiOff
} from 'lucide-react'
import { adminApi } from '@/api/adminApi'
import type { UserDto, UserRole, CreateUserRequest } from '@/lib/types'
import {Card ,CardHeader,CardTitle,CardDescription , CardContent} from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import { Modal } from '@/components/ui/Modal'
import { Skeleton } from '@/components/ui/Skeleton'
import { UserTable } from '@/components/admin/users/UserTable'
import { useUsers  } from '@/components/admin/users/useUsers'
import { Pagination } from '@/components/ui/Pagination'
import toast from 'react-hot-toast'

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
  const {
  fetchUsers,
  handleSearch,
  handleRoleFilter,
  handleToggle,
  handleDelete,
  handleAddUser,
  setPage,
  page,
  handleRetry,
  handleUseMockData,



  

  } = useUsers()

  const setField = (key: keyof CreateUserRequest, val: unknown) => {
    setForm(p => ({ ...p, [key]: val })); setFormErrors(p => ({ ...p, [key]: undefined }))
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
                onChange={(e) => handleSearch(e.target.value)}
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
          {!loading && total > PAGE_SIZE && <div className="border-t border-gray-100 bg-gray-50/50 rounded-b-3xl"><Pagination total={total} page={page} pageSize={PAGE_SIZE} onChange={(p:any)=>{setPage(p);fetchUsers(p)}} /></div>}
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