import { useEffect, useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Search, UserPlus, Filter } from 'lucide-react'
import { adminApi } from '@/api/adminApi'
import type { UserDto, UserRole, CreateUserRequest } from '@/lib/types'
import { UserTable } from '@/components/admin/UserTable'
import { Modal } from '@/components/ui/Modal'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Pagination } from '@/components/ui/Table'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import toast from 'react-hot-toast'
import { debounce } from '@/lib/utils'

const PAGE_SIZE = 10

export default function AdminUsers() {
  const [users, setUsers] = useState<UserDto[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [roleFilter, setRoleFilter] = useState<UserRole | ''>('')
  const [loading, setLoading] = useState(true)
  const [showAddModal, setShowAddModal] = useState(false)

  // Add user form state
  const [form, setForm] = useState<CreateUserRequest>({
    fullName: '',
    email: '',
    password: '',
    role: 'Patient',
    specialityName: '',
    specialityNameAr: '',
    consultationFee: 0,
    yearsExperience: 0,
    bio: '',
  })
  const [formErrors, setFormErrors] = useState<Partial<Record<keyof CreateUserRequest, string>>>({})
  const [addLoading, setAddLoading] = useState(false)

  const fetchUsers = useCallback(async (p = 1, q = search, role = roleFilter) => {
    setLoading(true)
    try {
      const res = await adminApi.getUsers({ page: p, pageSize: PAGE_SIZE, search: q || undefined, role: role || undefined })
      setUsers(res.items)
      setTotal(res.total)
    } catch {
      toast.error('فشل تحميل المستخدمين')
    } finally {
      setLoading(false)
    }
  }, [search, roleFilter])

  useEffect(() => { fetchUsers() }, [fetchUsers])

  const debouncedSearch = useCallback(
    debounce((val: string) => { setPage(1); fetchUsers(1, val, roleFilter) }, 400),
    [roleFilter, fetchUsers]
  )

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearch(e.target.value)
    debouncedSearch(e.target.value)
  }

  const handleRoleFilter = (role: UserRole | '') => {
    setRoleFilter(role)
    setPage(1)
    fetchUsers(1, search, role)
  }

  const handleToggle = async (id: string) => {
    try {
      await adminApi.toggleUser(id)
      setUsers((prev) => prev.map((u) => u.userId === id ? { ...u, isActive: !u.isActive } : u))
      toast.success('تم تحديث حالة المستخدم')
    } catch { toast.error('فشلت العملية') }
  }

  const handleDelete = async (id: string) => {
    try {
      await adminApi.deleteUser(id)
      setUsers((prev) => prev.filter((u) => u.userId !== id))
      setTotal((t) => t - 1)
      toast.success('تم حذف المستخدم')
    } catch { toast.error('فشل الحذف') }
  }

  // Validate add user form
  const validateForm = (): boolean => {
    const errors: typeof formErrors = {}
    if (!form.fullName.trim()) errors.fullName = 'الاسم مطلوب'
    if (!form.email.trim()) errors.email = 'البريد الإلكتروني مطلوب'
    else if (!/\S+@\S+\.\S+/.test(form.email)) errors.email = 'بريد إلكتروني غير صحيح'
    if (!form.password || form.password.length < 8) errors.password = 'كلمة المرور يجب أن تكون 8 أحرف على الأقل'
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
      setUsers((prev) => [newUser, ...prev])
      setTotal((t) => t + 1)
      toast.success(`تم إنشاء حساب "${newUser.fullName}" بنجاح`)
      setShowAddModal(false)
      setForm({ fullName: '', email: '', password: '', role: 'Patient' })
      setFormErrors({})
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { message?: string } } })?.response?.data?.message || 'فشل إنشاء المستخدم'
      toast.error(msg)
    } finally {
      setAddLoading(false)
    }
  }

  const setField = (key: keyof CreateUserRequest, val: unknown) => {
    setForm((p) => ({ ...p, [key]: val }))
    setFormErrors((p) => ({ ...p, [key]: undefined }))
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-800">إدارة المستخدمين</h1>
          <p className="text-sm text-gray-500 mt-0.5">إجمالي {total} مستخدم مسجل</p>
        </div>
        <Button
          variant="primary"
          icon={<UserPlus size={16} />}
          onClick={() => setShowAddModal(true)}
        >
          إضافة مستخدم
        </Button>
      </div>

      <Card padding="none">
        {/* Filters */}
        <div className="flex flex-wrap items-center gap-3 p-4 border-b border-gray-100">
          <div className="relative flex-1 min-w-48">
            <Search size={15} className="absolute top-1/2 -translate-y-1/2 right-3 text-gray-400" />
            <input
              type="text"
              value={search}
              onChange={handleSearch}
              placeholder="بحث بالاسم أو الإيميل..."
              className="w-full pr-9 pl-4 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 focus:border-primary-400"
            />
          </div>
          <div className="flex items-center gap-1">
            <Filter size={14} className="text-gray-400" />
            {(['', 'Admin', 'Doctor', 'Patient'] as const).map((role) => (
              <button
                key={role}
                onClick={() => handleRoleFilter(role)}
                className={`px-3 py-1.5 text-xs rounded-xl font-medium transition-colors ${
                  roleFilter === role
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {role === '' ? 'الكل' : role === 'Admin' ? 'مدراء' : role === 'Doctor' ? 'أطباء' : 'مرضى'}
              </button>
            ))}
          </div>
        </div>

        {loading ? <PageLoader /> : (
          <>
            <UserTable users={users} onToggle={handleToggle} onDelete={handleDelete} />
            <Pagination total={total} page={page} pageSize={PAGE_SIZE} onChange={(p) => { setPage(p); fetchUsers(p) }} />
          </>
        )}
      </Card>

      {/* Add User Modal - Admin Only */}
      <Modal
        open={showAddModal}
        onClose={() => { setShowAddModal(false); setFormErrors({}) }}
        title="إضافة مستخدم جديد"
        size="lg"
        footer={
          <>
            <Button variant="outline" onClick={() => { setShowAddModal(false); setFormErrors({}) }}>إلغاء</Button>
            <Button variant="primary" onClick={handleAddUser} loading={addLoading}>إنشاء الحساب</Button>
          </>
        }
      >
        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-100 rounded-xl p-3 text-xs text-blue-700 flex gap-2">
            <span>ℹ️</span>
            <span>فقط مدير النظام يمكنه إضافة مستخدمين جدد للمنصة. يتلقى المستخدم بيانات الدخول عبر البريد الإلكتروني.</span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* Full Name */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">الاسم الكامل *</label>
              <input
                type="text"
                value={form.fullName}
                onChange={(e) => setField('fullName', e.target.value)}
                placeholder="د. أحمد محمد"
                className={`w-full px-3 py-2 text-sm border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 ${formErrors.fullName ? 'border-red-300 bg-red-50' : 'border-gray-200'}`}
              />
              {formErrors.fullName && <p className="mt-1 text-xs text-red-500">{formErrors.fullName}</p>}
            </div>

            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">البريد الإلكتروني *</label>
              <input
                type="email"
                value={form.email}
                onChange={(e) => setField('email', e.target.value)}
                placeholder="user@medbook.com"
                className={`w-full px-3 py-2 text-sm border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 ${formErrors.email ? 'border-red-300 bg-red-50' : 'border-gray-200'}`}
              />
              {formErrors.email && <p className="mt-1 text-xs text-red-500">{formErrors.email}</p>}
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">كلمة المرور *</label>
              <input
                type="password"
                value={form.password}
                onChange={(e) => setField('password', e.target.value)}
                placeholder="8 أحرف على الأقل"
                className={`w-full px-3 py-2 text-sm border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 ${formErrors.password ? 'border-red-300 bg-red-50' : 'border-gray-200'}`}
              />
              {formErrors.password && <p className="mt-1 text-xs text-red-500">{formErrors.password}</p>}
            </div>

            {/* Role */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">الدور *</label>
              <select
                value={form.role}
                onChange={(e) => setField('role', e.target.value as UserRole)}
                className="w-full px-3 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30"
              >
                <option value="Patient">مريض</option>
                <option value="Doctor">طبيب</option>
                <option value="Admin">مدير</option>
              </select>
            </div>
          </div>

          {/* Doctor-specific fields */}
          {form.role === 'Doctor' && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="border-t border-gray-100 pt-4 space-y-4"
            >
              <p className="text-sm font-semibold text-gray-700">بيانات الطبيب</p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">التخصص (إنجليزي) *</label>
                  <input
                    type="text"
                    value={form.specialityName}
                    onChange={(e) => setField('specialityName', e.target.value)}
                    placeholder="Cardiology"
                    className={`w-full px-3 py-2 text-sm border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 ${formErrors.specialityName ? 'border-red-300 bg-red-50' : 'border-gray-200'}`}
                  />
                  {formErrors.specialityName && <p className="mt-1 text-xs text-red-500">{formErrors.specialityName}</p>}
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">التخصص (عربي) *</label>
                  <input
                    type="text"
                    value={form.specialityNameAr}
                    onChange={(e) => setField('specialityNameAr', e.target.value)}
                    placeholder="أمراض القلب"
                    className={`w-full px-3 py-2 text-sm border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 ${formErrors.specialityNameAr ? 'border-red-300 bg-red-50' : 'border-gray-200'}`}
                  />
                  {formErrors.specialityNameAr && <p className="mt-1 text-xs text-red-500">{formErrors.specialityNameAr}</p>}
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">سنوات الخبرة</label>
                  <input
                    type="number"
                    min={0}
                    value={form.yearsExperience}
                    onChange={(e) => setField('yearsExperience', Number(e.target.value))}
                    className="w-full px-3 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">رسوم الاستشارة (ج.م)</label>
                  <input
                    type="number"
                    min={0}
                    value={form.consultationFee}
                    onChange={(e) => setField('consultationFee', Number(e.target.value))}
                    className="w-full px-3 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">نبذة تعريفية</label>
                <textarea
                  rows={3}
                  value={form.bio}
                  onChange={(e) => setField('bio', e.target.value)}
                  placeholder="اكتب نبذة عن الطبيب..."
                  className="w-full px-3 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 resize-none"
                />
              </div>
            </motion.div>
          )}
        </div>
      </Modal>
    </div>
  )
}
