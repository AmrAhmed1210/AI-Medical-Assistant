import { NavLink, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  LayoutDashboard, Calendar, Users, FileText, MessageSquare,
  Clock, User, BarChart2, Settings, LogOut, Shield,
  Cpu, TrendingUp,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useAuthStore } from '@/store/authStore'
import { authApi } from '@/api/authApi'
import { ROUTES } from '@/constants/config'
import toast from 'react-hot-toast'

interface NavItem {
  to: string
  icon: React.ReactNode
  label: string
}

const doctorNav: NavItem[] = [
  { to: ROUTES.DOCTOR_DASHBOARD, icon: <LayoutDashboard size={18} />, label: 'لوحة التحكم' },
  { to: ROUTES.DOCTOR_APPOINTMENTS, icon: <Calendar size={18} />, label: 'المواعيد' },
  { to: ROUTES.DOCTOR_PATIENTS, icon: <Users size={18} />, label: 'المرضى' },
  { to: ROUTES.DOCTOR_REPORTS, icon: <FileText size={18} />, label: 'تقارير AI' },
  { to: ROUTES.DOCTOR_CHAT, icon: <MessageSquare size={18} />, label: 'المحادثات' },
  { to: ROUTES.DOCTOR_SCHEDULE, icon: <Clock size={18} />, label: 'الجدول الزمني' },
  { to: ROUTES.DOCTOR_PROFILE, icon: <User size={18} />, label: 'ملفي الشخصي' },
]

const adminNav: NavItem[] = [
  { to: ROUTES.ADMIN_DASHBOARD, icon: <LayoutDashboard size={18} />, label: 'لوحة التحكم' },
  { to: ROUTES.ADMIN_USERS, icon: <Users size={18} />, label: 'المستخدمون' },
  { to: ROUTES.ADMIN_STATISTICS, icon: <TrendingUp size={18} />, label: 'الإحصائيات' },
  { to: ROUTES.ADMIN_MODELS, icon: <Cpu size={18} />, label: 'نماذج AI' },
]

export function Sidebar() {
  const { user, role, logout } = useAuthStore()
  const navigate = useNavigate()

  const navItems = role === 'Admin' ? adminNav : doctorNav

  const handleLogout = async () => {
    try { await authApi.logout() } catch { /* silent */ }
    logout()
    navigate(ROUTES.LOGIN)
    toast.success('تم تسجيل الخروج')
  }

  return (
    <motion.aside
      initial={{ x: 256 }}
      animate={{ x: 0 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      className="fixed top-0 right-0 h-screen w-64 bg-white border-l border-gray-100 flex flex-col z-30 shadow-sm"
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-gray-100">
        <div className="w-9 h-9 bg-primary-600 rounded-xl flex items-center justify-center shadow-sm">
          <span className="text-white text-base font-bold">M</span>
        </div>
        <div>
          <p className="text-base font-bold text-gray-800">MedBook</p>
          <p className="text-xs text-gray-400">منصة طبية ذكية</p>
        </div>
      </div>

      {/* Role badge */}
      <div className="px-4 py-3">
        <div className={cn(
          'flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium',
          role === 'Admin' ? 'bg-purple-50 text-purple-700' : 'bg-primary-50 text-primary-700'
        )}>
          {role === 'Admin' ? <Shield size={14} /> : <User size={14} />}
          {role === 'Admin' ? 'مدير النظام' : 'طبيب'}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-2 overflow-y-auto space-y-0.5">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => cn(
              'flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all',
              isActive
                ? 'bg-primary-600 text-white shadow-sm'
                : 'text-gray-600 hover:bg-gray-50 hover:text-gray-800'
            )}
          >
            {item.icon}
            {item.label}
          </NavLink>
        ))}
      </nav>

      {/* User info + logout */}
      <div className="border-t border-gray-100 p-3">
        <div className="flex items-center gap-3 px-2 py-2 mb-1">
          <div className="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
            <span className="text-primary-700 text-xs font-semibold">
              {user?.fullName?.charAt(0) || 'U'}
            </span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-800 truncate">{user?.fullName}</p>
            <p className="text-xs text-gray-400 truncate">{user?.email}</p>
          </div>
        </div>
        <button
          onClick={handleLogout}
          className="w-full flex items-center gap-3 px-3 py-2 text-sm text-red-500 hover:bg-red-50 rounded-xl transition-colors font-medium"
        >
          <LogOut size={16} />
          تسجيل الخروج
        </button>
      </div>
    </motion.aside>
  )
}
