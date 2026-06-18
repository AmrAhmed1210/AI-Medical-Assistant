import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useEffect } from 'react'
import {
  LayoutDashboard, Calendar, Users, FileText, MessageSquare,
  Clock, User, BarChart2, Settings, LogOut, Shield,
  Cpu, TrendingUp, Star, LifeBuoy, Stethoscope, CheckSquare
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useAuthStore } from '@/store/authStore'
import { authApi } from '@/api/authApi'
import { ROUTES } from '@/constants/config'
import toast from 'react-hot-toast'
import { useNotificationStore } from '@/store/notificationStore'
import { useThemeStore } from '@/store/themeStore'

interface NavItem {
  to: string
  icon: React.ReactNode
  label: string
}

const doctorNav: NavItem[] = [
  { to: ROUTES.DOCTOR_DASHBOARD, icon: <LayoutDashboard size={18} />, label: 'Dashboard' },
  { to: ROUTES.DOCTOR_TODAY, icon: <CheckSquare size={18} />, label: "Today's Visits" },
  { to: ROUTES.DOCTOR_APPOINTMENTS, icon: <Calendar size={18} />, label: 'Appointments' },
  { to: ROUTES.DOCTOR_PATIENTS, icon: <Users size={18} />, label: 'Patients' },
  { to: ROUTES.DOCTOR_REVIEWS, icon: <Star size={18} />, label: 'Reviews' },
  { to: ROUTES.DOCTOR_CHAT, icon: <MessageSquare size={18} />, label: 'Messages' },
  { to: '/doctor/staff', icon: <Users size={18} />, label: 'Manage Staff' },
  { to: ROUTES.DOCTOR_SCHEDULE, icon: <Clock size={18} />, label: 'Schedule' },
  { to: ROUTES.DOCTOR_PROFILE, icon: <User size={18} />, label: 'My Profile' },
]

const adminNav: NavItem[] = [
  { to: ROUTES.ADMIN_DASHBOARD, icon: <LayoutDashboard size={18} />, label: 'Dashboard' },
  { to: ROUTES.ADMIN_USERS, icon: <Users size={18} />, label: 'Users' },
  { to: ROUTES.ADMIN_STATISTICS, icon: <TrendingUp size={18} />, label: 'Statistics' },

  { to: ROUTES.ADMIN_APPLICATIONS, icon: <FileText size={18} />, label: 'Applications' },
  { to: ROUTES.ADMIN_SUPPORT, icon: <LifeBuoy size={18} />, label: 'Support Center' },
]

const secretaryNav: NavItem[] = [
  { to: '/secretary/dashboard', icon: <Calendar size={18} />, label: 'Appointments' },
]

export function Sidebar() {
  const { user, role, logout } = useAuthStore()
  const navigate = useNavigate()
  const location = useLocation()
  const { isDark } = useThemeStore()
  const {
    unreadMessages,
    unreadAppointments,
    unreadDoctorApplications,
    clearAllMessages,
    clearAppointments,
    clearDoctorApplications,
  } = useNotificationStore()

  const lowerRole = role?.toLowerCase()
  const navItems = lowerRole === 'admin' ? adminNav : (lowerRole === 'secretary' ? secretaryNav : doctorNav)

  const handleLogout = async () => {
    try {
      await authApi.logout()
    } catch { /* silent */ }
    localStorage.clear()
    sessionStorage.clear()
    logout()
    navigate(ROUTES.LOGIN, { replace: true })
    toast.success('Logged out successfully')
    window.location.href = ROUTES.LOGIN
  }

  useEffect(() => {
    if (location.pathname === ROUTES.DOCTOR_CHAT) clearAllMessages()
    if (location.pathname === ROUTES.DOCTOR_APPOINTMENTS || location.pathname === ROUTES.DOCTOR_TODAY) clearAppointments()
    if (location.pathname === ROUTES.ADMIN_APPLICATIONS) clearDoctorApplications()
    if (location.pathname === ROUTES.ADMIN_SUPPORT) clearAllMessages()
  }, [location.pathname, clearAppointments, clearAllMessages, clearDoctorApplications])

  return (
    <motion.aside
      initial={{ x: -256 }}
      animate={{ x: 0 }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      className="fixed top-4 left-4 h-[calc(100vh-2rem)] w-64 flex flex-col z-30 rounded-3xl overflow-hidden glass-card dark:bg-slate-900/80 dark:border-slate-800 shadow-xl"
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-6 py-6 border-b border-slate-100 dark:border-slate-800/80">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center shadow-glow-primary bg-gradient-primary">
          <span className="text-white text-lg font-black font-outfit">M</span>
        </div>
        <div>
          <p className="text-lg font-black text-slate-900 dark:text-white font-outfit tracking-tight">MedBook</p>
          <p className="text-xs text-slate-500 dark:text-slate-400">Smart Medical</p>
        </div>
      </div>

      {/* Role badge */}
      <div className="px-5 py-4">
        <div className="flex items-center gap-2 px-3 py-2.5 rounded-xl text-xs font-semibold bg-primary-50 text-primary-600 dark:bg-primary-950/30 dark:text-primary-400 border border-primary-100 dark:border-primary-900/50">
          {lowerRole === 'admin' ? <Shield size={14} /> : lowerRole === 'secretary' ? <Users size={14} /> : <Stethoscope size={14} />}
          {lowerRole === 'admin' ? 'System Administrator' : lowerRole === 'secretary' ? 'Secretary' : 'Doctor'}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-2 overflow-y-auto space-y-1 scrollbar-hide">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => cn(
              "flex items-center gap-3 px-4 py-3 rounded-2xl text-sm font-semibold transition-all duration-300 relative group",
              isActive 
                ? "bg-gradient-primary text-white shadow-lg shadow-primary-500/30" 
                : "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-slate-100"
            )}
          >
            {({ isActive }) => (
              <>
                <div className="relative z-10">
                  {item.icon}
                  {item.to === ROUTES.DOCTOR_CHAT && unreadMessages > 0 && (
                    <span className="absolute -top-1 -right-2 min-w-[18px] h-[18px] flex items-center justify-center bg-emerald-500 rounded-full text-white text-[10px] font-bold shadow-sm ring-2 ring-white dark:ring-slate-900">
                      {unreadMessages > 9 ? '9+' : unreadMessages}
                    </span>
                  )}
                  {item.to === ROUTES.DOCTOR_APPOINTMENTS && unreadAppointments > 0 && (
                    <span className="absolute -top-1 -right-2 min-w-[18px] h-[18px] flex items-center justify-center bg-red-500 rounded-full text-white text-[10px] font-bold shadow-sm animate-pulse ring-2 ring-white dark:ring-slate-900">
                      {unreadAppointments > 9 ? '9+' : unreadAppointments}
                    </span>
                  )}
                  {item.to === ROUTES.ADMIN_APPLICATIONS && unreadDoctorApplications > 0 && (
                    <span className="absolute -top-1 -right-2 min-w-[18px] h-[18px] flex items-center justify-center bg-amber-500 rounded-full text-white text-[10px] font-bold shadow-sm animate-pulse ring-2 ring-white dark:ring-slate-900">
                      {unreadDoctorApplications > 9 ? '9+' : unreadDoctorApplications}
                    </span>
                  )}
                </div>
                <span className="z-10">{item.label}</span>
                
                {/* Hover Effect Background */}
                {!isActive && (
                  <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-primary-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* User info + logout */}
      <div className="p-4 border-t border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-900/50">
        <div className="flex items-center gap-3 px-2 py-2 mb-2 rounded-xl">
          <div className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 bg-primary-100 dark:bg-primary-900/40">
            <span className="text-sm font-bold text-primary-600 dark:text-primary-400">
              {user?.name?.charAt(0) || 'U'}
            </span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-bold truncate text-slate-800 dark:text-slate-200">{user?.name}</p>
            <p className="text-xs truncate text-slate-500 dark:text-slate-400 font-medium">{user?.email}</p>
          </div>
        </div>
        <button
          onClick={handleLogout}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-bold rounded-xl transition-all duration-300 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30"
        >
          <LogOut size={16} />
          Log Out
        </button>
      </div>
    </motion.aside>
  )
}
