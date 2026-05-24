import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useEffect } from 'react'
import {
  LayoutDashboard, Calendar, Users, FileText, MessageSquare,
  Clock, User, BarChart2, Settings, LogOut, Shield,
  Cpu, TrendingUp, Star, LifeBuoy, Stethoscope,
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

  // Theme tokens for sidebar
  const bg = isDark ? 'rgba(10,15,35,0.98)' : '#ffffff'
  const border = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)'
  const logoText = isDark ? '#ffffff' : '#111827'
  const logoSub = isDark ? 'rgba(148,163,184,0.7)' : '#9ca3af'
  const navTextDefault = isDark ? 'rgba(148,163,184,0.85)' : '#4b5563'
  const navHoverBg = isDark ? 'rgba(99,102,241,0.1)' : '#f3f4f6'
  const navHoverText = isDark ? '#ffffff' : '#111827'
  const userNameColor = isDark ? '#ffffff' : '#111827'
  const userEmailColor = isDark ? 'rgba(148,163,184,0.7)' : '#9ca3af'
  const avatarBg = isDark ? 'rgba(99,102,241,0.25)' : '#eef2ff'
  const avatarText = isDark ? '#818cf8' : '#6366f1'

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
      className="fixed top-0 left-0 h-screen w-64 flex flex-col z-30"
      style={{
        background: bg,
        borderRight: `1px solid ${border}`,
        backdropFilter: isDark ? 'blur(24px)' : 'none',
        boxShadow: isDark ? '4px 0 24px rgba(0,0,0,0.4)' : '2px 0 12px rgba(0,0,0,0.06)',
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-5" style={{ borderBottom: `1px solid ${border}` }}>
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center shadow-lg"
          style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', boxShadow: '0 4px 16px rgba(99,102,241,0.4)' }}
        >
          <span className="text-white text-base font-black">M</span>
        </div>
        <div>
          <p className="text-base font-black" style={{ color: logoText }}>MedBook</p>
          <p className="text-xs" style={{ color: logoSub }}>Smart Medical Platform</p>
        </div>
      </div>

      {/* Role badge */}
      <div className="px-4 py-3">
        <div
          className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold"
          style={{
            background: isDark ? 'rgba(99,102,241,0.12)' : '#eef2ff',
            color: isDark ? '#818cf8' : '#6366f1',
            border: `1px solid ${isDark ? 'rgba(99,102,241,0.25)' : 'rgba(99,102,241,0.2)'}`,
          }}
        >
          {lowerRole === 'admin' ? <Shield size={14} /> : lowerRole === 'secretary' ? <Users size={14} /> : <Stethoscope size={14} />}
          {lowerRole === 'admin' ? 'System Administrator' : lowerRole === 'secretary' ? 'Secretary' : 'Doctor'}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-2 overflow-y-auto space-y-0.5">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200"
            style={({ isActive }) => isActive
              ? {
                  background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                  color: '#ffffff',
                  boxShadow: '0 4px 16px rgba(99,102,241,0.35)',
                }
              : { color: navTextDefault }
            }
            onMouseEnter={e => {
              const el = e.currentTarget as HTMLElement
              if (!el.style.boxShadow) {
                el.style.background = navHoverBg
                el.style.color = navHoverText
              }
            }}
            onMouseLeave={e => {
              const el = e.currentTarget as HTMLElement
              if (!el.style.boxShadow) {
                el.style.background = 'transparent'
                el.style.color = navTextDefault
              }
            }}
          >
            <div className="relative">
              {item.icon}
              {item.to === ROUTES.DOCTOR_CHAT && unreadMessages > 0 && (
                <span className="absolute -top-1 -right-2 min-w-[18px] h-[18px] flex items-center justify-center bg-green-500 rounded-full text-white text-[10px] font-bold shadow-sm">
                  {unreadMessages > 9 ? '9+' : unreadMessages}
                </span>
              )}
              {item.to === ROUTES.DOCTOR_APPOINTMENTS && unreadAppointments > 0 && (
                <span className="absolute -top-1 -right-2 min-w-[18px] h-[18px] flex items-center justify-center bg-red-500 rounded-full text-white text-[10px] font-bold shadow-sm animate-pulse">
                  {unreadAppointments > 9 ? '9+' : unreadAppointments}
                </span>
              )}
              {item.to === ROUTES.ADMIN_APPLICATIONS && unreadDoctorApplications > 0 && (
                <span className="absolute -top-1 -right-2 min-w-[18px] h-[18px] flex items-center justify-center bg-amber-500 rounded-full text-white text-[10px] font-bold shadow-sm animate-pulse">
                  {unreadDoctorApplications > 9 ? '9+' : unreadDoctorApplications}
                </span>
              )}
            </div>
            {item.label}
          </NavLink>
        ))}
      </nav>

      {/* User info + logout */}
      <div className="p-3" style={{ borderTop: `1px solid ${border}` }}>
        <div className="flex items-center gap-3 px-2 py-2 mb-1">
          <div
            className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
            style={{ background: avatarBg }}
          >
            <span className="text-xs font-bold" style={{ color: avatarText }}>
              {user?.name?.charAt(0) || 'U'}
            </span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold truncate" style={{ color: userNameColor }}>{user?.name}</p>
            <p className="text-xs truncate" style={{ color: userEmailColor }}>{user?.email}</p>
          </div>
        </div>
        <button
          onClick={handleLogout}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium rounded-xl transition-colors"
          style={{ color: '#ef4444' }}
          onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = isDark ? 'rgba(239,68,68,0.1)' : '#fef2f2'}
          onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = 'transparent'}
        >
          <LogOut size={16} />
          Log Out
        </button>
      </div>
    </motion.aside>
  )
}
