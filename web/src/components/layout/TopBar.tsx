import { useEffect, useState } from 'react'
import { Bell, Search, X, Sun, Moon } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNotificationStore } from '@/store/notificationStore'
import { useAuthStore } from '@/store/authStore'
import { useThemeStore } from '@/store/themeStore'
import { cn } from '@/lib/utils'
import { formatTimeAgo } from '@/lib/utils'

interface TopBarProps {
  title?: string
}

export function TopBar({ title }: TopBarProps) {
  const [showNotif, setShowNotif] = useState(false)
  const { notifications, unreadCount, markAllRead, removeNotification } = useNotificationStore()
  const { user } = useAuthStore()
  const { isDark, toggleTheme } = useThemeStore()

  const bg = isDark ? 'rgba(15,23,42,0.85)' : '#ffffff'
  const border = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)'
  const inputBg = isDark ? 'rgba(255,255,255,0.06)' : '#f9fafb'
  const inputBorder = isDark ? 'rgba(255,255,255,0.12)' : '#e5e7eb'
  const inputText = isDark ? 'rgba(255,255,255,0.8)' : '#374151'
  const textMain = isDark ? '#ffffff' : '#111827'
  const textMuted = isDark ? 'rgba(148,163,184,0.8)' : '#6b7280'
  const bellHover = isDark ? 'rgba(255,255,255,0.08)' : '#f3f4f6'
  const notifBg = isDark ? 'rgba(15,23,42,0.98)' : '#ffffff'
  const notifBorder = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)'
  const notifItemHover = isDark ? 'rgba(255,255,255,0.04)' : '#f9fafb'
  const divider = isDark ? 'rgba(255,255,255,0.05)' : '#f3f4f6'

  return (
    <header
      className="fixed top-0 right-0 left-64 h-16 flex items-center justify-between px-6 z-20"
      style={{
        background: isDark ? 'rgba(15,23,42,0.85)' : '#ffffff',
        borderBottom: `1px solid ${border}`,
        backdropFilter: isDark ? 'blur(20px)' : 'none',
        boxShadow: isDark ? '0 1px 20px rgba(0,0,0,0.3)' : '0 1px 8px rgba(0,0,0,0.06)',
      }}
    >
      <div className="flex items-center gap-4">
        {title && <h1 className="text-base font-semibold" style={{ color: textMain }}>{title}</h1>}
      </div>

      <div className="flex items-center gap-4">
        {/* Search */}
        <div className="relative hidden md:block">
          <Search size={15} className="absolute top-1/2 -translate-y-1/2 left-3" style={{ color: textMuted }} />
          <input
            type="text"
            placeholder="Search..."
            className="pl-9 pr-4 py-1.5 text-sm rounded-xl w-60 focus:outline-none focus:ring-2 focus:ring-indigo-500/30 transition-all"
            style={{
              background: inputBg,
              border: `1px solid ${inputBorder}`,
              color: inputText,
            }}
          />
        </div>

        {/* Theme Toggle */}
        <motion.button
          id="global-theme-toggle"
          whileHover={{ scale: 1.06 }}
          whileTap={{ scale: 0.94 }}
          onClick={toggleTheme}
          className="flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all"
          style={{
            background: isDark ? 'rgba(99,102,241,0.15)' : '#eef2ff',
            border: `1px solid ${isDark ? 'rgba(99,102,241,0.3)' : 'rgba(99,102,241,0.2)'}`,
            color: isDark ? '#818cf8' : '#6366f1',
          }}
          title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          <AnimatePresence mode="wait">
            {isDark ? (
              <motion.span key="sun" initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.2 }}>
                <Sun size={14} />
              </motion.span>
            ) : (
              <motion.span key="moon" initial={{ rotate: 90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: -90, opacity: 0 }} transition={{ duration: 0.2 }}>
                <Moon size={14} />
              </motion.span>
            )}
          </AnimatePresence>
          <span className="hidden sm:inline">{isDark ? 'Light' : 'Dark'}</span>
        </motion.button>

        {/* Notification Bell */}
        <div className="relative">
          <button
            onClick={() => { setShowNotif(!showNotif); if (!showNotif) markAllRead() }}
            className="relative p-2 rounded-xl transition-colors"
            style={{ color: textMuted }}
            onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = bellHover}
            onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = 'transparent'}
          >
            <Bell size={18} />
            {unreadCount > 0 && (
              <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-red-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center">
                {unreadCount > 9 ? '9+' : unreadCount}
              </span>
            )}
          </button>

          <AnimatePresence>
            {showNotif && (
              <motion.div
                initial={{ opacity: 0, y: 8, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 8, scale: 0.95 }}
                className="absolute right-0 top-full mt-2 w-80 rounded-2xl overflow-hidden z-50"
                style={{
                  background: notifBg,
                  border: `1px solid ${notifBorder}`,
                  backdropFilter: isDark ? 'blur(20px)' : 'none',
                  boxShadow: isDark ? '0 20px 60px rgba(0,0,0,0.5)' : '0 8px 32px rgba(0,0,0,0.12)',
                }}
              >
                <div className="px-4 py-3 flex items-center justify-between" style={{ borderBottom: `1px solid ${divider}` }}>
                  <span className="font-semibold text-sm" style={{ color: textMain }}>Notifications</span>
                  <button onClick={() => setShowNotif(false)} style={{ color: textMuted }}>
                    <X size={14} />
                  </button>
                </div>
                <div className="max-h-72 overflow-y-auto">
                  {notifications.length === 0 ? (
                    <p className="text-center text-sm py-8" style={{ color: textMuted }}>No notifications</p>
                  ) : notifications.map((n) => (
                    <div
                      key={n.id}
                      onClick={() => removeNotification(n.id)}
                      className="w-full text-left flex items-start gap-3 p-3 cursor-pointer transition-colors"
                      style={{ borderBottom: `1px solid ${divider}` }}
                      onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = notifItemHover}
                      onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = 'transparent'}
                    >
                      <div className={cn(
                        'w-2 h-2 rounded-full mt-1.5 flex-shrink-0',
                        n.type === 'success' ? 'bg-green-500' :
                        n.type === 'error' ? 'bg-red-500' :
                        n.type === 'warning' ? 'bg-amber-500' : 'bg-blue-500'
                      )} />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium" style={{ color: textMain }}>{n.title}</p>
                        {n.message && <p className="text-xs mt-0.5" style={{ color: textMuted }}>{n.message}</p>}
                        <p className="text-xs mt-1" style={{ color: textMuted }}>{formatTimeAgo(n.createdAt)}</p>
                      </div>
                      <button
                        onClick={(e) => { e.stopPropagation(); removeNotification(n.id) }}
                        style={{ color: textMuted }}
                      >
                        <X size={12} />
                      </button>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* User Profile */}
        <div className="flex items-center gap-3 pl-4 ml-1" style={{ borderLeft: `1px solid ${border}` }}>
          <div className="flex flex-col items-end hidden sm:flex">
            <span className="text-xs font-bold leading-none" style={{ color: textMain }}>{user?.name}</span>
            <span className="text-[10px] font-medium mt-1 uppercase tracking-wider" style={{ color: textMuted }}>{user?.role}</span>
          </div>
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center text-white text-xs font-black shadow-lg overflow-hidden ring-2 ring-indigo-500/30">
            {user?.photoUrl ? (
              <img src={user.photoUrl} alt={user.name} className="w-full h-full object-cover" />
            ) : (
              user?.name?.charAt(0) ?? 'U'
            )}
          </div>
        </div>
      </div>
    </header>
  )
}
