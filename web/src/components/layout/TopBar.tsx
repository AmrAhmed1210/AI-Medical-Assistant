import { useEffect, useState } from 'react'
import { Bell, Search, X, Sun, Moon } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNotificationStore } from '@/store/notificationStore'
import { useAuthStore } from '@/store/authStore'
import { useThemeStore } from '@/store/themeStore'
import { cn } from '@/lib/utils'
import { formatTimeAgo } from '@/lib/utils'
import { useLanguage } from '@/lib/language'

interface TopBarProps {
  title?: string
}

export function TopBar({ title }: TopBarProps) {
  const [showNotif, setShowNotif] = useState(false)
  const { notifications, unreadCount, markAllRead, removeNotification } = useNotificationStore()
  const { user } = useAuthStore()
  const { isDark, toggleTheme } = useThemeStore()
  const { isRTL, toggleLanguage, t } = useLanguage()

  return (
    <header className="sticky top-0 z-20 h-16 flex items-center justify-between px-6 glass-card border-b-0 rounded-none dark:bg-slate-900/80 shadow-sm backdrop-blur-xl">
      <div className="flex items-center gap-4">
        {title && <h1 className="text-lg font-bold text-slate-800 dark:text-slate-100">{title}</h1>}
      </div>

      <div className="flex items-center gap-4">
        {/* Search */}
        <div className="relative hidden md:block group">
          <Search size={15} className="absolute top-1/2 -translate-y-1/2 left-3 text-slate-400 group-focus-within:text-primary-500 transition-colors" />
          <input
            type="text"
            placeholder={t('search')}
            className={`${isRTL ? 'pr-9 pl-4 text-right' : 'pl-9 pr-4'} py-2 text-sm rounded-xl w-64 focus:outline-none focus:ring-2 focus:ring-primary-500/30 transition-all bg-slate-100 dark:bg-slate-800/50 border border-transparent dark:border-slate-700/50 text-slate-700 dark:text-slate-200`}
          />
        </div>

        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={toggleLanguage}
          className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-bold transition-all bg-slate-100 text-slate-700 dark:bg-slate-800/60 dark:text-slate-200 border border-slate-200 dark:border-slate-700/50"
        >
          {t('language')}
        </motion.button>

        {/* Theme Toggle */}
        <motion.button
          id="global-theme-toggle"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={toggleTheme}
          className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-bold transition-all bg-primary-50 text-primary-600 dark:bg-primary-900/30 dark:text-primary-400 border border-primary-100 dark:border-primary-800/50 hover:bg-primary-100 dark:hover:bg-primary-800/50"
          title={isDark ? t('switchToLight') : t('switchToDark')}
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
          <span className="hidden sm:inline">{isDark ? t('light') : t('dark')}</span>
        </motion.button>

        {/* Notification Bell */}
        <div className="relative">
          <button
            onClick={() => { setShowNotif(!showNotif); if (!showNotif) markAllRead() }}
            className="relative p-2 rounded-xl transition-colors text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            <Bell size={18} />
            {unreadCount > 0 && (
              <span className="absolute top-0 right-0 w-4 h-4 bg-red-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center ring-2 ring-white dark:ring-slate-900">
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
                className={`absolute ${isRTL ? 'left-0' : 'right-0'} top-full mt-3 w-80 rounded-2xl overflow-hidden z-50 glass-card dark:bg-slate-900 shadow-2xl dark:shadow-none border border-slate-200 dark:border-slate-800`}
              >
                <div className="px-4 py-3 flex items-center justify-between border-b border-slate-100 dark:border-slate-800">
                  <span className="font-bold text-sm text-slate-800 dark:text-slate-200">{t('notifications')}</span>
                  <button onClick={() => setShowNotif(false)} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors">
                    <X size={14} />
                  </button>
                </div>
                <div className="max-h-72 overflow-y-auto scrollbar-hide">
                  {notifications.length === 0 ? (
                    <p className="text-center text-sm py-8 text-slate-400 dark:text-slate-500">{t('noNotifications')}</p>
                  ) : notifications.map((n) => (
                    <div
                      key={n.id}
                      onClick={() => removeNotification(n.id)}
                      className={`w-full ${isRTL ? 'text-right' : 'text-left'} flex items-start gap-3 p-3 cursor-pointer transition-colors border-b border-slate-50 dark:border-slate-800/50 hover:bg-slate-50 dark:hover:bg-slate-800/50`}
                    >
                      <div className={cn(
                        'w-2 h-2 rounded-full mt-1.5 flex-shrink-0',
                        n.type === 'success' ? 'bg-emerald-500 shadow-glow-primary' :
                        n.type === 'error' ? 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.4)]' :
                        n.type === 'warning' ? 'bg-amber-500' : 'bg-blue-500'
                      )} />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">{n.title}</p>
                        {n.message && <p className="text-xs mt-0.5 text-slate-500 dark:text-slate-400">{n.message}</p>}
                        <p className="text-[10px] mt-1 text-slate-400 dark:text-slate-500 font-medium">{formatTimeAgo(n.createdAt)}</p>
                      </div>
                      <button
                        onClick={(e) => { e.stopPropagation(); removeNotification(n.id) }}
                        className="text-slate-300 dark:text-slate-600 hover:text-slate-500 dark:hover:text-slate-400"
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
        <div className={`flex items-center gap-3 ${isRTL ? 'pr-5 mr-2 border-r' : 'pl-5 ml-2 border-l'} border-slate-200 dark:border-slate-700/50`}>
          <div className="flex flex-col items-end hidden sm:flex">
            <span className="text-sm font-bold leading-none text-slate-800 dark:text-slate-200">{user?.name}</span>
            <span className="text-[10px] font-semibold mt-1 uppercase tracking-wider text-primary-600 dark:text-primary-400">{user?.role}</span>
          </div>
          <div className="w-10 h-10 rounded-xl bg-gradient-primary flex items-center justify-center text-white text-sm font-black shadow-lg shadow-primary-500/30 overflow-hidden ring-2 ring-primary-500/20">
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
