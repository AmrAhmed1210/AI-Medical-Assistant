import { useEffect, useState } from 'react'
import { Bell, Search, X } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNotificationStore } from '@/store/notificationStore'
import { useAuthStore } from '@/store/authStore'
import { cn } from '@/lib/utils'
import { formatTimeAgo } from '@/lib/utils'

interface TopBarProps {
  title?: string
}

export function TopBar({ title }: TopBarProps) {
  const [showNotif, setShowNotif] = useState(false)
  const { notifications, unreadCount, markAllRead, removeNotification, addNotification } = useNotificationStore()
  const { user, token } = useAuthStore()

  return (
    <header className="fixed top-0 right-0 left-64 h-16 bg-white border-b border-gray-100 flex items-center justify-between px-6 z-20 shadow-sm">
      <div className="flex items-center gap-4">
        {title && <h1 className="text-base font-semibold text-gray-800">{title}</h1>}
      </div>

      <div className="flex items-center gap-6">
        {/* Search */}
        <div className="relative hidden md:block">
          <Search size={16} className="absolute top-1/2 -translate-y-1/2 left-3 text-gray-400" />
          <input
            type="text"
            placeholder="Search..."
            className="pl-9 pr-4 py-1.5 text-sm bg-gray-50 border border-gray-200 rounded-xl w-64 focus:outline-none focus:ring-2 focus:ring-primary-500/20 focus:border-primary-400"
          />
        </div>

        {/* Notification Bell */}
        <div className="relative">
          <button
            onClick={() => { setShowNotif(!showNotif); if (!showNotif) markAllRead() }}
            className="relative p-2 rounded-xl hover:bg-gray-100 transition-colors text-gray-500"
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
                className="absolute right-0 top-full mt-2 w-80 bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden z-50"
              >
                <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
                  <span className="font-semibold text-sm text-gray-800">Notifications</span>
                  <button onClick={() => setShowNotif(false)} className="text-gray-400 hover:text-gray-600">
                    <X size={14} />
                  </button>
                </div>
                <div className="max-h-72 overflow-y-auto divide-y divide-gray-50">
                  {notifications.length === 0 ? (
                    <p className="text-center text-sm text-gray-400 py-8">No notifications</p>
                  ) : notifications.map((n) => (
                    <div
                      key={n.id}
                      onClick={() => removeNotification(n.id)}
                      className={cn(
                        'w-full text-left flex items-start gap-3 p-3 hover:bg-gray-50 cursor-pointer',
                        !n.read && 'bg-blue-50/40'
                      )}
                    >
                      <div className={cn(
                        'w-2 h-2 rounded-full mt-1.5 flex-shrink-0',
                        n.type === 'success' ? 'bg-green-500' :
                        n.type === 'error' ? 'bg-red-500' :
                        n.type === 'warning' ? 'bg-amber-500' : 'bg-blue-500'
                      )} />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-800">{n.title}</p>
                        {n.message && <p className="text-xs text-gray-500 mt-0.5">{n.message}</p>}
                        <p className="text-xs text-gray-400 mt-1">{formatTimeAgo(n.createdAt)}</p>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          removeNotification(n.id)
                        }}
                        className="text-gray-300 hover:text-gray-500 flex-shrink-0"
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
        <div className="flex items-center gap-3 border-l border-gray-100 pl-6 ml-2">
          <div className="flex flex-col items-end hidden sm:flex">
            <span className="text-xs font-bold text-gray-900 leading-none">{user?.name}</span>
            <span className="text-[10px] text-gray-400 font-medium mt-1 uppercase tracking-wider">{user?.role}</span>
          </div>
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-primary-500 to-primary-600 flex items-center justify-center text-white text-xs font-black shadow-lg shadow-primary-500/20 overflow-hidden ring-2 ring-white">
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
