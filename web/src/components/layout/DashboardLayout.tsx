import { Outlet, useNavigate } from 'react-router-dom'
import { useEffect } from 'react'
import { motion } from 'framer-motion'
import toast from 'react-hot-toast'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'
import { startConnection, stopConnection } from '@/lib/signalr'
import { useAuthStore } from '@/store/authStore'
import { useNotificationStore } from '@/store/notificationStore'
import { useAppointmentStore } from '@/store/appointmentStore'
import { useThemeStore } from '@/store/themeStore'
import { useLanguage } from '@/lib/language'

export function DashboardLayout() {
  const { token, role } = useAuthStore()
  const navigate = useNavigate()
  const { incrementSessionMessage, incrementAppointments, incrementDoctorApplications, setLatestMessagePayload, addNotification } = useNotificationStore()
  const { handleNewBooking, handleAppointmentUpdated } = useAppointmentStore()
  const { isRTL, t } = useLanguage()

  useEffect(() => {
    if (!token || !role) return
    let disposed = false
    let cleanup: (() => void) | undefined
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    const setupConnection = async () => {
      try {
        const conn = await startConnection(token)
        if (disposed) return

        const onNewMessage = (payload: any) => {
          // Deduplicate: check if message already exists in current session
          const sessionId = payload?.sessionId ?? payload?.SessionId
          const messageId = payload?.messageId ?? payload?.id ?? payload?.Id

          if (sessionId && messageId) {
            const dedupKey = `msg_${sessionId}_${messageId}`
            if (sessionStorage.getItem(dedupKey)) return // Already processed
            sessionStorage.setItem(dedupKey, Date.now().toString())
            // Cleanup old dedup keys after 5 minutes
            setTimeout(() => sessionStorage.removeItem(dedupKey), 300000)
          }

          setLatestMessagePayload(payload)
          const isChatActive = window.location.pathname.includes('/doctor/chat') || window.location.pathname.includes('/messages')
          if (!isChatActive && sessionId) {
            incrementSessionMessage(sessionId)
          }
        }

        const onNewBooking = (data: any) => {
          incrementAppointments()
          handleNewBooking(data)
          toast.success(`📅 ${t('newBooking')}: ${data?.patientName ?? 'Patient'}`, { duration: 5000 })
        }

        const onBookingCancelled = (data: any) => {
          handleAppointmentUpdated({ appointmentId: data?.appointmentId ?? data?.id, status: 'Cancelled' })
          toast.error(`❌ ${t('cancelled')}: ${data?.patientName ?? 'Patient'}`, { duration: 5000 })
        }

        const onNotificationReceived = (payload: any) => {
          const category = String(payload?.category ?? payload?.Category ?? '').toLowerCase()
          const title = String(payload?.title ?? payload?.Title ?? 'Notification')
          const message = String(payload?.message ?? payload?.Message ?? '')
          const level =
            category.includes('cancel') ? 'warning' :
              category.includes('confirm') || category.includes('booking') ? 'success' :
                category.includes('error') ? 'error' : 'info'

          addNotification(level, title, message)

          if (category === 'new_doctor_application') {
            incrementDoctorApplications()
          }

          // Don't toast duplicates if we already show NewBooking or BookingCancelled
          if (!category.includes('new_booking') && !category.includes('cancel')) {
            toast(message)
          }
        }

        const onNewDoctorApplication = (payload: any) => {
          incrementDoctorApplications()
          toast.success(`New application from ${payload.applicantName || 'a doctor'}`)
        }

        const onAppointmentUpdated = (payload: any) => {
          handleAppointmentUpdated(payload)
          const status = String(payload?.status ?? '').toLowerCase()
          const type =
            status === 'confirmed' ? 'success' :
              status === 'cancelled' ? 'warning' : 'info'
          addNotification(type, t('appointmentUpdate'), String(payload?.message ?? 'Appointment updated'))
          const message = String(payload?.message ?? 'Appointment updated')
          toast.success(message)
        }

        // New Consultation handler for Patients
        const onNewConsultation = (payload: any) => {
          const doctorName = String(payload?.doctorName ?? payload?.DoctorName ?? 'Doctor')
          const title = String(payload?.title ?? payload?.Title ?? 'Consultation')

          addNotification('info', t('newConsultation'), `Dr. ${doctorName} scheduled: ${title}`)
          toast.success(`🩺 Dr. ${doctorName} scheduled a consultation: ${title}`, { duration: 6000 })

          // If patient is on appointments page, refresh appointments
          if (window.location.pathname.includes('/appointments') || window.location.pathname.includes('/profile')) {
            // The page will auto-refresh via its own useEffect
          }
        }

        conn.on('NewMessage', onNewMessage)
        conn.on('NewBooking', onNewBooking)
        conn.on('BookingCancelled', onBookingCancelled)
        conn.on('NotificationReceived', onNotificationReceived)
        conn.on('AppointmentUpdated', onAppointmentUpdated)
        conn.on('NewDoctorApplication', onNewDoctorApplication)

        // Patient-specific events
        if (role === 'Patient') {
          conn.on('NewConsultation', onNewConsultation)
        }

        // Handle reconnection
        conn.onreconnecting(() => {
          console.log('[SignalR] Reconnecting...')
        })

        conn.onreconnected(() => {
          console.log('[SignalR] Reconnected')
          toast.success(t('connectionRestored'))
        })

        conn.onclose(() => {
          console.log('[SignalR] Connection closed')
          if (!disposed) {
            // Try to reconnect after 5 seconds
            reconnectTimer = setTimeout(() => {
              if (!disposed) setupConnection()
            }, 5000)
          }
        })

        cleanup = () => {
          conn.off('NewMessage', onNewMessage)
          conn.off('NewBooking', onNewBooking)
          conn.off('BookingCancelled', onBookingCancelled)
          conn.off('NotificationReceived', onNotificationReceived)
          conn.off('AppointmentUpdated', onAppointmentUpdated)
          conn.off('NewDoctorApplication', onNewDoctorApplication)
          conn.off('NewConsultation', onNewConsultation)
        }
      } catch (err) {
        console.error('[SignalR] Setup failed:', err)
        // Retry connection after 10 seconds
        if (!disposed) {
          reconnectTimer = setTimeout(() => {
            if (!disposed) setupConnection()
          }, 10000)
        }
      }
    }

    setupConnection()

    return () => {
      disposed = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      cleanup?.()
      stopConnection().catch(() => undefined)
    }
  }, [token, role, incrementSessionMessage, incrementAppointments, incrementDoctorApplications, setLatestMessagePayload, addNotification, navigate, t])

  const { isDark } = useThemeStore()

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [isDark])

  return (
    <div
      className="min-h-screen font-outfit relative overflow-hidden transition-colors duration-500"
      dir={isRTL ? 'rtl' : 'ltr'}
      style={{
        background: isDark
          ? 'linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e1b4b 100%)'
          : 'linear-gradient(135deg, #f0f4ff 0%, #ffffff 50%, #f5f3ff 100%)',
      }}
    >
      {/* Animated Aesthetic Bubbles */}
      <motion.div
        className="fixed pointer-events-none"
        animate={{ y: [0, 15, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
        style={{
          width: 350, height: 350, top: -100, left: -100, borderRadius: '50%',
          background: isDark ? 'rgba(16, 185, 129, 0.16)' : 'rgba(16, 185, 129, 0.09)',
          filter: 'blur(80px)', zIndex: 0
        }}
      />
      <motion.div
        className="fixed pointer-events-none"
        animate={{ y: [0, -20, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
        style={{
          width: 400, height: 400, bottom: -150, right: -150, borderRadius: '50%',
          background: isDark ? 'rgba(14, 165, 233, 0.15)' : 'rgba(14, 165, 233, 0.08)',
          filter: 'blur(80px)', zIndex: 0
        }}
      />
      <motion.div
        className="fixed pointer-events-none"
        animate={{ y: [0, -10, 0] }}
        transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut', delay: 2 }}
        style={{
          width: 250, height: 250, top: '40%', left: '20%', borderRadius: '50%',
          background: isDark ? 'rgba(20, 184, 166, 0.10)' : 'rgba(20, 184, 166, 0.06)',
          filter: 'blur(100px)', zIndex: 0
        }}
      />

      <div className="relative z-10 flex">
        <Sidebar />
        <div className={`flex-1 flex flex-col min-w-0 ${isRTL ? 'mr-[280px]' : 'ml-[280px]'}`}>
          <TopBar />
          <main className="p-6 min-h-[calc(100vh-4rem)]">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  )
}
