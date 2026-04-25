import { Outlet } from 'react-router-dom'
import { useEffect } from 'react'
import toast from 'react-hot-toast'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'
import { startConnection } from '@/lib/signalr'
import { useAuthStore } from '@/store/authStore'
import { useNotificationStore } from '@/store/notificationStore'
import { useAppointmentStore } from '@/store/appointmentStore'

export function DashboardLayout() {
  const { token, role } = useAuthStore()
  const { incrementSessionMessage, incrementAppointments, incrementDoctorApplications, setLatestMessagePayload, addNotification } = useNotificationStore()
  const { handleNewBooking, handleAppointmentUpdated } = useAppointmentStore()

  useEffect(() => {
    if (!token || (role !== 'Doctor' && role !== 'Admin')) return
    let disposed = false
    let cleanup: (() => void) | undefined

    startConnection(token)
      .then((conn) => {
        if (disposed) return

        const onNewMessage = (payload: any) => {
          setLatestMessagePayload(payload);
          const sessionId = payload?.sessionId ?? payload?.SessionId;
          const isChatActive = window.location.pathname.includes('/doctor/chat');
          if (!isChatActive && sessionId) {
            incrementSessionMessage(sessionId);
          }
        }
        
        const onNewBooking = (data: any) => {
          incrementAppointments()
          handleNewBooking(data)
          toast.success(`📅 New booking: ${data?.patientName ?? 'Patient'}`, { duration: 5000 })
        }
        
        const onBookingCancelled = (data: any) => {
          handleAppointmentUpdated({ appointmentId: data?.appointmentId ?? data?.id, status: 'Cancelled' })
          toast.error(`❌ Cancelled: ${data?.patientName ?? 'Patient'}`, { duration: 5000 })
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
          addNotification(type, 'Appointment Update', String(payload?.message ?? 'Appointment updated'))
          const message = String(payload?.message ?? 'Appointment updated')
          toast.success(message)
        }

        conn.on('NewMessage', onNewMessage)
        conn.on('NewBooking', onNewBooking)
        conn.on('BookingCancelled', onBookingCancelled)
        conn.on('NotificationReceived', onNotificationReceived)
        conn.on('AppointmentUpdated', onAppointmentUpdated)

        conn.on('NewDoctorApplication', onNewDoctorApplication)

        cleanup = () => {
          conn.off('NewMessage', onNewMessage)
          conn.off('NewBooking', onNewBooking)
          conn.off('BookingCancelled', onBookingCancelled)
          conn.off('NotificationReceived', onNotificationReceived)
          conn.off('AppointmentUpdated', onAppointmentUpdated)
          conn.off('NewDoctorApplication', onNewDoctorApplication)
        }
      })
      .catch(() => undefined)

    return () => {
      disposed = true
      cleanup?.()
    }
  }, [token, role, incrementSessionMessage, incrementAppointments, incrementDoctorApplications, setLatestMessagePayload, addNotification])

  return (
    <div className="min-h-screen bg-gray-50 font-outfit" dir="ltr">
      <Sidebar />
      <TopBar />
      <main className="ml-64 mt-16 p-6 min-h-[calc(100vh-4rem)]">
        <Outlet />
      </main>
    </div>
  )
}
