import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { useAuthStore } from '@/store/authStore'
import { DashboardLayout } from '@/components/layout/DashboardLayout'
import { FullPageLoader } from '@/components/ui/LoadingSpinner'
import { ROUTES } from '@/constants/config'

// Lazy-loaded pages
const LoginPage = lazy(() => import('@/pages/auth/LoginPage'))
const ApplyPage = lazy(() => import('@/pages/auth/ApplyPage'))

// Admin
const AdminDashboard = lazy(() => import('@/pages/admin/AdminDashboard'))
const AdminUsers = lazy(() => import('@/pages/admin/UsersPage'))
const AdminStatistics = lazy(() => import('@/pages/admin/StatisticsPage'))
const AdminModels = lazy(() => import('@/pages/admin/ModelManagementPage'))
const AdminApplications = lazy(() => import('@/pages/admin/ApplicationsPage'))
const AdminSupport = lazy(() => import('@/pages/admin/SupportPage'))

// Doctor
const DoctorDashboard = lazy(() => import('@/pages/doctor/DoctorDashboard'))
const DoctorProfile = lazy(() => import('@/pages/doctor/DoctorProfile'))
const DoctorSchedule = lazy(() => import('@/pages/doctor/DoctorSchedule'))
const DoctorAppointments = lazy(() => import('@/pages/doctor/DoctorAppointments'))
const DoctorPatients = lazy(() => import('@/pages/doctor/DoctorPatients'))
const DoctorReports = lazy(() => import('@/pages/doctor/DoctorReports'))
const DoctorChat = lazy(() => import('@/pages/doctor/DoctorChat'))
const DoctorReviews = lazy(() => import('@/pages/doctor/DoctorReviews'))

// Public pages
const DoctorsList = lazy(() => import('@/pages/doctor/DoctorsList'))
const DoctorDetails = lazy(() => import('@/pages/doctor/DoctorDetails'))

// Auth Guard
function AuthGuard() {
  const { isAuthenticated } = useAuthStore()
  const isHydrated = useAuthStore.persist.hasHydrated()

  if (!isHydrated) return <FullPageLoader />
  if (!isAuthenticated) return <Navigate to={ROUTES.LOGIN} replace />
  return <Outlet />
}

// Role Guard
function RoleGuard({ roles }: { roles: string[] }) {
  const { role } = useAuthStore()
  const isHydrated = useAuthStore.persist.hasHydrated()

  if (!isHydrated) return <FullPageLoader />
  if (!role || !roles.includes(role)) return <Navigate to={ROUTES.LOGIN} replace />
  return <Outlet />
}

// Public-only Guard (redirect if already logged in)
function PublicGuard() {
  const { isAuthenticated, role } = useAuthStore()
  const isHydrated = useAuthStore.persist.hasHydrated()

  if (!isHydrated) return <FullPageLoader />
  if (isAuthenticated) {
    if (role === 'Admin') return <Navigate to={ROUTES.ADMIN_DASHBOARD} replace />
    if (role === 'Doctor') return <Navigate to={ROUTES.DOCTOR_DASHBOARD} replace />
  }
  return <Outlet />
}

export default function App() {
  return (
    <BrowserRouter>
      <Toaster
        position="top-center"
        reverseOrder={false}
        toastOptions={{
          style: { fontFamily: 'Tajawal, sans-serif', fontSize: '14px', direction: 'rtl' },
          success: { style: { background: '#f0fdf4', color: '#15803d', border: '1px solid #bbf7d0' } },
          error: { style: { background: '#fef2f2', color: '#dc2626', border: '1px solid #fecaca' } },
        }}
      />
      <Suspense fallback={<FullPageLoader />}>
        <Routes>
          {/* Root redirect */}
          <Route path="/" element={<RootRedirect />} />

          {/* Public routes */}
          <Route element={<PublicGuard />}>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/apply" element={<ApplyPage />} />
            <Route path="/doctors" element={<DoctorsList />} />
            <Route path="/doctor/:id" element={<DoctorDetails />} />
          </Route>

          {/* Protected routes */}
          <Route element={<AuthGuard />}>
            <Route element={<DashboardLayout />}>

              {/* Admin routes */}
              <Route element={<RoleGuard roles={['Admin']} />}>
                <Route path="/admin/dashboard" element={<AdminDashboard />} />
                <Route path="/admin/users" element={<AdminUsers />} />
                <Route path="/admin/statistics" element={<AdminStatistics />} />
                <Route path="/admin/models" element={<AdminModels />} />
                <Route path="/admin/applications" element={<AdminApplications />} />
                <Route path="/admin/support" element={<AdminSupport />} />
              </Route>

              {/* Doctor routes */}
              <Route element={<RoleGuard roles={['Doctor']} />}>
                <Route path="/doctor/dashboard" element={<DoctorDashboard />} />
                <Route path="/doctor/profile" element={<DoctorProfile />} />
                <Route path="/doctor/schedule" element={<DoctorSchedule />} />
                <Route path="/doctor/appointments" element={<DoctorAppointments />} />
                <Route path="/doctor/patients" element={<DoctorPatients />} />
                <Route path="/doctor/reports" element={<DoctorReports />} />
                <Route path="/doctor/reviews" element={<DoctorReviews />} />
                <Route path="/doctor/chat" element={<DoctorChat />} />
              </Route>

            </Route>
          </Route>

          {/* 404 fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  )
}

function RootRedirect() {
  const { isAuthenticated, role } = useAuthStore()
  const isHydrated = useAuthStore.persist.hasHydrated()

  if (!isHydrated) return <FullPageLoader />
  if (!isAuthenticated) return <Navigate to={ROUTES.LOGIN} replace />
  if (role === 'Admin') return <Navigate to={ROUTES.ADMIN_DASHBOARD} replace />
  if (role === 'Doctor') return <Navigate to={ROUTES.DOCTOR_DASHBOARD} replace />
  return <Navigate to={ROUTES.LOGIN} replace />
}
