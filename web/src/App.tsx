import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { useAuthStore } from '@/store/authStore'
import { DashboardLayout } from '@/components/layout/DashboardLayout'
import { FullPageLoader } from '@/components/ui/LoadingSpinner'
import { ROUTES } from '@/constants/config'
import { LanguageProvider, useLanguage } from '@/lib/language'

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
const DoctorPatientRecords = lazy(() => import('@/pages/doctor/DoctorPatientRecords'))
const DoctorReports = lazy(() => import('@/pages/doctor/DoctorReports'))
const DoctorChat = lazy(() => import('@/pages/doctor/DoctorChat'))
const DoctorReviews = lazy(() => import('@/pages/doctor/DoctorReviews'))
const ManageSecretaries = lazy(() => import('@/pages/doctor/ManageSecretaries'))

// Secretary
const SecretaryDashboard = lazy(() => import('@/pages/secretary/SecretaryDashboard'))
const SecretaryDoctorSchedule = lazy(() => import('@/pages/secretary/SecretaryDoctorSchedule'))

// Public pages
const DoctorsList = lazy(() => import('@/pages/doctor/DoctorsList'))
const DoctorDetails = lazy(() => import('@/pages/doctor/DoctorDetails'))

// Doctor - Clinical
const DoctorToday = lazy(() => import('@/pages/doctor/DoctorToday'))
const DoctorWorkspace = lazy(() => import('@/pages/doctor/DoctorWorkspace'))
const DoctorVisitSummary = lazy(() => import('@/pages/doctor/DoctorVisitSummary'))

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
  if (!role || !roles.some(r => r.toLowerCase() === role.toLowerCase())) return <Navigate to={ROUTES.LOGIN} replace />
  return <Outlet />
}

// Public-only Guard (redirect if already logged in)
function PublicGuard() {
  const { isAuthenticated, role } = useAuthStore()
  const isHydrated = useAuthStore.persist.hasHydrated()

  if (!isHydrated) return <FullPageLoader />
  if (isAuthenticated) {
    const lowerRole = role?.toLowerCase()
    if (lowerRole === 'admin') return <Navigate to={ROUTES.ADMIN_DASHBOARD} replace />
    if (lowerRole === 'doctor') return <Navigate to={ROUTES.DOCTOR_DASHBOARD} replace />
    if (lowerRole === 'secretary') return <Navigate to="/secretary/dashboard" replace />
  }
  return <Outlet />
}

export default function App() {
  return (
    <LanguageProvider>
      <BrowserRouter>
        <AppShell />
      </BrowserRouter>
    </LanguageProvider>
  )
}

function AppShell() {
  const { isRTL } = useLanguage()

  return (
    <>
      <Toaster
        position="top-center"
        reverseOrder={false}
        toastOptions={{
          style: { fontFamily: 'Tajawal, sans-serif', fontSize: '14px', direction: isRTL ? 'rtl' : 'ltr' },
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
                <Route path="/doctor/patients/:patientId/records" element={<DoctorPatientRecords />} />
                <Route path="/doctor/reports" element={<DoctorReports />} />
                <Route path="/doctor/reviews" element={<DoctorReviews />} />
                <Route path="/doctor/staff" element={<ManageSecretaries />} />
                <Route path="/doctor/chat" element={<DoctorChat />} />
                <Route path="/doctor/today" element={<DoctorToday />} />
                <Route path="/doctor/workspace/:visitId" element={<DoctorWorkspace />} />
                <Route path="/doctor/visits/:id/summary" element={<DoctorVisitSummary />} />
              </Route>

              {/* Secretary routes */}
              <Route element={<RoleGuard roles={['Secretary']} />}>
                <Route path="/secretary/dashboard" element={<SecretaryDashboard />} />
                <Route path="/secretary/schedule" element={<SecretaryDoctorSchedule />} />
              </Route>

            </Route>
          </Route>

          {/* 404 fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </>
  )
}

function RootRedirect() {
  const { isAuthenticated, role } = useAuthStore()
  const isHydrated = useAuthStore.persist.hasHydrated()

  if (!isHydrated) return <FullPageLoader />
  if (!isAuthenticated) return <Navigate to={ROUTES.LOGIN} replace />
  const lowerRole = role?.toLowerCase()
  if (lowerRole === 'admin') return <Navigate to={ROUTES.ADMIN_DASHBOARD} replace />
  if (lowerRole === 'doctor') return <Navigate to={ROUTES.DOCTOR_DASHBOARD} replace />
  if (lowerRole === 'secretary') return <Navigate to="/secretary/dashboard" replace />
  return <Navigate to={ROUTES.LOGIN} replace />
}
