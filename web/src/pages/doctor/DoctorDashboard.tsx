import { motion } from 'framer-motion'
import { Calendar, Clock, Users, BarChart2 } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useDoctorDashboard } from '@/hooks/useDoctor'
import { StatCard } from '@/components/admin/StatCard'
import { AIReportCard } from '@/components/doctor/AIReportCard'
import { StatusBadge } from '@/components/ui/Badge'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { formatDateTime } from '@/lib/utils'
import { useNavigate } from 'react-router-dom'
import { ROUTES } from '@/constants/config'

export default function DoctorDashboard() {
  const { dashboard, isLoading } = useDoctorDashboard()
  const navigate = useNavigate()

  if (isLoading) return <PageLoader />

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-gray-800">لوحة التحكم</h1>
        <p className="text-sm text-gray-500 mt-0.5">مرحباً، هذا ملخص يومك</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="مواعيد اليوم" value={dashboard?.todayAppointments ?? 0} icon={<Calendar size={20} />} color="blue" index={0} />
        <StatCard title="قيد الانتظار" value={dashboard?.pendingAppointments ?? 0} icon={<Clock size={20} />} color="amber" index={1} />
        <StatCard title="إجمالي المرضى" value={dashboard?.totalPatients ?? 0} icon={<Users size={20} />} color="green" index={2} />
        <StatCard title="مواعيد الأسبوع" value={dashboard?.weekAppointments ?? 0} icon={<BarChart2 size={20} />} color="purple" index={3} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Today Appointments */}
        <Card>
          <CardHeader>
            <CardTitle>مواعيد اليوم</CardTitle>
            <button onClick={() => navigate(ROUTES.DOCTOR_APPOINTMENTS)} className="text-xs text-primary-600 hover:underline">
              عرض الكل
            </button>
          </CardHeader>
          <div className="space-y-3">
            {dashboard?.todayAppointmentsList?.length === 0 ? (
              <p className="text-sm text-gray-400 text-center py-8">لا توجد مواعيد اليوم</p>
            ) : dashboard?.todayAppointmentsList?.slice(0, 5).map((appt) => (
              <motion.div
                key={appt.appointmentId}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-center gap-3 p-3 bg-gray-50 rounded-xl"
              >
                <div className="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center flex-shrink-0">
                  <span className="text-primary-700 text-xs font-semibold">{appt.patientName.charAt(0)}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-800 truncate">{appt.patientName}</p>
                  <p className="text-xs text-gray-400">{formatDateTime(appt.scheduledAt)}</p>
                </div>
                <StatusBadge status={appt.status} />
              </motion.div>
            ))}
          </div>
        </Card>

        {/* Weekly chart */}
        <Card>
          <CardHeader>
            <CardTitle>الجلسات الأسبوعية</CardTitle>
          </CardHeader>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={dashboard?.weeklySessionsChart ?? []} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis dataKey="day" tick={{ fontSize: 11, fill: '#9ca3af' }} />
              <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }} />
              <Bar dataKey="count" fill="#2563eb" radius={[4, 4, 0, 0]} name="جلسات" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Recent AI Reports */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-gray-800">آخر تقارير AI</h2>
          <button onClick={() => navigate(ROUTES.DOCTOR_REPORTS)} className="text-xs text-primary-600 hover:underline">
            عرض الكل
          </button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {dashboard?.recentReports?.slice(0, 3).map((report, i) => (
            <AIReportCard
              key={report.reportId}
              report={report}
              index={i}
              onClick={() => navigate(`${ROUTES.DOCTOR_REPORTS}/${report.reportId}`)}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
