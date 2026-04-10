import { motion } from 'framer-motion'
import { Calendar, Clock, Users, BarChart2, Activity, TrendingUp, Eye, FileText } from 'lucide-react'
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
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-blue-50/30 p-6">
      {/* Background decorative elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 right-10 w-72 h-72 bg-gradient-to-br from-primary-100/20 to-transparent rounded-full blur-3xl" />
        <div className="absolute bottom-20 left-10 w-96 h-96 bg-gradient-to-tr from-blue-100/15 to-transparent rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-r from-purple-100/10 via-transparent to-emerald-100/10 rounded-full blur-3xl" />
      </div>

      <div className="relative space-y-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-4 pb-6 border-b border-gray-100"
        >
          <div className="p-3 bg-gradient-to-br from-primary-500 to-primary-600 rounded-2xl shadow-lg shadow-primary-500/25">
            <Activity size={24} className="text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Doctor Dashboard</h1>
            <p className="text-gray-600 mt-1">Welcome back! Here's your practice overview</p>
          </div>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-2 lg:grid-cols-4 gap-6"
        >
          <StatCard title="Today's Appointments" value={dashboard?.todayAppointments ?? 0} icon={<Calendar size={20} />} color="blue" index={0} />
          <StatCard title="Pending" value={dashboard?.pendingAppointments ?? 0} icon={<Clock size={20} />} color="amber" index={1} />
          <StatCard title="Total Patients" value={dashboard?.totalPatients ?? 0} icon={<Users size={20} />} color="green" index={2} />
          <StatCard title="Weekly Appointments" value={dashboard?.weekAppointments ?? 0} icon={<BarChart2 size={20} />} color="purple" index={3} />
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Today Appointments */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card className="border-0 shadow-xl hover:shadow-2xl transition-all duration-300 bg-gradient-to-br from-white via-gray-50/30 to-white">
              <CardHeader className="pb-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl shadow-lg shadow-blue-500/25">
                      <Calendar size={20} className="text-white" />
                    </div>
                    <CardTitle className="text-xl font-bold text-gray-900">Today's Appointments</CardTitle>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => navigate(ROUTES.DOCTOR_APPOINTMENTS)}
                    className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-primary-600 hover:text-primary-700 hover:bg-primary-50 rounded-xl transition-all duration-200"
                  >
                    <Eye size={14} />
                    View All
                  </motion.button>
                </div>
              </CardHeader>
              <div className="px-6 pb-6">
                <div className="space-y-4">
                  {dashboard?.todayAppointmentsList?.length === 0 ? (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="flex flex-col items-center justify-center py-12 text-center"
                    >
                      <div className="p-4 bg-gradient-to-br from-gray-100 to-gray-50 rounded-2xl mb-4">
                        <Calendar size={32} className="text-gray-400" />
                      </div>
                      <p className="text-gray-500 font-medium">No appointments today</p>
                      <p className="text-sm text-gray-400 mt-1">Enjoy your day off!</p>
                    </motion.div>
                  ) : dashboard?.todayAppointmentsList?.slice(0, 5).map((appt, idx) => (
                    <motion.div
                      key={appt.id}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: idx * 0.1 }}
                      whileHover={{ scale: 1.02, x: 5 }}
                      className="flex items-center gap-4 p-4 bg-gradient-to-r from-white to-gray-50/50 rounded-2xl border border-gray-100 hover:border-primary-200 hover:shadow-lg transition-all duration-200 cursor-pointer"
                      onClick={() => navigate(`${ROUTES.DOCTOR_APPOINTMENTS}/${appt.id}`)}
                    >
                      <div className="relative">
                        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center flex-shrink-0 shadow-lg shadow-primary-500/25">
                          <span className="text-white text-sm font-bold">{appt.patientName.charAt(0)}</span>
                        </div>
                        <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white shadow-sm"></div>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-gray-900 truncate">{appt.patientName}</p>
                        <div className="flex items-center gap-2 mt-1">
                          <Clock size={12} className="text-gray-400" />
                          <p className="text-xs text-gray-500">{formatDateTime(appt.scheduledAt)}</p>
                        </div>
                      </div>
                      <div className="flex-shrink-0">
                        <StatusBadge status={appt.status} />
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </Card>
          </motion.div>

          {/* Weekly chart */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card className="border-0 shadow-xl hover:shadow-2xl transition-all duration-300 bg-gradient-to-br from-white via-gray-50/30 to-white">
              <CardHeader className="pb-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl shadow-lg shadow-purple-500/25">
                    <TrendingUp size={20} className="text-white" />
                  </div>
                  <CardTitle className="text-xl font-bold text-gray-900">Weekly Sessions</CardTitle>
                </div>
              </CardHeader>
              <div className="px-6 pb-6">
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={dashboard?.weeklySessionsChart ?? []} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
                    <defs>
                      <linearGradient id="weeklyGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.8} />
                        <stop offset="100%" stopColor="#a855f7" stopOpacity={0.6} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                    <XAxis
                      dataKey="day"
                      tick={{ fontSize: 12, fill: '#64748b', fontWeight: 500 }}
                      axisLine={false}
                      tickLine={false}
                      dy={10}
                    />
                    <YAxis
                      tick={{ fontSize: 12, fill: '#64748b', fontWeight: 500 }}
                      axisLine={false}
                      tickLine={false}
                      dx={-10}
                    />
                    <Tooltip
                      cursor={{ fill: 'rgba(139, 92, 246, 0.1)', radius: 8 }}
                      contentStyle={{
                        fontSize: 13,
                        borderRadius: 12,
                        border: '1px solid #e2e8f0',
                        background: 'white',
                        boxShadow: '0 10px 40px rgba(0,0,0,0.15)',
                        padding: '12px 16px',
                        fontWeight: 600
                      }}
                      labelStyle={{ fontWeight: 700, color: '#1e293b', marginBottom: '8px' }}
                    />
                    <Bar
                      dataKey="count"
                      fill="url(#weeklyGradient)"
                      radius={[8, 8, 0, 0]}
                      name="Sessions"
                      animationDuration={1200}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </motion.div>
        </div>

        {/* Recent AI Reports */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-emerald-500 to-emerald-600 rounded-xl shadow-lg shadow-emerald-500/25">
                <FileText size={20} className="text-white" />
              </div>
              <h2 className="text-xl font-bold text-gray-900">Recent AI Reports</h2>
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate(ROUTES.DOCTOR_REPORTS)}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-primary-600 hover:text-primary-700 hover:bg-primary-50 rounded-xl transition-all duration-200"
            >
              <Eye size={14} />
              View All
            </motion.button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {dashboard?.recentReports?.slice(0, 3).map((report, i) => (
              <AIReportCard
                key={report.reportId}
                report={report}
                index={i}
                onClick={() => navigate(`${ROUTES.DOCTOR_REPORTS}/${report.reportId}`)}
              />
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )
}
