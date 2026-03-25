import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Users, Stethoscope, Activity, Calendar, AlertTriangle } from 'lucide-react'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import { adminApi } from '@/api/adminApi'
import type { SystemStatsDto } from '@/lib/types'
import { StatCard } from '@/components/admin/StatCard'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { PageLoader } from '@/components/ui/LoadingSpinner'

const URGENCY_COLORS = { LOW: '#22c55e', MEDIUM: '#f59e0b', HIGH: '#ef4444', EMERGENCY: '#7f1d1d' }
const URGENCY_LABELS = { LOW: 'منخفض', MEDIUM: 'متوسط', HIGH: 'مرتفع', EMERGENCY: 'طوارئ' }

export default function AdminDashboard() {
  const [stats, setStats] = useState<SystemStatsDto | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    adminApi.getStats()
      .then(setStats)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <PageLoader />

  const urgencyData = stats ? Object.entries(stats.urgencyDistribution).map(([key, value]) => ({
    name: URGENCY_LABELS[key as keyof typeof URGENCY_LABELS],
    value,
    color: URGENCY_COLORS[key as keyof typeof URGENCY_COLORS],
  })) : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-gray-800">لوحة تحكم المدير</h1>
        <p className="text-sm text-gray-500 mt-0.5">نظرة عامة على النظام</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="إجمالي المستخدمين" value={stats?.totalUsers ?? 0} icon={<Users size={20} />} color="blue" index={0} />
        <StatCard title="الأطباء" value={stats?.totalDoctors ?? 0} icon={<Stethoscope size={20} />} color="green" index={1} />
        <StatCard title="المرضى" value={stats?.totalPatients ?? 0} icon={<Users size={20} />} color="purple" index={2} />
        <StatCard title="جلسات اليوم" value={stats?.sessionsToday ?? 0} icon={<Activity size={20} />} color="amber" index={3} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sessions per day bar chart */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>الجلسات اليومية</CardTitle>
              <span className="text-xs text-gray-400">آخر 30 يوم</span>
            </CardHeader>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={stats?.sessionsPerDay ?? []} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#9ca3af' }} tickFormatter={(v) => v.slice(5)} />
                <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }} />
                <Bar dataKey="count" fill="#2563eb" radius={[4, 4, 0, 0]} name="جلسات" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </motion.div>

        {/* Urgency pie chart */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
          <Card className="h-full">
            <CardHeader>
              <CardTitle>توزيع مستوى الطوارئ</CardTitle>
            </CardHeader>
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie data={urgencyData} cx="50%" cy="50%" innerRadius={50} outerRadius={75} paddingAngle={3} dataKey="value">
                  {urgencyData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
              </PieChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-2 gap-1 mt-2">
              {urgencyData.map((item) => (
                <div key={item.name} className="flex items-center gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: item.color }} />
                  <span className="text-xs text-gray-600">{item.name}: {item.value}</span>
                </div>
              ))}
            </div>
          </Card>
        </motion.div>
      </div>

      {/* User growth line chart */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
        <Card>
          <CardHeader>
            <CardTitle>نمو المستخدمين</CardTitle>
            <span className="text-xs text-gray-400">آخر 30 يوم</span>
          </CardHeader>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={stats?.userGrowth ?? []} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#9ca3af' }} tickFormatter={(v) => v.slice(5)} />
              <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }} />
              <Line type="monotone" dataKey="count" stroke="#2563eb" strokeWidth={2} dot={false} name="مستخدمون جدد" />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </motion.div>
    </div>
  )
}
