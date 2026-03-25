import { useEffect, useState } from 'react'
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { adminApi } from '@/api/adminApi'
import type { SystemStatsDto } from '@/lib/types'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { PageLoader } from '@/components/ui/LoadingSpinner'

const URGENCY_COLORS = { LOW: '#22c55e', MEDIUM: '#f59e0b', HIGH: '#ef4444', EMERGENCY: '#7f1d1d' }
const URGENCY_LABELS = { LOW: 'منخفض', MEDIUM: 'متوسط', HIGH: 'مرتفع', EMERGENCY: 'طوارئ' }

export default function AdminStatistics() {
  const [stats, setStats] = useState<SystemStatsDto | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    adminApi.getStats().then(setStats).finally(() => setLoading(false))
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
        <h1 className="text-xl font-bold text-gray-800">الإحصائيات التفصيلية</h1>
        <p className="text-sm text-gray-500 mt-0.5">تحليل شامل لأداء المنصة</p>
      </div>

      {/* Summary boxes */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'جلسات هذا الأسبوع', value: stats?.sessionsThisWeek ?? 0 },
          { label: 'جلسات اليوم', value: stats?.sessionsToday ?? 0 },
          { label: 'إجمالي الأطباء', value: stats?.totalDoctors ?? 0 },
          { label: 'إجمالي المرضى', value: stats?.totalPatients ?? 0 },
        ].map((s) => (
          <Card key={s.label}>
            <p className="text-2xl font-bold text-primary-600">{s.value}</p>
            <p className="text-sm text-gray-500 mt-1">{s.label}</p>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader><CardTitle>الجلسات اليومية</CardTitle></CardHeader>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={stats?.sessionsPerDay ?? []} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#9ca3af' }} tickFormatter={(v) => v.slice(5)} />
              <YAxis tick={{ fontSize: 10, fill: '#9ca3af' }} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
              <Bar dataKey="count" fill="#2563eb" radius={[4, 4, 0, 0]} name="جلسات" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <CardHeader><CardTitle>نمو المستخدمين</CardTitle></CardHeader>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={stats?.userGrowth ?? []} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#9ca3af' }} tickFormatter={(v) => v.slice(5)} />
              <YAxis tick={{ fontSize: 10, fill: '#9ca3af' }} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
              <Line type="monotone" dataKey="count" stroke="#22c55e" strokeWidth={2} dot={false} name="مستخدمون جدد" />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <CardHeader><CardTitle>توزيع مستويات الطوارئ</CardTitle></CardHeader>
          <div className="flex items-center gap-6">
            <ResponsiveContainer width="60%" height={200}>
              <PieChart>
                <Pie data={urgencyData} cx="50%" cy="50%" innerRadius={55} outerRadius={80} paddingAngle={3} dataKey="value">
                  {urgencyData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                </Pie>
                <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
              </PieChart>
            </ResponsiveContainer>
            <div className="space-y-3">
              {urgencyData.map((item) => (
                <div key={item.name} className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                  <div>
                    <p className="text-xs font-medium text-gray-700">{item.name}</p>
                    <p className="text-xs text-gray-400">{item.value} جلسة</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}
