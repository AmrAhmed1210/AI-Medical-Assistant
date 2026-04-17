import { useEffect, useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { 
  Users, Stethoscope, Activity, TrendingUp, RefreshCw, 
  Calendar, ArrowUpRight, ArrowDownRight,
  Zap, Shield, BarChart3, PieChart as PieChartIcon, Clock, Sparkles
} from 'lucide-react'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Area, AreaChart
} from 'recharts'
import { adminApi } from '@/api/adminApi'
import type { SystemStatsDto } from '@/lib/types'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'

// ── Mock Data for Dev (Fallback) ───────────────────────────────────────
const MOCK_STATS: SystemStatsDto = {
  totalUsers: 1250,
  totalDoctors: 48,
  totalPatients: 1202,
  sessionsToday: 24,
  urgencyDistribution: { LOW: 45, MEDIUM: 30, HIGH: 15, EMERGENCY: 10 },
  sessionsPerDay: [
    { date: '2024-03-01', count: 12 }, { date: '2024-03-02', count: 18 }, { date: '2024-03-03', count: 15 },
    { date: '2024-03-04', count: 25 }, { date: '2024-03-05', count: 20 }, { date: '2024-03-06', count: 30 },
    { date: '2024-03-07', count: 22 },
  ],
  userGrowth: [
    { date: '2024-03-01', count: 1000 }, { date: '2024-03-02', count: 1020 }, { date: '2024-03-03', count: 1050 },
    { date: '2024-03-04', count: 1100 }, { date: '2024-03-05', count: 1150 }, { date: '2024-03-06', count: 1200 },
    { date: '2024-03-07', count: 1250 },
  ]
} as any

const URGENCY_CONFIG = {
  LOW:       { label: 'Low', color: '#10b981', bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-100' },
  MEDIUM:    { label: 'Medium', color: '#f59e0b', bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-100' },
  HIGH:      { label: 'High', color: '#ef4444', bg: 'bg-rose-50', text: 'text-rose-700', border: 'border-rose-100' },
  EMERGENCY: { label: 'Emergency', color: '#7f1d1d', bg: 'bg-red-900/10', text: 'text-red-900', border: 'border-red-200' },
}

const STAT_CONFIG = {
  users:    { gradient: 'from-blue-600 to-indigo-700', icon: Users, shadow: 'shadow-blue-500/20' },
  doctors:  { gradient: 'from-emerald-600 to-teal-700', icon: Stethoscope, shadow: 'shadow-emerald-500/20' },
  patients: { gradient: 'from-orange-600 to-amber-700', icon: Users, shadow: 'shadow-orange-500/20' },
  sessions: { gradient: 'from-rose-600 to-pink-700', icon: Activity, shadow: 'shadow-rose-500/20' },
}

// ── Stat Card Component ───────────────────────────────────────────────
const StatCard = ({ title, value, icon: Icon, gradient, shadow, index, trend }: any) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
    transition={{ delay: index * 0.1 }}
    whileHover={{ y: -5 }} className="group"
  >
    <Card className="relative overflow-hidden border-0 shadow-2xl bg-white/90 backdrop-blur-xl rounded-3xl">
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="space-y-4">
            <p className="text-sm font-bold text-slate-500 uppercase tracking-widest">{title}</p>
            <h3 className="text-4xl font-black text-slate-900">{value.toLocaleString()}</h3>
            {trend && (
              <div className={`flex items-center gap-1.5 text-xs font-bold ${trend > 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                {trend > 0 ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                {Math.abs(trend)}% vs last month
              </div>
            )}
          </div>
          <div className={`p-4 rounded-2xl bg-gradient-to-br ${gradient} text-white shadow-xl ${shadow} group-hover:rotate-12 transition-transform`}>
            <Icon size={24} />
          </div>
        </div>
      </CardContent>
    </Card>
  </motion.div>
)

export default function AdminDashboard() {
  const [stats, setStats] = useState<SystemStatsDto | null>(MOCK_STATS)
  const [loading, setLoading] = useState(true)

  const fetchData = () => {
    setLoading(true)
    adminApi.getStats()
      .then(setStats)
      .catch(() => setStats(MOCK_STATS))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchData() }, [])

  const urgencyData = useMemo(() => {
    if (!stats?.urgencyDistribution) return []
    return Object.entries(stats.urgencyDistribution).map(([key, value]) => ({
      name: URGENCY_CONFIG[key as keyof typeof URGENCY_CONFIG]?.label ?? key,
      value: value as number,
      color: URGENCY_CONFIG[key as keyof typeof URGENCY_CONFIG]?.color ?? '#64748b',
      config: URGENCY_CONFIG[key as keyof typeof URGENCY_CONFIG] ?? URGENCY_CONFIG.MEDIUM,
    })).filter(item => item.value > 0)
  }, [stats])

  return (
    <motion.div 
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      className="min-h-screen bg-[#f8fafc] p-4 md:p-8 space-y-8"
    >
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <h1 className="text-4xl font-black text-slate-900 tracking-tight">Executive Dashboard</h1>
            <Badge className="bg-blue-600/10 text-blue-600 border-0 flex gap-1.5 items-center px-3 py-1 text-xs">
              <Sparkles size={12} fill="currentColor" /> Admin Portal
            </Badge>
          </div>
          <p className="text-slate-500 font-medium flex items-center gap-2">
            <Clock size={16} /> Latest intelligence as of {new Date().toLocaleTimeString()}
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline" onClick={fetchData} className="rounded-2xl gap-2 bg-white/50 backdrop-blur-sm border-slate-200">
             <RefreshCw size={18} className={loading ? 'animate-spin' : ''} /> Refresh Sync
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard title="Overall Users" value={stats?.totalUsers ?? 0} icon={STAT_CONFIG.users.icon} gradient={STAT_CONFIG.users.gradient} shadow={STAT_CONFIG.users.shadow} index={0} trend={12} />
        <StatCard title="Active Doctors" value={stats?.totalDoctors ?? 0} icon={STAT_CONFIG.doctors.icon} gradient={STAT_CONFIG.doctors.gradient} shadow={STAT_CONFIG.doctors.shadow} index={1} trend={5} />
        <StatCard title="Patient Base" value={stats?.totalPatients ?? 0} icon={STAT_CONFIG.patients.icon} gradient={STAT_CONFIG.patients.gradient} shadow={STAT_CONFIG.patients.shadow} index={2} trend={8} />
        <StatCard title="Daily Sessions" value={stats?.sessionsToday ?? 0} icon={STAT_CONFIG.sessions.icon} gradient={STAT_CONFIG.sessions.gradient} shadow={STAT_CONFIG.sessions.shadow} index={3} trend={24} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <Card className="lg:col-span-2 border-0 shadow-2xl rounded-3xl overflow-hidden bg-white">
          <CardHeader className="p-6 border-b border-slate-100 flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-xl font-bold">Activity Velocity</CardTitle>
              <p className="text-sm font-medium text-slate-400">System throughput over last 30 intervals</p>
            </div>
          </CardHeader>
          <CardContent className="p-6">
            <ResponsiveContainer width="100%" height={320}>
              <AreaChart data={stats?.sessionsPerDay || []}>
                <defs>
                  <linearGradient id="colorArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.15}/>
                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12}} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12}} />
                <Tooltip contentStyle={{ borderRadius: 16, border: 'none', boxShadow: '0 10px 30px rgba(0,0,0,0.1)' }} />
                <Area type="monotone" dataKey="count" stroke="#2563eb" strokeWidth={4} fillOpacity={1} fill="url(#colorArea)" />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-2xl rounded-3xl overflow-hidden bg-white">
          <CardHeader className="p-6 border-b border-slate-100">
            <CardTitle className="text-xl font-bold">Severity Balance</CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={urgencyData} cx="50%" cy="50%" innerRadius={70} outerRadius={100} paddingAngle={8} dataKey="value" stroke="white" strokeWidth={5}>
                  {urgencyData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div className="mt-8 space-y-3">
              {urgencyData.map((item) => (
                <div key={item.name} className={`flex items-center justify-between p-3 rounded-2xl border ${item.config.border} ${item.config.bg}`}>
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                    <span className={`text-sm font-bold ${item.config.text}`}>{item.name}</span>
                  </div>
                  <span className={`text-sm font-black ${item.config.text}`}>{item.value}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </motion.div>
  )
}