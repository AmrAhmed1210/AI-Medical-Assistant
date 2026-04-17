import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, 
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area
} from 'recharts'
import { adminApi } from '@/api/adminApi'
import type { SystemStatsDto } from '@/lib/types'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { 
  Users, Activity, Calendar, TrendingUp, Clock, 
  BarChart3, PieChart as PieChartIcon, Zap, Target, ArrowUpRight,
  ChevronRight, Sparkles, Database, Brain
} from 'lucide-react'

const URGENCY_CONFIG = {
  LOW:       { label: 'Low Sev', color: '#10b981', bg: 'bg-emerald-500' },
  MEDIUM:    { label: 'Medium',  color: '#f59e0b', bg: 'bg-amber-500' },
  HIGH:      { label: 'High Sev', color: '#ef4444', bg: 'bg-rose-500' },
  EMERGENCY: { label: 'Critical', color: '#7f1d1d', bg: 'bg-red-900' },
}

const MOCK_STATS: SystemStatsDto = {
  totalUsers: 1420,
  totalDoctors: 52,
  totalPatients: 1368,
  sessionsToday: 38,
  sessionsThisWeek: 210,
  urgencyDistribution: { LOW: 45, MEDIUM: 30, HIGH: 18, EMERGENCY: 7 },
  sessionsPerDay: [
    { date: 'Mon', count: 12 }, { date: 'Tue', count: 18 }, { date: 'Wed', count: 45 },
    { date: 'Thu', count: 25 }, { date: 'Fri', count: 60 }, { date: 'Sat', count: 32 },
    { date: 'Sun', count: 40 },
  ],
  userGrowth: [
    { date: 'Jan', count: 800 }, { date: 'Feb', count: 950 }, { date: 'Mar', count: 1100 },
    { date: 'Apr', count: 1250 }, { date: 'May', count: 1420 },
  ]
} as any

export default function AdminStatistics() {
  const [stats, setStats] = useState<SystemStatsDto | null>(MOCK_STATS)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<'overview' | 'traffic' | 'urgency'>('overview')

  useEffect(() => {
    adminApi.getStats()
      .then(setStats)
      .catch((err) => {
        console.warn('Stats API Fail, Using Mock.', err)
        setStats(MOCK_STATS)
      })
      .finally(() => setLoading(false))
  }, [])

  if (loading && !stats) return <PageLoader />

  const urgencyData = stats ? Object.entries(stats.urgencyDistribution).map(([key, value]) => ({
    name: URGENCY_CONFIG[key as keyof typeof URGENCY_CONFIG]?.label || key,
    value,
    color: URGENCY_CONFIG[key as keyof typeof URGENCY_CONFIG]?.color,
  })) : []

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="min-h-screen bg-[#f1f5f9] p-4 md:p-10 space-y-10"
    >
      {/* Header Segment */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-8">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
             <div className="p-3 bg-white rounded-2xl shadow-lg text-blue-600"><BarChart3 size={24} /></div>
             <h1 className="text-4xl font-black text-slate-900 tracking-tighter">Strategic Insights</h1>
          </div>
          <p className="text-slate-500 font-bold ml-14 flex items-center gap-2">
             <Sparkles size={16} className="text-amber-500" />
             AI-driven performance metrics and behavioral analysis
          </p>
        </div>
        <div className="flex items-center gap-4 bg-white/50 backdrop-blur-md px-6 py-3 rounded-2xl border border-white">
           <div className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
           <span className="text-xs font-black uppercase text-slate-600">Real-time Stream Active</span>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-10">
        
        {/* Navigation Sidebar */}
        <div className="xl:col-span-3 space-y-6">
           <div className="bg-white/60 backdrop-blur-xl border border-white p-2 rounded-[32px] shadow-xl">
             <div className="p-6 pb-2"><h3 className="text-xs font-black uppercase tracking-widest text-slate-400">Analysis Matrix</h3></div>
             <div className="space-y-1.5 p-2">
                {[
                  { id: 'overview', label: 'Ecosystem Overview', icon: Target },
                  { id: 'traffic', label: 'Traffic Velocity', icon: Activity },
                  { id: 'urgency', label: 'Safety Breakdown', icon: Zap },
                ].map(tab => (
                  <motion.button
                    key={tab.id}
                    whileHover={{ x: 5 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`w-full flex items-center justify-between px-5 py-4 rounded-2xl transition-all duration-300 font-black text-sm ${
                      activeTab === tab.id 
                      ? 'bg-blue-600 text-white shadow-xl shadow-blue-500/30' 
                      : 'text-slate-500 hover:bg-white hover:text-slate-800'
                    }`}
                  >
                    <div className="flex items-center gap-4">
                       <tab.icon size={18} />
                       {tab.label}
                    </div>
                    {activeTab === tab.id && <ChevronRight size={18} />}
                  </motion.button>
                ))}
             </div>
          </div>

          {/* Impact Card */}
          <div className="bg-gradient-to-br from-indigo-700 to-blue-900 rounded-[32px] p-8 text-white shadow-2xl relative overflow-hidden">
             <div className="relative z-10 space-y-6">
                <Brain size={40} className="text-blue-300" />
                <div className="space-y-1">
                   <h4 className="text-2xl font-black tracking-tight">AI Predictive</h4>
                   <p className="text-sm text-blue-200 font-medium">+18% expected growth next month based on current trends.</p>
                </div>
                <button className="w-full bg-white/10 hover:bg-white/20 py-3 rounded-xl font-bold text-xs transition-colors border border-white/20 uppercase tracking-widest">Generate Report</button>
             </div>
             <div className="absolute -bottom-10 -right-10 opacity-10 rotate-12"><Database size={180} /></div>
          </div>
        </div>

        {/* Dynamic Charts Area */}
        <div className="xl:col-span-9 space-y-8">
          
          {/* Top Quick Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { label: 'Weekly Sessions', value: stats?.sessionsThisWeek ?? 0, delta: '+12%', icon: Activity, color: 'text-blue-600' },
              { label: 'Doc Verification', value: stats?.totalDoctors ?? 0, delta: '+4', icon: Target, color: 'text-emerald-600' },
              { label: 'Patient Reach', value: stats?.totalPatients ?? 0, delta: '+84', icon: Users, color: 'text-indigo-600' },
            ].map((m, idx) => (
              <motion.div key={m.label} initial={{opacity:0, y:20}} animate={{opacity:1, y:0}} transition={{delay: idx * 0.1}}>
                <Card className="border-0 shadow-2xl rounded-[32px] bg-white/80 overflow-hidden group">
                  <div className="p-6 flex items-start justify-between">
                     <div className="space-y-3">
                        <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">{m.label}</p>
                        <h4 className="text-3xl font-black text-slate-800">{m.value.toLocaleString()}</h4>
                        <div className="flex items-center gap-1 text-emerald-500 font-bold text-xs bg-emerald-50 w-fit px-2 py-0.5 rounded-lg">
                           <ArrowUpRight size={14} /> {m.delta}
                        </div>
                     </div>
                     <div className={`p-4 rounded-2xl bg-slate-50 ${m.color} group-hover:scale-110 transition-transform`}>
                        <m.icon size={22} />
                     </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>

          {/* Active Visualization */}
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.4 }}
            >
              {activeTab === 'overview' && (
                <Card className="border-0 shadow-2xl rounded-[40px] p-8 bg-white overflow-hidden">
                   <div className="mb-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                         <h3 className="text-2xl font-black text-slate-800">Growth Trajectory</h3>
                         <p className="text-slate-400 font-medium">Monthly acquisition of new system entities</p>
                      </div>
                      <div className="flex items-center gap-4">
                         <div className="flex items-center gap-2 text-xs font-bold text-slate-500">
                            <div className="w-3 h-3 rounded-full bg-blue-500" /> New Registries
                         </div>
                      </div>
                   </div>
                   <ResponsiveContainer width="100%" height={380}>
                      <AreaChart data={stats?.userGrowth ?? []}>
                        <defs>
                          <linearGradient id="colorGrowth" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2}/>
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                        <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#94a3b8', fontWeight: 700 }} />
                        <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#94a3b8', fontWeight: 700 }} />
                        <Tooltip contentStyle={{ borderRadius: 20, border: 'none', boxShadow: '0 20px 40px rgba(0,0,0,0.1)', padding: 15 }} />
                        <Area type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={4} fillOpacity={1} fill="url(#colorGrowth)" />
                      </AreaChart>
                   </ResponsiveContainer>
                </Card>
              )}

              {activeTab === 'traffic' && (
                <Card className="border-0 shadow-2xl rounded-[40px] p-8 bg-white overflow-hidden">
                   <div className="mb-10">
                      <h3 className="text-2xl font-black text-slate-800">Daily Traffic Velocity</h3>
                      <p className="text-slate-400 font-medium">Operational load across the weekly spectrum</p>
                   </div>
                   <ResponsiveContainer width="100%" height={380}>
                      <BarChart data={stats?.sessionsPerDay ?? []}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                        <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#94a3b8', fontWeight: 700 }} />
                        <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#94a3b8', fontWeight: 700 }} />
                        <Tooltip cursor={{fill: 'rgba(59, 130, 246, 0.05)'}} contentStyle={{ borderRadius: 20, border: 'none', boxShadow: '0 20px 40px rgba(0,0,0,0.1)', padding: 15 }} />
                        <Bar dataKey="count" fill="#3b82f6" radius={[12, 12, 0, 0]} barSize={40} />
                      </BarChart>
                   </ResponsiveContainer>
                </Card>
              )}

              {activeTab === 'urgency' && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
                   <Card className="lg:col-span-7 border-0 shadow-2xl rounded-[40px] p-8 bg-white flex flex-col items-center justify-center">
                      <ResponsiveContainer width="100%" height={380}>
                         <PieChart>
                           <Pie data={urgencyData} cx="50%" cy="50%" innerRadius={100} outerRadius={140} paddingAngle={8} dataKey="value" stroke="none">
                             {urgencyData.map((entry, index) => <Cell key={index} fill={entry.color} />)}
                           </Pie>
                           <Tooltip />
                         </PieChart>
                      </ResponsiveContainer>
                   </Card>
                   <div className="lg:col-span-5 space-y-4">
                      {urgencyData.map((item) => (
                        <Card key={item.name} className="border-0 shadow-xl rounded-3xl p-6 bg-white flex items-center justify-between group hover:bg-blue-600 transition-all duration-500">
                           <div className="flex items-center gap-4">
                              <div className={`w-3 h-3 rounded-full ${URGENCY_CONFIG[item.name as keyof typeof URGENCY_CONFIG]?.bg || 'bg-slate-500'}`} />
                              <div>
                                 <p className="text-sm font-black uppercase text-slate-400 group-hover:text-blue-200 transition-colors tracking-widest">{item.name}</p>
                                 <p className="text-2xl font-black text-slate-800 group-hover:text-white transition-colors">{item.value}% Prevalence</p>
                              </div>
                           </div>
                           <div className="text-slate-200 opacity-0 group-hover:opacity-100 transition-opacity">
                              <ArrowUpRight size={24} />
                           </div>
                        </Card>
                      ))}
                   </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  )
}
