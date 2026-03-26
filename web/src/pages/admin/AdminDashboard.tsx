import { useEffect, useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { 
  Users, Stethoscope, Activity, TrendingUp, AlertCircle, 
  RefreshCw, Calendar, ArrowUpRight, ArrowDownRight,
  Zap, Shield, BarChart3, PieChart as PieChartIcon
} from 'lucide-react'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Area, AreaChart
} from 'recharts'
import { adminApi } from '@/api/adminApi'
import type { SystemStatsDto } from '@/lib/types'

// ── مكونات الواجهة ───────────────────────────────────────────────────────
const Card = ({ className = '', ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`rounded-3xl border border-white/20 bg-white/80 backdrop-blur-xl ${className}`} {...props} />
)

const CardHeader = ({ className = '', ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`flex flex-col space-y-2 p-6 ${className}`} {...props} />
)

const CardTitle = ({ className = '', ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
  <h3 className={`text-xl font-bold tracking-tight ${className}`} {...props} />
)

const CardDescription = ({ className = '', ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
  <p className={`text-sm ${className}`} {...props} />
)

const CardContent = ({ className = '', ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`p-6 pt-0 ${className}`} {...props} />
)

const Skeleton = ({ className = '', ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`animate-pulse rounded-xl bg-gray-200/80 ${className}`} {...props} />
)

const Badge = ({ 
  variant = 'default', 
  className = '', 
  ...props 
}: React.HTMLAttributes<HTMLSpanElement> & { variant?: 'default' | 'success' | 'destructive' | 'secondary' | 'outline' | 'glow' }) => {
  const variants = {
    default: 'bg-gray-100/80 text-gray-700',
    success: 'bg-gradient-to-r from-green-400 to-emerald-500 text-white shadow-lg shadow-green-500/30',
    destructive: 'bg-gradient-to-r from-red-400 to-rose-500 text-white shadow-lg shadow-red-500/30',
    secondary: 'bg-gradient-to-r from-blue-400 to-indigo-500 text-white shadow-lg shadow-blue-500/30',
    outline: 'border-2 border-gray-200 text-gray-600 bg-transparent',
    glow: 'bg-gradient-to-r from-violet-500 to-purple-600 text-white shadow-xl shadow-purple-500/40 animate-pulse',
  }
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold ${variants[variant]} ${className}`} {...props} />
  )
}

const Button = ({ 
  variant = 'primary', 
  size = 'md', 
  className = '', 
  ...props 
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { 
  variant?: 'primary' | 'outline' | 'ghost' | 'destructive' | 'glass'; 
  size?: 'sm' | 'md' | 'lg' 
}) => {
  const base = 'inline-flex items-center justify-center font-semibold rounded-2xl transition-all duration-300 focus:outline-none focus:ring-4 disabled:opacity-50 disabled:cursor-not-allowed'
  
  const variants = {
    primary: 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 shadow-xl shadow-blue-500/30 hover:shadow-2xl hover:shadow-blue-500/40 hover:-translate-y-0.5',
    outline: 'border-2 border-gray-200 bg-white/80 text-gray-700 hover:bg-white hover:border-gray-300 hover:shadow-lg',
    ghost: 'text-gray-600 hover:text-gray-900 hover:bg-gray-100/80',
    destructive: 'bg-gradient-to-r from-red-500 to-rose-600 text-white hover:from-red-600 hover:to-rose-700 shadow-xl shadow-red-500/30',
    glass: 'bg-white/20 backdrop-blur-md border border-white/30 text-gray-700 hover:bg-white/30 hover:shadow-xl',
  }
  
  const sizes = {
    sm: 'h-9 px-4 text-xs gap-2',
    md: 'h-11 px-5 text-sm gap-2',
    lg: 'h-12 px-6 text-base gap-2.5',
  }
  
  return (
    <button className={`${base} ${variants[variant]} ${sizes[size]} ${className}`} {...props} />
  )
}

// ── ثوابت التصميم ───────────────────────────────────────────────────────
const URGENCY_CONFIG = {
  LOW:       { label: 'منخفض', color: '#22c55e', bg: 'bg-gradient-to-br from-green-400 to-emerald-500', text: 'text-green-700', border: 'border-green-200', glow: 'shadow-green-500/30' },
  MEDIUM:    { label: 'متوسط', color: '#f59e0b', bg: 'bg-gradient-to-br from-amber-400 to-orange-500', text: 'text-amber-700', border: 'border-amber-200', glow: 'shadow-amber-500/30' },
  HIGH:      { label: 'مرتفع', color: '#ef4444', bg: 'bg-gradient-to-br from-red-400 to-rose-500', text: 'text-red-700', border: 'border-red-200', glow: 'shadow-red-500/30' },
  EMERGENCY: { label: 'طوارئ', color: '#7f1d1d', bg: 'bg-gradient-to-br from-rose-600 to-red-700', text: 'text-rose-50', border: 'border-rose-700', glow: 'shadow-rose-500/40' },
}

const STAT_CONFIG = {
  users:    { gradient: 'from-blue-500 via-blue-600 to-indigo-600', icon: Users, shadow: 'shadow-blue-500/30', bg: 'bg-gradient-to-br from-blue-50 to-indigo-50' },
  doctors:  { gradient: 'from-emerald-500 via-teal-500 to-cyan-600', icon: Stethoscope, shadow: 'shadow-emerald-500/30', bg: 'bg-gradient-to-br from-emerald-50 to-teal-50' },
  patients: { gradient: 'from-orange-500 via-amber-500 to-yellow-600', icon: Users, shadow: 'shadow-orange-500/30', bg: 'bg-gradient-to-br from-orange-50 to-amber-50' },
  sessions: { gradient: 'from-rose-500 via-pink-500 to-fuchsia-600', icon: Activity, shadow: 'shadow-rose-500/30', bg: 'bg-gradient-to-br from-rose-50 to-pink-50' },
}

// ── بطاقة الإحصائيات الكبيرة جداً ───────────────────────────────────────
// ── بطاقة الإحصائيات (حجم متوسط مع وضوح أفضل) ──────────────────────────
const StatCard = ({ 
  title, value, icon: Icon, gradient, shadow, index, trend, subtitle 
}: { 
  title: string; value: number; icon: any; 
  gradient: string; shadow: string; index: number; trend?: number; subtitle?: string 
}) => (
  <motion.div
    initial={{ opacity: 0, y: 20, scale: 0.95 }}
    animate={{ opacity: 1, y: 0, scale: 1 }}
    transition={{ delay: index * 0.1, type: "spring", stiffness: 100, damping: 20 }}
    whileHover={{ y: -6, scale: 1.02, transition: { duration: 0.3 } }}
    className="group"
  >
    <Card className="relative overflow-hidden border-0 shadow-xl hover:shadow-2xl transition-all duration-300 bg-white/90 backdrop-blur-xl">
      {/* شريط علوي ملون */}
   
      
      <CardContent className="p-5">
        <div className=" flex-5 flex-col h-full justify-between padding-5">
          {/* العنوان والأيقونة */}
          <div className="flex flex-5 items-start justify-between mb-3 padding-5">
            <div className="space-y-1 flex-5">
              <p className="text-sm font-bold text-gray-700">{title}</p>
              {subtitle && <p className="text-xs text-gray-500 leading-relaxed">{subtitle}</p>}
            </div>
            <motion.div 
              className={`p-3.5 rounded-2xl bg-gradient-to-br ${gradient} text-white shadow-lg ${shadow}`}
              whileHover={{ rotate: [0, -8, 8, 0], scale: 1.1 }}
              transition={{ duration: 0.5 }}
            >
              <Icon size={24} strokeWidth={2.5} />
            </motion.div>
          </div>
          
          {/* القيمة والنسبة */}
          <div className="space-y-3">
            <motion.div 
              className="flex items-baseline gap-2"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 + index * 0.1 }}
            >
              <span className="text-4xl font-black text-gray-900">
                {value.toLocaleString('ar-EG')}
              </span>
            </motion.div>
            
            {trend !== undefined && (
              <motion.div
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + index * 0.1 }}
              >
                <Badge 
                  variant={trend >= 0 ? 'success' : 'destructive'} 
                  className="text-xs px-3 py-1.5 font-semibold"
                >
                  {trend >= 0 ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
                  <span>{Math.abs(trend)}%</span>
                  <span className="mr-1.5 text-xs">عن الشهر الماضي</span>
                </Badge>
              </motion.div>
            )}
          </div>
        </div>
        
        {/* تأثير اللمعان */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
      </CardContent>
    </Card>
  </motion.div>
)


// ── مكون الحالة الفارغة ───────────────────────────────────────────────
const EmptyState = ({ message, icon: Icon = Activity }: { message: string; icon?: any }) => (
  <motion.div 
    initial={{ opacity: 0, scale: 0.9 }}
    animate={{ opacity: 1, scale: 1 }}
    className="flex flex-col items-center justify-center py-16 text-center"
  >
    <motion.div 
      className="p-5 rounded-3xl bg-gradient-to-br from-gray-100 to-gray-50 mb-4 shadow-inner"
      animate={{ y: [0, -5, 0] }}
      transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
    >
      <Icon className="w-8 h-8 text-gray-400" />
    </motion.div>
    <p className="text-sm font-medium text-gray-500">{message}</p>
  </motion.div>
)

// ── المخطط الدائري المحسّن ────────────────────────────────────────────
interface PieChartData {
  name: string;
  value: number;
  color: string;
  config: {
    label: string;
    bg: string;
    text: string;
    border: string;
    glow: string;
  };
}

const EnhancedPieChart = ({ data }: { data: Array<{ name: string; value: number; color: string; config: any }> }) => {
  const total = useMemo(() => data.reduce((sum, item) => sum + item.value, 0), [data])
  
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null
    const item = payload[0].payload
    return (
      <motion.div 
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white/95 backdrop-blur-xl border border-white/30 rounded-2xl shadow-2xl p-4 min-w-[160px]"
      >
        <div className="flex items-center gap-2.5 mb-2">
          <div className="w-4 h-4 rounded-full shadow-lg" style={{ backgroundColor: item.color }} />
          <span className="text-sm font-bold text-gray-800">{item.name}</span>
        </div>
        <p className="text-2xl font-black text-gray-900">{item.value}</p>
        <p className="text-xs text-gray-500 mt-1">{((item.value / total) * 100).toFixed(1)}% من الإجمالي</p>
      </motion.div>
    )
  }

  return (
    <div className="space-y-6">
      <ResponsiveContainer width="100%" height={240}>
        <PieChart>
          <defs>
            {data.map((item, i) => (
              <radialGradient key={i} id={`grad-${i}`} cx="50%" cy="50%" r="80%">
                <stop offset="0%" stopColor={item.color} stopOpacity={1} />
                <stop offset="100%" stopColor={item.color} stopOpacity={0.7} />
              </radialGradient>
            ))}
          </defs>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={65}
            outerRadius={95}
            paddingAngle={5}
            dataKey="value"
            stroke="white"
            strokeWidth={3}
          >
            {data.map((entry, i) => (
              <Cell 
                key={`cell-${i}`} 
                fill={`url(#grad-${i})`}
                className="transition-all duration-300 hover:opacity-80 cursor-pointer"
              />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
        </PieChart>
      </ResponsiveContainer>
      
      {/* وسيلة الإيضاح */}
      <div className="grid grid-cols-2 gap-3">
        {data.map((item, idx) => (
          <motion.div 
            key={item.name}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: idx * 0.1 }}
            whileHover={{ scale: 1.03, x: 5 }}
            className={`flex items-center gap-2.5 p-3 rounded-2xl border ${item.config.border} ${item.config.bg} shadow-lg ${item.config.glow} transition-all`}
          >
            <div className="w-4 h-4 rounded-full shadow-lg" style={{ backgroundColor: item.color }} />
            <span className={`text-xs font-bold ${item.config.text}`}>{item.name}</span>
            <span className={`text-sm font-black mr-auto ${item.config.text}`}>{item.value}</span>
          </motion.div>
        ))}
      </div>
      
      {/* الإجمالي */}
      <motion.div 
        className="text-center pt-4 border-t-2 border-gray-100"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <span className="text-sm text-gray-500 font-medium">إجمالي الحالات: </span>
        <span className="text-2xl font-black bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">{total.toLocaleString('ar-EG')}</span>
      </motion.div>
    </div>
  )
}

// ── المكون الرئيسي ────────────────────────────────────────────────────
export default function AdminDashboard() {
  const [stats, setStats] = useState<SystemStatsDto | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())

  const fetchData = async () => {
    try {
      const data = await adminApi.getStats()
      setStats(data)
      setLastUpdated(new Date())
      setError(null)
    } catch (err: any) {
      console.error('AdminDashboard – getStats failed:', err)
      setError(err?.response?.data?.message ?? err?.message ?? 'حدث خطأ في جلب البيانات')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchData() }, [])

  const urgencyData = useMemo(() => {
    if (!stats?.urgencyDistribution) return []
    return Object.entries(stats.urgencyDistribution).map(([key, value]) => ({
      name: URGENCY_CONFIG[key as keyof typeof URGENCY_CONFIG]?.label ?? key,
      value: value as number,
      color: URGENCY_CONFIG[key as keyof typeof URGENCY_CONFIG]?.color ?? '#6b7280',
      config: URGENCY_CONFIG[key as keyof typeof URGENCY_CONFIG] ?? URGENCY_CONFIG.MEDIUM,
    })).filter(item => item.value > 0)
  }, [stats])

  const sessionsPerDay = stats?.sessionsPerDay ?? []
  const userGrowth = stats?.userGrowth ?? []

  // ── حالة التحميل ────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="space-y-6 min-h-screen bg-gradient-to-br from-gray-50 via-white to-blue-50/30 p-6" dir="rtl">
        <div className="space-y-3">
          <Skeleton className="h-10 w-56" />
          <Skeleton className="h-5 w-40" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
          {[...Array(4)].map((_, i) => (
            <Card key={i} className="p-6">
              <div className="flex justify-between mb-4">
                <div className="space-y-3 flex-1">
                  <Skeleton className="h-5 w-28" />
                  <Skeleton className="h-12 w-20" />
                </div>
                <Skeleton className="w-16 h-16 rounded-2xl" />
              </div>
              <Skeleton className="h-8 w-24" />
            </Card>
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2 p-6"><Skeleton className="h-64 w-full rounded-2xl" /></Card>
          <Card className="p-6"><Skeleton className="h-64 w-full rounded-2xl" /></Card>
        </div>
      </div>
    )
  }

  // ── حالة الخطأ ──────────────────────────────────────────────────────
  if (error) {
    return (
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex flex-col items-center justify-center min-h-[500px] gap-6 p-8 bg-gradient-to-br from-gray-50 via-white to-red-50/30"
        dir="rtl"
      >
        <motion.div 
          className="p-6 rounded-3xl bg-gradient-to-br from-red-50 to-rose-100 border-2 border-red-200 shadow-2xl shadow-red-500/20"
          animate={{ rotate: [0, -5, 5, 0] }}
          transition={{ duration: 0.5 }}
        >
          <AlertCircle className="w-12 h-12 text-red-500" />
        </motion.div>
        <div className="text-center space-y-2">
          <h3 className="text-2xl font-black text-gray-900">تعذر تحميل البيانات</h3>
          <p className="text-gray-500">{error}</p>
        </div>
        <Button onClick={fetchData} variant="outline" size="lg" className="gap-3">
          <RefreshCw className="w-5 h-5" />
          إعادة المحاولة
        </Button>
      </motion.div>
    )
  }

  if (!stats) return null

  // ── الواجهة الرئيسية ───────────────────────────────────────────────
  return (
    <motion.div 
      className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-blue-50/30 p-4 md:p-6 lg:p-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      dir="rtl"
    >
      {/* الهيدر */}
      <motion.div 
        className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 pb-6 mb-6 border-b-2 border-gray-100"
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
      >
        <div className="space-y-2">
          <motion.h1 
            className="text-4xl font-black bg-gradient-to-r from-gray-900 via-blue-900 to-indigo-900 bg-clip-text text-transparent"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
          >
            لوحة تحكم المدير
          </motion.h1>
          <motion.p 
            className="text-sm text-gray-500 flex items-center gap-2 font-medium"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <Calendar className="w-4 h-4" />
            آخر تحديث: {lastUpdated.toLocaleTimeString('ar-EG', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </motion.p>
        </div>
        
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
        >
          <Button 
            variant="glass" 
            size="lg" 
            onClick={fetchData}
            className="gap-3 shadow-xl hover:shadow-2xl"
          >
            <RefreshCw className="w-5 h-5" />
            تحديث البيانات
          </Button>
        </motion.div>
      </motion.div>

      {/* البطاقات الإحصائية الكبيرة جداً */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-8">
        <StatCard 
          title="إجمالي المستخدمين"
          subtitle="المسجلين في النظام"
          value={stats.totalUsers ?? 0}
          icon={STAT_CONFIG.users.icon}
          gradient={STAT_CONFIG.users.gradient}
          shadow={STAT_CONFIG.users.shadow}
          index={0}
          trend={12}
        />
        <StatCard 
          title="الأطباء"
          subtitle="المتاحين للاستشارات"
          value={stats.totalDoctors ?? 0}
          icon={STAT_CONFIG.doctors.icon}
          gradient={STAT_CONFIG.doctors.gradient}
          shadow={STAT_CONFIG.doctors.shadow}
          index={1}
          trend={5}
        />
        <StatCard 
          title="المرضى"
          subtitle="المستفيدين من الخدمة"
          value={stats.totalPatients ?? 0}
          icon={STAT_CONFIG.patients.icon}
          gradient={STAT_CONFIG.patients.gradient}
          shadow={STAT_CONFIG.patients.shadow}
          index={2}
          trend={-2}
        />
        <StatCard 
          title="جلسات اليوم"
          subtitle="المجدولة والمكتملة"
          value={stats.sessionsToday ?? 0}
          icon={STAT_CONFIG.sessions.icon}
          gradient={STAT_CONFIG.sessions.gradient}
          shadow={STAT_CONFIG.sessions.shadow}
          index={3}
          trend={23}
        />
      </div>

      {/* الرسوم البيانية */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        
        {/* المخطط الشريطي */}
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
          className="lg:col-span-2"
        >
          <Card className="h-full border-0 shadow-2xl hover:shadow-3xl transition-all duration-500 bg-white/90 backdrop-blur-xl">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <CardTitle className="text-2xl font-black">الجلسات اليومية</CardTitle>
                  <CardDescription className="text-gray-500 font-medium">تحليل النشاط خلال آخر 30 يوم</CardDescription>
                </div>
                <Badge variant="secondary" className="text-sm shadow-lg">
                  <TrendingUp className="w-4 h-4" />
                  +{sessionsPerDay.slice(-7).reduce((a, b) => a + (b.count || 0), 0)} هذا الأسبوع
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              {sessionsPerDay.length === 0 ? (
                <EmptyState message="لا توجد بيانات جلسات متاحة" icon={BarChart3} />
              ) : (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={sessionsPerDay} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity={1} />
                        <stop offset="100%" stopColor="#6366f1" stopOpacity={0.8} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="4 4" stroke="#f1f5f9" vertical={false} />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 11, fill: '#64748b', fontFamily: 'inherit' }}
                      tickFormatter={(v) => String(v).slice(5)}
                      axisLine={false}
                      tickLine={false}
                      dy={10}
                    />
                    <YAxis 
                      tick={{ fontSize: 11, fill: '#64748b', fontFamily: 'inherit' }} 
                      axisLine={false}
                      tickLine={false}
                      dx={-10}
                    />
                    <Tooltip 
                      cursor={{ fill: 'rgba(59, 130, 246, 0.1)', radius: 8 }}
                      contentStyle={{ 
                        fontSize: 13, borderRadius: 16, border: '1px solid #e2e8f0',
                        background: 'white', boxShadow: '0 10px 40px rgba(0,0,0,0.15)',
                        padding: '12px 16px', fontWeight: 600
                      }}
                      labelStyle={{ fontWeight: 700, color: '#1e293b', marginBottom: '8px' }}
                    />
                    <Bar 
                      dataKey="count" 
                      fill="url(#barGradient)" 
                      radius={[10, 10, 0, 0]} 
                      name="جلسات"
                      animationDuration={1000}
                    />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* المخطط الدائري */}
        <motion.div
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
        >
          <Card className="h-full border-0 shadow-2xl hover:shadow-3xl transition-all duration-500 bg-white/90 backdrop-blur-xl">
            <CardHeader>
              <div className="space-y-1">
                <CardTitle className="text-2xl font-black">توزيع مستوى الطوارئ</CardTitle>
                <CardDescription className="text-gray-500 font-medium">حسب الأولوية والتصنيف</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              {urgencyData.length === 0 ? (
                <EmptyState message="لا توجد بيانات طوارئ" icon={PieChartIcon} />
              ) : (
                <EnhancedPieChart data={urgencyData} />
              )}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* نمو المستخدمين */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <Card className="border-0 shadow-2xl hover:shadow-3xl transition-all duration-500 bg-white/90 backdrop-blur-xl">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <CardTitle className="text-2xl font-black">نمو المستخدمين</CardTitle>
                <CardDescription className="text-gray-500 font-medium">تسجيلات جديدة خلال آخر 30 يوم</CardDescription>
              </div>
              <div className="flex items-center gap-3">
                <Badge variant="glow" className="text-sm">
                  <Zap className="w-4 h-4" />
                  إجمالي: {(userGrowth.slice(-1)[0]?.count ?? 0).toLocaleString('ar-EG')}
                </Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {userGrowth.length === 0 ? (
              <EmptyState message="لا توجد بيانات نمو متاحة" icon={TrendingUp} />
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={userGrowth} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="4 4" stroke="#f1f5f9" vertical={false} />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 11, fill: '#64748b', fontFamily: 'inherit' }}
                    tickFormatter={(v) => String(v).slice(5)}
                    axisLine={false}
                    tickLine={false}
                    dy={10}
                  />
                  <YAxis 
                    tick={{ fontSize: 11, fill: '#64748b', fontFamily: 'inherit' }}
                    axisLine={false}
                    tickLine={false}
                    dx={-10}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      fontSize: 13, borderRadius: 16, border: '1px solid #e2e8f0',
                      background: 'white', boxShadow: '0 10px 40px rgba(0,0,0,0.15)',
                      padding: '12px 16px', fontWeight: 600
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="count"
                    stroke="#3b82f6"
                    strokeWidth={3}
                    fill="url(#areaGradient)"
                    dot={{ fill: '#3b82f6', strokeWidth: 3, r: 4, stroke: 'white' }}
                    activeDot={{ r: 7, strokeWidth: 0, fill: '#2563eb' }}
                    name="مستخدمون جدد"
                    animationDuration={1200}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  )
}