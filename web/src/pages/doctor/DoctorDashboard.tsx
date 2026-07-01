import { motion, AnimatePresence } from 'framer-motion'
import {
  Calendar, Clock, Users, BarChart2, TrendingUp, Eye,
  Stethoscope, ChevronRight, ChevronLeft, Zap,
} from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useDoctorDashboard } from '@/hooks/useDoctor'
import { AIReportCard } from '@/components/doctor/AIReportCard'
import { StatusBadge } from '@/components/ui/Badge'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { formatDateTime } from '@/lib/utils'
import { useThemeStore } from '@/store/themeStore'
import { useNavigate } from 'react-router-dom'
import { ROUTES } from '@/constants/config'
import { useAuthStore } from '@/store/authStore'
import { useNotificationStore } from '@/store/notificationStore'
import { useLanguage } from '@/lib/language'

// ─── Theme tokens ──────────────────────────────────────────────────────────────
const DARK = {
  page: 'linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e1b4b 100%)',
  card: 'rgba(15,23,42,0.72)',
  cardBorder: 'rgba(255,255,255,0.07)',
  cardShadow: '0 8px 40px rgba(0,0,0,0.32)',
  headerBorder: 'rgba(255,255,255,0.06)',
  title: '#ffffff',
  subtitle: 'rgba(148,163,184,0.85)',
  rowBg: 'rgba(255,255,255,0.03)',
  rowBorder: 'rgba(255,255,255,0.06)',
  rowHoverBg: 'rgba(99,102,241,0.09)',
  rowHoverBorder: 'rgba(99,102,241,0.28)',
  mutedText: 'rgba(148,163,184,0.7)',
  avatarBorder: '#0f172a',
  gridLine: 'rgba(255,255,255,0.04)',
  axisColor: 'rgba(148,163,184,0.7)',
  tooltipBg: 'rgba(15,23,42,0.96)',
  tooltipBorder: 'rgba(99,102,241,0.35)',
  tooltipColor: '#ffffff',
  btnBg: 'rgba(99,102,241,0.16)',
  btnColor: '#818cf8',
  btnBorder: 'rgba(99,102,241,0.28)',
  btnGreenBg: 'rgba(16,185,129,0.13)',
  btnGreenColor: '#34d399',
  btnGreenBorder: 'rgba(16,185,129,0.28)',
  emptyIconBg: 'rgba(99,102,241,0.11)',
  emptyIconBorder: 'rgba(99,102,241,0.22)',
  emptyIconColor: 'rgba(99,102,241,0.55)',
  statText: '#ffffff',
  statSubText: 'rgba(148,163,184,0.9)',
  statCardBg: 'rgba(15,23,42,0.62)',
  statCardBorder: 'rgba(255,255,255,0.08)',
  toggleBg: 'rgba(99,102,241,0.15)',
  toggleBorder: 'rgba(99,102,241,0.35)',
  toggleColor: '#818cf8',
}

const LIGHT = {
  page: 'linear-gradient(135deg, #f0f4ff 0%, #ffffff 50%, #f5f3ff 100%)',
  card: '#ffffff',
  cardBorder: 'rgba(0,0,0,0.07)',
  cardShadow: '0 4px 24px rgba(99,102,241,0.08)',
  headerBorder: 'rgba(0,0,0,0.07)',
  title: '#111827',
  subtitle: '#6b7280',
  rowBg: '#fafafa',
  rowBorder: 'rgba(0,0,0,0.06)',
  rowHoverBg: '#eef2ff',
  rowHoverBorder: 'rgba(99,102,241,0.3)',
  mutedText: '#9ca3af',
  avatarBorder: '#ffffff',
  gridLine: 'rgba(0,0,0,0.05)',
  axisColor: '#9ca3af',
  tooltipBg: '#ffffff',
  tooltipBorder: 'rgba(99,102,241,0.25)',
  tooltipColor: '#111827',
  btnBg: '#eef2ff',
  btnColor: '#6366f1',
  btnBorder: 'rgba(99,102,241,0.25)',
  btnGreenBg: '#ecfdf5',
  btnGreenColor: '#059669',
  btnGreenBorder: 'rgba(16,185,129,0.25)',
  emptyIconBg: '#eef2ff',
  emptyIconBorder: 'rgba(99,102,241,0.2)',
  emptyIconColor: '#6366f1',
  statText: '#111827',
  statSubText: '#6b7280',
  statCardBg: '#ffffff',
  statCardBorder: 'rgba(0,0,0,0.08)',
  toggleBg: '#f3f4f6',
  toggleBorder: 'rgba(0,0,0,0.1)',
  toggleColor: '#374151',
}

// ─── Stat Card ─────────────────────────────────────────────────────────────────
interface StatCardProps {
  title: string
  value: string | number
  icon: React.ReactNode
  gradient: string
  glowColor: string
  index: number
  tk: typeof DARK
}

function StatCard({ title, value, icon, gradient, glowColor, index, tk }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 28, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ delay: index * 0.08, duration: 0.45, ease: 'easeOut' }}
      whileHover={{ y: -4, scale: 1.02 }}
      className="relative overflow-hidden rounded-2xl p-6 cursor-default"
      style={{
        background: tk.statCardBg,
        backdropFilter: 'blur(16px)',
        border: `1px solid ${tk.statCardBorder}`,
        boxShadow: `0 0 32px ${glowColor}18, 0 6px 24px rgba(0,0,0,0.12)`,
      }}
    >
      <div className="absolute -top-6 -right-6 w-28 h-28 rounded-full opacity-[0.15] blur-2xl"
        style={{ background: gradient }} />
      <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-4"
        style={{ background: gradient, boxShadow: `0 4px 18px ${glowColor}40` }}>
        <div className="text-white">{icon}</div>
      </div>
      <p className="text-3xl font-black tracking-tight mb-1" style={{ color: tk.statText }}>{value}</p>
      <p className="text-sm font-medium" style={{ color: tk.statSubText }}>{title}</p>
      <div className="absolute bottom-0 left-0 h-0.5 w-full opacity-50" style={{ background: gradient }} />
    </motion.div>
  )
}

// ─── Chart Tooltip ─────────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label, tk }: any) {
  const { t } = useLanguage()
  if (active && payload?.length) {
    return (
      <div className="px-4 py-3 rounded-xl text-sm" style={{
        background: tk.tooltipBg,
        border: `1px solid ${tk.tooltipBorder}`,
        backdropFilter: 'blur(16px)',
        boxShadow: '0 8px 28px rgba(0,0,0,0.18)',
        color: tk.tooltipColor,
      }}>
        <p className="font-bold text-indigo-500 mb-1">{label}</p>
        <p className="font-semibold" style={{ color: tk.tooltipColor }}>{payload[0].value} {t('sessions')}</p>
      </div>
    )
  }
  return null
}

// ─── Main Dashboard ────────────────────────────────────────────────────────────
export default function DoctorDashboard() {
  const { dashboard, isLoading } = useDoctorDashboard()
  const navigate = useNavigate()
  const { token } = useAuthStore()
  const { addNotification } = useNotificationStore()

  // Use the global theme store (toggle is in TopBar)
  const { isDark } = useThemeStore()
  const { t, isRTL } = useLanguage()

  const tk = isDark ? DARK : LIGHT

  if (isLoading) return <PageLoader />

  const stats = [
    { title: t('todaysAppts'), value: dashboard?.todayAppointments ?? 0, icon: <Calendar size={22} />, gradient: 'linear-gradient(135deg,#6366f1,#8b5cf6)', glowColor: '#6366f1' },
    { title: t('Pending'), value: dashboard?.pendingAppointments ?? 0, icon: <Clock size={22} />, gradient: 'linear-gradient(135deg,#f59e0b,#ef4444)', glowColor: '#f59e0b' },
    { title: t('totalPatients'), value: dashboard?.totalPatients ?? 0, icon: <Users size={22} />, gradient: 'linear-gradient(135deg,#10b981,#059669)', glowColor: '#10b981' },
    { title: t('weeklySessions'), value: dashboard?.weekAppointments ?? 0, icon: <BarChart2 size={22} />, gradient: 'linear-gradient(135deg,#3b82f6,#06b6d4)', glowColor: '#3b82f6' },
  ]

  return (
    <motion.div dir={isRTL ? "rtl" : "ltr"} className={`min-h-screen p-6 relative ${isRTL ? "rtl" : ""}`} animate={{ background: tk.page }} transition={{ duration: 0.5 }} style={{ background: tk.page }}>
      {/* Dark-mode blobs */}
      <AnimatePresence>
        {isDark && (
          <motion.div
            key="blobs"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 overflow-hidden pointer-events-none"
          >
            <div className="absolute top-0 right-0 w-[480px] h-[480px] opacity-10 blur-3xl rounded-full"
              style={{ background: 'radial-gradient(circle,#6366f1,transparent)' }} />
            <div className="absolute bottom-0 left-0 w-[380px] h-[380px] opacity-[0.07] blur-3xl rounded-full"
              style={{ background: 'radial-gradient(circle,#10b981,transparent)' }} />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[650px] h-[650px] opacity-[0.04] blur-3xl rounded-full"
              style={{ background: 'radial-gradient(circle,#8b5cf6,transparent)' }} />
            <div className="absolute inset-0 opacity-[0.04]"
              style={{
                backgroundImage: 'linear-gradient(rgba(99,102,241,0.35) 1px,transparent 1px),linear-gradient(90deg,rgba(99,102,241,0.35) 1px,transparent 1px)',
                backgroundSize: '60px 60px',
              }} />
          </motion.div>
        )}
      </AnimatePresence>

      <div className="relative space-y-8 max-w-screen-xl mx-auto">

        {/* ── Header ── */}
        <motion.div
          initial={{ opacity: 0, y: -22 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55 }}
          className="flex items-center justify-between"
        >
          <div className="flex items-center gap-5">
            <div className="p-3.5 rounded-2xl" style={{ background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', boxShadow: '0 8px 28px rgba(99,102,241,0.38)' }}>
              <Stethoscope size={26} className="text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-black tracking-tight" style={{ color: tk.title }}>{t('doctorDashboard')}</h1>
              <p className="text-sm mt-0.5" style={{ color: tk.subtitle }}>{t('welcomeDoctor')}</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Live badge */}
            <div className="flex items-center gap-2 px-4 py-2 rounded-full"
              style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)' }}>
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500" />
              </span>
              <span className="text-emerald-400 text-xs font-semibold">{t('live')}</span>
            </div>
          </div>
        </motion.div>

        {/* ── Stats Grid ── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
          {stats.map((s, i) => <StatCard key={s.title} index={i} tk={tk} {...s} />)}
        </div>

        {/* ── Main Content: Appointments + Chart ── */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

          {/* Appointments — 3 cols */}
          <motion.div className="lg:col-span-3"
            initial={{ opacity: 0, x: -28 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.35, duration: 0.5 }}>
            <motion.div className="rounded-2xl overflow-hidden h-full" animate={{ background: tk.card }}
              style={{ background: tk.card, border: `1px solid ${tk.cardBorder}`, boxShadow: tk.cardShadow }}>

              {/* Card header */}
              <div className="px-6 py-5 flex items-center justify-between"
                style={{ borderBottom: `1px solid ${tk.headerBorder}` }}>
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-xl"
                    style={{ background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', boxShadow: '0 4px 14px rgba(99,102,241,0.32)' }}>
                    <Calendar size={18} className="text-white" />
                  </div>
                  <h2 className="text-base font-bold" style={{ color: tk.title }}>{t('todaysAppts')}</h2>
                </div>
                <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                  onClick={() => navigate(ROUTES.DOCTOR_APPOINTMENTS)}
                  className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg transition-all"
                  style={{ background: tk.btnBg, color: tk.btnColor, border: `1px solid ${tk.btnBorder}` }}>
                  <Eye size={12} /> {t('viewAll')} {isRTL ? <ChevronLeft size={12} /> : <ChevronRight size={12} />}
                </motion.button>
              </div>

              {/* List */}
              <div className="p-5 space-y-3">
                {!dashboard?.todayAppointmentsList?.length ? (
                  <div className="flex flex-col items-center justify-center py-16 text-center">
                    <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
                      style={{ background: tk.emptyIconBg, border: `1px solid ${tk.emptyIconBorder}` }}>
                      <Calendar size={28} style={{ color: tk.emptyIconColor }} />
                    </div>
                    <p className="font-semibold" style={{ color: tk.title }}>{t('noApptsToday')}</p>
                    <p className="text-sm mt-1" style={{ color: tk.mutedText }}>{t('enjoyFreeTime')}</p>
                  </div>
                ) : dashboard.todayAppointmentsList.slice(0, 5).map((appt, idx) => (
                  <motion.div key={appt.id}
                    initial={{ opacity: 0, x: 18 }} animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 + idx * 0.07 }}
                    whileHover={{ x: 4, scale: 1.01 }}
                    onClick={() => navigate(`${ROUTES.DOCTOR_APPOINTMENTS}/${appt.id}`)}
                    className="flex items-center gap-4 p-4 rounded-xl cursor-pointer transition-all duration-200"
                    style={{ background: tk.rowBg, border: `1px solid ${tk.rowBorder}` }}
                    onMouseEnter={e => {
                      (e.currentTarget as HTMLElement).style.background = tk.rowHoverBg
                        ; (e.currentTarget as HTMLElement).style.borderColor = tk.rowHoverBorder
                    }}
                    onMouseLeave={e => {
                      (e.currentTarget as HTMLElement).style.background = tk.rowBg
                        ; (e.currentTarget as HTMLElement).style.borderColor = tk.rowBorder
                    }}
                  >
                    <div className="relative flex-shrink-0">
                      <div className="w-11 h-11 rounded-full flex items-center justify-center overflow-hidden"
                        style={{ background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', boxShadow: '0 4px 12px rgba(99,102,241,0.28)' }}>
                        {appt.patientPhotoUrl
                          ? <img src={appt.patientPhotoUrl} alt={appt.patientName} className="w-full h-full object-cover" />
                          : <span className="text-white text-sm font-bold">{appt.patientName.charAt(0)}</span>}
                      </div>
                      <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-500 rounded-full border-2"
                        style={{ borderColor: tk.avatarBorder }} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold truncate" style={{ color: tk.title }}>{appt.patientName}</p>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <Clock size={11} style={{ color: tk.mutedText }} />
                        <p className="text-xs" style={{ color: tk.mutedText }}>{formatDateTime(appt.scheduledAt)}</p>
                      </div>
                    </div>
                    <StatusBadge status={appt.status} />
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </motion.div>

          {/* Weekly Chart — 2 cols */}
          <motion.div className="lg:col-span-2"
            initial={{ opacity: 0, x: 28 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4, duration: 0.5 }}>
            <motion.div className="rounded-2xl overflow-hidden h-full" animate={{ background: tk.card }}
              style={{ background: tk.card, border: `1px solid ${tk.cardBorder}`, boxShadow: tk.cardShadow }}>

              <div className="px-6 py-5 flex items-center gap-3"
                style={{ borderBottom: `1px solid ${tk.headerBorder}` }}>
                <div className="p-2 rounded-xl"
                  style={{ background: 'linear-gradient(135deg,#3b82f6,#06b6d4)', boxShadow: '0 4px 14px rgba(59,130,246,0.32)' }}>
                  <TrendingUp size={18} className="text-white" />
                </div>
                <h2 className="text-base font-bold" style={{ color: tk.title }}>Weekly {t('sessions')}</h2>
              </div>

              <div className="p-5">
                <ResponsiveContainer width="100%" height={260}>
                  <AreaChart data={dashboard?.weeklySessionsChart ?? []} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#6366f1" stopOpacity={isDark ? 0.5 : 0.3} />
                        <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={tk.gridLine} vertical={false} />
                    <XAxis dataKey="day" tick={{ fontSize: 11, fill: tk.axisColor, fontWeight: 500 }} axisLine={false} tickLine={false} dy={8} />
                    <YAxis tick={{ fontSize: 11, fill: tk.axisColor, fontWeight: 500 }} axisLine={false} tickLine={false} />
                    <Tooltip content={<ChartTooltip tk={tk} />} />
                    <Area type="monotone" dataKey="count" stroke="#6366f1" strokeWidth={2.5}
                      fill="url(#areaGrad)" name={t('sessions')}
                      dot={{ fill: '#6366f1', strokeWidth: 0, r: 4 }}
                      activeDot={{ r: 6, fill: '#818cf8', strokeWidth: 0 }}
                      animationDuration={1400} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          </motion.div>
        </div>

        {/* ── Recent AI Reports ── */}
        <motion.div
          initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5, duration: 0.5 }}>
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl"
                style={{ background: 'linear-gradient(135deg,#10b981,#059669)', boxShadow: '0 4px 14px rgba(16,185,129,0.32)' }}>
                <Zap size={18} className="text-white" />
              </div>
              <h2 className="text-base font-bold" style={{ color: tk.title }}>{t('recentAiReports')}</h2>
            </div>
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
              onClick={() => navigate(ROUTES.DOCTOR_REPORTS)}
              className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg transition-all"
              style={{ background: tk.btnGreenBg, color: tk.btnGreenColor, border: `1px solid ${tk.btnGreenBorder}` }}>
              <Eye size={12} /> {t('viewAll')} {isRTL ? <ChevronLeft size={12} /> : <ChevronRight size={12} />}
            </motion.button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {dashboard?.recentReports?.slice(0, 3).map((report, i) => (
              <AIReportCard key={report.reportId} report={report} index={i}
                onClick={() => navigate(`${ROUTES.DOCTOR_REPORTS}/${report.reportId}`)} />
            ))}
          </div>
        </motion.div>

      </div>
    </motion.div>
  )
}