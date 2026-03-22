import { Users, Stethoscope, CalendarDays, DollarSign, Activity, TrendingUp, ChevronRight } from "lucide-react"
import { adminStats } from "../../lib/data"
import "../../styles/ultimate-dashboard.css"

const recentActivity = [
  { id: 1, action: "New Doctor Registered", detail: "Dr. Sarah Kim - Dermatology", time: "2m ago", type: "doctor", color: "#dcfce7" },
  { id: 2, action: "AI Report Generated", detail: "Patient: Ahmed Hassan - Critical", time: "30m ago", type: "report", color: "#ffedd5" },
  { id: 3, action: "New Hospital Added", detail: "City Medical Center - Downtown", time: "3h ago", type: "hospital", color: "#eff6ff" },
]

export function AdminDashboard() {
  const stats = [
    { label: "Total Users", value: adminStats.totalUsers.toLocaleString(), icon: Users, bg: "#eff6ff", color: "#2563eb", trend: "+12.5%" },
    { label: "Active Doctors", value: adminStats.totalDoctors.toLocaleString(), icon: Stethoscope, bg: "#f0fdf4", color: "#16a34a", trend: "+4.2%" },
    { label: "Appointments", value: adminStats.totalAppointments.toLocaleString(), icon: CalendarDays, bg: "#fffbeb", color: "#d97706", trend: "+18%" },
    { label: "Net Revenue", value: `$${(adminStats.revenue / 1000).toFixed(0)}K`, icon: DollarSign, bg: "#f5f3ff", color: "#7c3aed", trend: "+22%" },
  ]

  return (
    <div className="ultimate-dashboard">
      <header className="dashboard-header__welcome">
        <div>
          <h1 className="dashboard-header__title">Healthcare Dashboard</h1>
          <p style={{ margin: 0, fontSize: '12px', color: '#64748b' }}>System Overview & Real-time Analytics</p>
        </div>
      </header>

      <div className="ultimate-stats-grid">
        {stats.map((s) => (
          <div key={s.label} className="premium-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '15px' }}>
              <div className="premium-card__icon" style={{ backgroundColor: s.bg, color: s.color }}>
                <s.icon size={24} />
              </div>
              <span style={{ fontSize: '11px', fontWeight: 700, color: '#16a34a', height: 'fit-content', padding: '4px 8px', background: '#f0fdf4', borderRadius: '20px' }}>
                {s.trend}
              </span>
            </div>
            <p className="premium-card__value">{s.value}</p>
            <p style={{ margin: 0, fontSize: '13px', color: '#64748b', fontWeight: 500 }}>{s.label}</p>
          </div>
        ))}
      </div>

      <div className="active-bar" style={{ padding: '24px', background: 'linear-gradient(90deg, #2563eb, #3b82f6)', display: 'flex', alignItems: 'center', gap: '20px', borderRadius: '20px', color: 'white' }}>
        <div style={{ background: 'rgba(255,255,255,0.2)', padding: '12px', borderRadius: '12px' }}>
          <Activity size={24} />
        </div>
        <div style={{ flex: 1 }}>
          <p style={{ margin: 0, fontSize: '20px', fontWeight: 800 }}>{adminStats.activeConsultations}</p>
          <p style={{ margin: 0, fontSize: '12px', opacity: 0.9 }}>Live Consultations in Progress</p>
        </div>
        <div className="active-bar__pulse" style={{ background: 'white', width: '12px', height: '12px', borderRadius: '50%', position: 'relative' }}>
          <div className="active-bar__ping" style={{ position: 'absolute', width: '100%', height: '100%', background: 'white', borderRadius: '50%', animation: 'ping 1.5s infinite' }}></div>
        </div>
      </div>

      <section className="activity-feed" style={{ background: 'white', padding: '24px', borderRadius: '24px', border: '1px solid #e2e8f0' }}>
        <h2 style={{ fontSize: '18px', fontWeight: 800, marginBottom: '20px' }}>Recent System Activity</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {recentActivity.map((item) => (
            <div key={item.id} className="activity-item" style={{ display: 'flex', gap: '16px', padding: '12px', borderRadius: '12px', borderBottom: '1px solid #f1f5f9' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: item.color, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Activity size={18} />
              </div>
              <div style={{ flex: 1 }}>
                <p style={{ margin: 0, fontSize: '14px', fontWeight: 700 }}>{item.action}</p>
                <p style={{ margin: 0, fontSize: '12px', color: '#64748b' }}>{item.detail}</p>
              </div>
              <div style={{ textAlign: 'right' }}>
                <p style={{ fontSize: '10px', color: '#94a3b8', margin: 0 }}>{item.time}</p>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}