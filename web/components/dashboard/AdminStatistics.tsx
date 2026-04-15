import { TrendingUp, Users, CalendarDays, Activity, DollarSign, ArrowUpRight } from "lucide-react"
import "../../styles/stats.css"

const monthlyData = [
  { month: "Aug", patients: 820 },
  { month: "Sep", patients: 950 },
  { month: "Oct", patients: 1100 },
  { month: "Nov", patients: 980 },
  { month: "Dec", patients: 1250 },
  { month: "Jan", patients: 1400 },
]

const topSpecialties = [
  { name: "Neurology", percentage: 85, color: "#2563eb" },
  { name: "Cardiology", percentage: 72, color: "#10b981" },
  { name: "Orthopedics", percentage: 64, color: "#f59e0b" },
]

const maxVal = Math.max(...monthlyData.map(d => d.patients));

export function AdminStatistics() {
  return (
    <div className="insane-stats">
      <div className="dashboard-header__welcome">
        <h1 className="dashboard-header__title">Global Metrics <span style={{color: '#2563eb'}}>Live</span></h1>
        <div className="status-badge status-badge--active">System Stable</div>
      </div>

      <div className="glass-metric-grid">
        {[
          { label: "Revenue", val: "$284.5K", icon: DollarSign, trend: "+22%" },
          { label: "Wait Time", val: "12 min", icon: Activity, trend: "-15%" },
          { label: "New Patients", val: "1,400", icon: Users, trend: "+12%" },
          { label: "Daily Appts", val: "142", icon: CalendarDays, trend: "+8%" }
        ].map((m, i) => (
          <div key={i} className="glass-card">
            <m.icon size={20} color="#2563eb" />
            <p className="metric-card__value">{m.val}</p>
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
              <span className="stat-card__label">{m.label}</span>
              <span style={{fontSize: '11px', color: '#10b981', fontWeight: 800}}>{m.trend}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="insane-chart-container">
        <div style={{display: 'flex', justifyContent: 'space-between'}}>
          <h2 style={{margin: 0}}>Patient Growth Flow</h2>
          <ArrowUpRight color="#60a5fa" />
        </div>
        <div className="insane-bar-chart">
          {monthlyData.map((d) => (
            <div key={d.month} className="bar-wrapper">
              <div 
                className="insane-bar" 
                data-value={d.patients}
                style={{ height: `${(d.patients / maxVal) * 100}%` }}
              ></div>
              <span style={{marginTop: '15px', fontSize: '11px', fontWeight: 700}}>{d.month}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="glass-card">
        <h2 style={{fontSize: '18px', fontWeight: 800, marginBottom: '25px'}}>Specialty Domain Performance</h2>
        {topSpecialties.map((s) => (
          <div key={s.name} className="specialty-row">
            <span style={{width: '100px', fontSize: '13px', fontWeight: 700}}>{s.name}</span>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${s.percentage}%` }}></div>
            </div>
            <span style={{fontSize: '13px', fontWeight: 800, color: '#2563eb'}}>{s.percentage}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}