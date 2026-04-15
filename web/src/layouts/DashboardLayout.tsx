import { Link, Outlet, useLocation } from "react-router-dom"
import { useState } from "react"
import {
  LayoutDashboard,
  Building2,
  Users,
  BarChart3,
  LogOut,
  Bell,
  Menu
} from "lucide-react"
import { cn } from "../../lib/utils"
import "../../styles/layout.css"

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Hospitals', href: '/hospitals', icon: Building2 },
  { name: 'Users', href: '/users', icon: Users },
  { name: 'Statistics', href: '/statistics', icon: BarChart3 },
]

export default function DashboardLayout() {
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className={cn("main-layout", collapsed && "collapsed")}>
      <aside className="sidebar-wrapper">
        <div className="sidebar-logo">
          <div className="logo-icon" />
          {!collapsed && <span className="logo-text">Medical AI</span>}
        </div>

        <nav className="nav-container">
          {navigation.map((item) => (
            <Link
              key={item.name}
              to={item.href}
              className={cn(
                "nav-item",
                location.pathname === item.href && "nav-item--active"
              )}
            >
              <item.icon size={20} />
              {!collapsed && item.name}
            </Link>
          ))}
        </nav>

        <div className="sidebar-footer">
          <button className="nav-item logout-btn">
            <LogOut size={20} />
            {!collapsed && "Sign Out"}
          </button>
        </div>
      </aside>

      <div className="content-area">
        <header className="main-header">
          <div className="header-left">
            <button
              className="menu-btn"
              onClick={() => setCollapsed(!collapsed)}
            >
              <Menu size={22} />
            </button>
            <h2 className="header-title">System Overview</h2>
          </div>

          <div className="header-right">
            <button className="icon-btn">
              <Bell size={20} />
            </button>

            <div className="user-profile">
              <div className="user-info">
                <p className="user-name">Admin User</p>
                <p className="user-role">Super Admin</p>
              </div>
              <div className="avatar-circle">AU</div>
            </div>
          </div>
        </header>

        <main className="main-content">
          <Outlet />
        </main>
      </div>
    </div>
  )
}