import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'

export function DashboardLayout() {
  return (
    <div className="min-h-screen bg-gray-50 font-tajawal" dir="rtl">
      <Sidebar />
      <TopBar />
      <main className="mr-64 mt-16 p-6 min-h-[calc(100vh-4rem)]">
        <Outlet />
      </main>
    </div>
  )
}
