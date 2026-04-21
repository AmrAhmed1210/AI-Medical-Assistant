import type { UserDto, UserRole } from '@/lib/types'

// ── Role Configuration (Badges) ──────────────────────────────────────
export const ROLE_CONFIG: Record<UserRole, { 
  variant: 'admin' | 'success' | 'info'; 
  icon: React.ReactNode; 
  label: string 
}> = {
  Admin: { variant: 'admin', icon: null, label: 'Admin / مدير نظام' },
  Doctor: { variant: 'success', icon: null, label: 'Doctor / طبيب' },
  Patient: { variant: 'info', icon: null, label: 'Patient / مريض' },
}

export const ROLE_FILTERS: Array<{ value: UserRole | ''; label: string; emoji: string }> = [
  { value: '', label: 'All / الكل', emoji: '📋' },
  { value: 'Admin', label: 'Admins / المدراء', emoji: '👑' },
  { value: 'Doctor', label: 'Doctors / الأطباء', emoji: '🩺' },
  { value: 'Patient', label: 'Patients / المرضى', emoji: '👤' },
]

// ── Mock Data for Dev Testing ───────────────────────────────────────
export const MOCK_USERS: UserDto[] = [
  {
    id: 1,
    name: 'Dr. Ahmed Mohamed Ali',
    email: 'ahmed@medbook.com',
    role: 'Doctor',
    isActive: true,
  },
  {
    id: 2,
    name: 'Mona Ibrahim',
    email: 'mona@medbook.com',
    role: 'Patient',
    isActive: true,
  },
  {
    id: 3,
    name: 'Khaled Mahmoud',
    email: 'khaled@medbook.com',
    role: 'Admin',
    isActive: true,
  },
]

// ── Pagination Config ──────────────────────────────────────────────
export const PAGE_SIZE = 10