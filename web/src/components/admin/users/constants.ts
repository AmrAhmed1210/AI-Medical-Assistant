import type { UserDto, UserRole } from '@/lib/types'

// ── Role Configuration (Badges) ──────────────────────────────────────
export const ROLE_CONFIG: Record<UserRole, { 
  variant: 'admin' | 'success' | 'info'; 
  icon: React.ReactNode; 
  label: string 
}> = {
  Admin: { variant: 'admin', icon: null, label: 'Administrator' },
  Doctor: { variant: 'success', icon: null, label: 'Physician' },
  Patient: { variant: 'info', icon: null, label: 'Patient' },
}

export const ROLE_FILTERS: Array<{ value: UserRole | ''; label: string; emoji: string }> = [
  { value: '', label: 'All Users', emoji: '📋' },
  { value: 'Admin', label: 'Admins', emoji: '👑' },
  { value: 'Doctor', label: 'Doctors', emoji: '🩺' },
  { value: 'Patient', label: 'Patients', emoji: '👤' },
]

// ── Mock Data for Dev Testing ───────────────────────────────────────
export const MOCK_USERS: UserDto[] = [
  {
    id: '1',
    name: 'Dr. Ahmed Mohamed Ali',
    email: 'ahmed@medbook.com',
    role: 'Doctor',
    isActive: true,
  },
  {
    id: '2',
    name: 'Mona Ibrahim',
    email: 'mona@medbook.com',
    role: 'Patient',
    isActive: true,
  },
  {
    id: '3',
    name: 'Khaled Mahmoud',
    email: 'khaled@medbook.com',
    role: 'Admin',
    isActive: true,
  },
]

// ── Pagination Config ──────────────────────────────────────────────
export const PAGE_SIZE = 10