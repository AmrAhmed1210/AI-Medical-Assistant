import type { UserDto, UserRole } from '@/lib/types'

// ── تكوين الشارات (Badges) ──────────────────────────────────────────
export const ROLE_CONFIG: Record<UserRole, { 
  variant: 'admin' | 'success' | 'info'; 
  icon: React.ReactNode; 
  label: string 
}> = {
  Admin: { variant: 'admin', icon: null, label: 'مدير' },
  Doctor: { variant: 'success', icon: null, label: 'طبيب' },
  Patient: { variant: 'info', icon: null, label: 'مريض' },
}

export const ROLE_FILTERS: Array<{ value: UserRole | ''; label: string; emoji: string }> = [
  { value: '', label: 'الكل', emoji: '📋' },
  { value: 'Admin', label: 'مدراء', emoji: '👑' },
  { value: 'Doctor', label: 'أطباء', emoji: '🩺' },
  { value: 'Patient', label: 'مرضى', emoji: '👤' },
]

// ── بيانات تجريبية للاختبار ───────────────────────────────────────
export const MOCK_USERS: UserDto[] = [
  {
    userId: '1',
    fullName: 'د. أحمد محمد علي',
    email: 'ahmed@medbook.com',
    role: 'Doctor',
    isActive: true,
   
  },
  {
    userId: '2',
    fullName: 'منى إبراهيم',
    email: 'mona@medbook.com',
    role: 'Patient',
    isActive: true,

  },
  {
    userId: '3',
    fullName: 'خالد محمود',
    email: 'khaled@medbook.com',
    role: 'Admin',
    isActive: true,

  },
]

// ── إعدادات الصفحة ────────────────────────────────────────────────
export const PAGE_SIZE = 10