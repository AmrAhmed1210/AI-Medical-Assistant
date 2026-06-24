import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'

export type Lang = 'en' | 'ar'

const LANGUAGE_KEY = 'app_language'

const dictionary = {
  en: {
    search: 'Search...',
    notifications: 'Notifications',
    noNotifications: 'No notifications',
    light: 'Light',
    dark: 'Dark',
    switchToLight: 'Switch to Light Mode',
    switchToDark: 'Switch to Dark Mode',
    logOut: 'Log Out',
    loggedOut: 'Logged out successfully',
    dashboard: 'Dashboard',
    todaysVisits: "Today's Visits",
    appointments: 'Appointments',
    patients: 'Patients',
    reviews: 'Reviews',
    messages: 'Messages',
    manageStaff: 'Manage Staff',
    schedule: 'Schedule',
    myProfile: 'My Profile',
    users: 'Users',
    statistics: 'Statistics',
    applications: 'Applications',
    supportCenter: 'Support Center',
    systemAdministrator: 'System Administrator',
    secretary: 'Secretary',
    doctor: 'Doctor',
    smartMedical: 'Smart Medical',
    newBooking: 'New booking',
    cancelled: 'Cancelled',
    appointmentUpdate: 'Appointment Update',
    newConsultation: 'New Consultation',
    connectionRestored: 'Connection restored',
    supportTeam: 'Support Team',
    language: 'العربية',
  },
  ar: {
    search: 'بحث...',
    notifications: 'الإشعارات',
    noNotifications: 'لا توجد إشعارات',
    light: 'فاتح',
    dark: 'داكن',
    switchToLight: 'التبديل للوضع الفاتح',
    switchToDark: 'التبديل للوضع الداكن',
    logOut: 'تسجيل الخروج',
    loggedOut: 'تم تسجيل الخروج بنجاح',
    dashboard: 'لوحة التحكم',
    todaysVisits: 'زيارات اليوم',
    appointments: 'المواعيد',
    patients: 'المرضى',
    reviews: 'التقييمات',
    messages: 'الرسائل',
    manageStaff: 'إدارة الفريق',
    schedule: 'الجدول',
    myProfile: 'ملفي الشخصي',
    users: 'المستخدمون',
    statistics: 'الإحصائيات',
    applications: 'طلبات الانضمام',
    supportCenter: 'مركز الدعم',
    systemAdministrator: 'مدير النظام',
    secretary: 'السكرتير',
    doctor: 'الطبيب',
    smartMedical: 'طبي ذكي',
    newBooking: 'حجز جديد',
    cancelled: 'تم الإلغاء',
    appointmentUpdate: 'تحديث الموعد',
    newConsultation: 'استشارة جديدة',
    connectionRestored: 'تمت استعادة الاتصال',
    supportTeam: 'فريق الدعم',
    language: 'English',
  },
} as const

type TranslationKey = keyof typeof dictionary.en

type LanguageContextValue = {
  lang: Lang
  isRTL: boolean
  setLanguage: (nextLang: Lang) => void
  toggleLanguage: () => void
  t: (key: TranslationKey) => string
}

const LanguageContext = createContext<LanguageContextValue | null>(null)

function getInitialLanguage(): Lang {
  const stored = localStorage.getItem(LANGUAGE_KEY)
  if (stored === 'ar' || stored === 'en') return stored
  return navigator.language?.toLowerCase().startsWith('ar') ? 'ar' : 'en'
}

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLang] = useState<Lang>(getInitialLanguage)
  const isRTL = lang === 'ar'

  useEffect(() => {
    localStorage.setItem(LANGUAGE_KEY, lang)
    document.documentElement.lang = lang
    document.documentElement.dir = isRTL ? 'rtl' : 'ltr'
    document.body.dir = isRTL ? 'rtl' : 'ltr'
  }, [lang, isRTL])

  const setLanguage = useCallback((nextLang: Lang) => setLang(nextLang), [])
  const toggleLanguage = useCallback(() => setLang(current => current === 'ar' ? 'en' : 'ar'), [])
  const t = useCallback((key: TranslationKey) => dictionary[lang][key] ?? dictionary.en[key], [lang])

  const value = useMemo(() => ({ lang, isRTL, setLanguage, toggleLanguage, t }), [lang, isRTL, setLanguage, toggleLanguage, t])

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
}

export function useLanguage() {
  const context = useContext(LanguageContext)
  if (!context) throw new Error('useLanguage must be used inside LanguageProvider')
  return context
}
