import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface ThemeState {
  isDark: boolean
  toggleTheme: () => void
  setTheme: (isDark: boolean) => void
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set) => ({
      isDark: true, // Default to dark mode based on the new premium design
      toggleTheme: () => set((state) => ({ isDark: !state.isDark })),
      setTheme: (isDark) => set({ isDark }),
    }),
    {
      name: 'medbook-theme-storage',
    }
  )
)

export const DARK_THEME = {
  pageBg: 'linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e1b4b 100%)',
  sidebarBg: 'rgba(15,23,42,0.95)',
  sidebarBorder: 'rgba(255,255,255,0.06)',
  topbarBg: 'rgba(15,23,42,0.8)',
  topbarBorder: 'rgba(255,255,255,0.06)',
  textMain: '#ffffff',
  textMuted: 'rgba(148,163,184,0.85)',
  navItemHover: 'rgba(99,102,241,0.1)',
  navItemActive: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
  navTextHover: '#ffffff',
  navTextMain: 'rgba(148,163,184,0.9)',
}

export const LIGHT_THEME = {
  pageBg: 'linear-gradient(135deg, #f0f4ff 0%, #ffffff 50%, #f5f3ff 100%)',
  sidebarBg: '#ffffff',
  sidebarBorder: 'rgba(0,0,0,0.06)',
  topbarBg: '#ffffff',
  topbarBorder: 'rgba(0,0,0,0.06)',
  textMain: '#111827',
  textMuted: '#6b7280',
  navItemHover: '#f3f4f6',
  navItemActive: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
  navTextHover: '#111827',
  navTextMain: '#4b5563',
}
