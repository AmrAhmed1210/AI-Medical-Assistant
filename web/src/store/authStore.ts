import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { UserDto, UserRole } from '@/lib/types'
import { TOKEN_KEY, USER_KEY } from '@/constants/config'

interface AuthState {
  user: UserDto | null
  token: string | null
  role: UserRole | null
  isAuthenticated: boolean
  isLoading: boolean

  setAuth: (user: UserDto, token: string) => void
  logout: () => void
  setLoading: (loading: boolean) => void
  updateUser: (user: Partial<UserDto>) => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      role: null,
      isAuthenticated: false,
      isLoading: false,

      setAuth: (user, token) => {
        localStorage.setItem(TOKEN_KEY, token)
        localStorage.setItem(USER_KEY, JSON.stringify(user))
        set({ user, token, role: user.role, isAuthenticated: true })
      },

      logout: () => {
        localStorage.removeItem(TOKEN_KEY)
        localStorage.removeItem(USER_KEY)
        set({ user: null, token: null, role: null, isAuthenticated: false })
      },

      setLoading: (isLoading) => set({ isLoading }),

      updateUser: (partial) =>
        set((state) => ({
          user: state.user ? { ...state.user, ...partial } : null,
        })),
    }),
    {
      name: 'medbook-auth',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        role: state.role,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
)
