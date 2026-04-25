import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import AsyncStorage from '@react-native-async-storage/async-storage'

export interface NotificationState {
  unreadMessages: number
  unreadCounts: Record<number, number>
  latestMessagePayload: any | null
  
  incrementSessionMessage: (sessionId: number) => void
  clearSessionMessages: (sessionId: number) => void
  clearAllMessages: () => void

  setLatestMessagePayload: (payload: any) => void
}

// Build a user-specific storage key
const getUserStoreKey = async (): Promise<string> => {
  const patientId = await AsyncStorage.getItem('patientId') || await AsyncStorage.getItem('userId') || 'guest'
  return `patient-notification-store-${patientId}`
}

export const useNotificationStore = create<NotificationState>()(
  persist(
    (set) => ({
      unreadMessages: 0,
      unreadCounts: {},
      latestMessagePayload: null,
      
      incrementSessionMessage: (sessionId) => set((state) => {
        const currentCount = state.unreadCounts[sessionId] || 0;
        return {
          unreadCounts: { ...state.unreadCounts, [sessionId]: currentCount + 1 },
          unreadMessages: state.unreadMessages + 1
        };
      }),
      
      clearSessionMessages: (sessionId) => set((state) => {
        const count = state.unreadCounts[sessionId] || 0;
        if (count === 0) return state;
        const newCounts = { ...state.unreadCounts };
        delete newCounts[sessionId];
        return {
          unreadCounts: newCounts,
          unreadMessages: Math.max(0, state.unreadMessages - count)
        };
      }),

      clearAllMessages: () => set({ unreadMessages: 0, unreadCounts: {} }),

      setLatestMessagePayload: (payload) => set({ latestMessagePayload: payload })
    }),
    {
      name: 'patient-notification-store',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({ unreadMessages: state.unreadMessages, unreadCounts: state.unreadCounts }),
    }
  )
)
