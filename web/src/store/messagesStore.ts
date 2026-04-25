import { create } from 'zustand'
import { consultApi } from '@/api/consultApi'
import type { SessionDto, SessionDetailDto, MessageDto } from '@/lib/types'

interface MessagesState {
  sessions: SessionDto[]
  selectedSession: SessionDetailDto | null
  isLoadingSessions: boolean
  
  fetchSessions: () => Promise<void>
  openSession: (sessionId: string | number) => Promise<void>
  sendMessage: (content: string) => Promise<void>
  deleteSession: (id: string | number) => Promise<void>
  
  handleIncomingMessage: (payload: any, isChatOpen: boolean) => void
}

export const useMessagesStore = create<MessagesState>((set, get) => ({
  sessions: [],
  selectedSession: null,
  isLoadingSessions: false,

  fetchSessions: async () => {
    set({ isLoadingSessions: true })
    try {
      const data = await consultApi.getSessions()
      const sorted = data.sort((a, b) => {
        const timeA = new Date(a.lastMessageAt || a.updatedAt || a.createdAt).getTime()
        const timeB = new Date(b.lastMessageAt || b.updatedAt || b.createdAt).getTime()
        return timeB - timeA
      })
      set({ sessions: sorted })
    } finally {
      set({ isLoadingSessions: false })
    }
  },

  openSession: async (sessionId) => {
    try {
      const details = await consultApi.getSession(String(sessionId))
      set({ selectedSession: details })
    } catch {
      throw new Error('Failed to load conversation')
    }
  },

  deleteSession: async (id) => {
    await consultApi.deleteSession(String(id));
    set(state => ({
      sessions: state.sessions.filter(s => String(s.id) !== String(id)),
      selectedSession: state.selectedSession?.id === id || String(state.selectedSession?.id) === String(id) ? null : state.selectedSession
    }));
  },

  sendMessage: async (content) => {
    const { selectedSession } = get()
    if (!selectedSession) return

    const tempMsg: MessageDto = {
      id: 'temp-' + Date.now(),
      role: 'doctor',
      content,
      timestamp: new Date().toISOString(),
    }

    set(state => ({
      selectedSession: state.selectedSession
        ? { ...state.selectedSession, messages: [...state.selectedSession.messages, tempMsg] }
        : null
    }))

    try {
      const saved = await consultApi.sendMessage(selectedSession.id, content)
      set(state => {
        if (!state.selectedSession) return state
        // Remove temp messages and add saved
        const confirmed = state.selectedSession.messages.filter(m => !String(m.id).startsWith('temp-'))
        
        // Update session list
        const prevSessions = [...state.sessions]
        const idx = prevSessions.findIndex(s => s.id == selectedSession.id)
        if (idx !== -1) {
          const updated = { ...prevSessions[idx], updatedAt: saved.timestamp, lastMessage: content }
          prevSessions.splice(idx, 1)
          prevSessions.unshift(updated)
        }

        return {
          selectedSession: { ...state.selectedSession, messages: [...confirmed, saved] },
          sessions: prevSessions
        }
      })
    } catch (e) {
      set(state => ({
        selectedSession: state.selectedSession
          ? { ...state.selectedSession, messages: state.selectedSession.messages.filter(m => m.id !== tempMsg.id) }
          : null
      }))
      throw e
    }
  },

  handleIncomingMessage: (payload, isChatOpen) => {
    const sessionId = payload?.sessionId ?? payload?.SessionId
    if (!sessionId) return

    const content = payload?.message ?? payload?.Message ?? ''
    const timestamp = payload?.timestamp ?? new Date().toISOString()
    const senderName = payload?.patientName ?? payload?.PatientName ?? 'Patient'
    const patientPhotoUrl = payload?.patientPhotoUrl ?? payload?.PatientPhotoUrl

    set(state => {
      let newSelectedSession = state.selectedSession
      let prevSessions = [...state.sessions]

      if (isChatOpen && String(state.selectedSession?.id) === String(sessionId)) {
        const msg: MessageDto = {
          id: 'rt-' + Date.now() + Math.random(),
          role: 'user',
          content,
          timestamp,
          senderName,
          senderPhotoUrl: patientPhotoUrl,
        }
        
        // Prevent duplicate append if SignalR triggers twice
        const alreadyExists = state.selectedSession?.messages.some(m => 
          m.content === content && 
          Math.abs(new Date(m.timestamp).getTime() - new Date(timestamp).getTime()) < 5000
        )
        
        if (!alreadyExists && state.selectedSession) {
           newSelectedSession = { 
             ...state.selectedSession, 
             messages: [...state.selectedSession.messages, msg] 
           }
        }
      }

      const idx = prevSessions.findIndex(s => s.id == sessionId)
      if (idx !== -1) {
        const updated = { ...prevSessions[idx], updatedAt: timestamp, lastMessage: content }
        prevSessions.splice(idx, 1)
        prevSessions.unshift(updated) // Move to top
      } else {
        // New session not in list - add it
        const newSession: SessionDto = {
          id: String(sessionId),
          title: senderName !== 'Patient' ? senderName : `chat|p:${payload.patientId || 0}|d:0|`,
          createdAt: timestamp,
          updatedAt: timestamp,
          messageCount: 1,
          urgencyLevel: payload.urgencyLevel || null,
          lastMessage: content,
          lastMessageAt: timestamp,
          patientPhotoUrl: patientPhotoUrl,
        };
        prevSessions.unshift(newSession);
      }

      return {
        selectedSession: newSelectedSession,
        sessions: prevSessions
      }
    })
  }
}))
