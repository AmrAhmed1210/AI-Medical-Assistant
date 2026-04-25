import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, User, MessageSquare, LifeBuoy, FileText, ImageIcon } from 'lucide-react'
import toast from 'react-hot-toast'
import { consultApi } from '@/api/consultApi'
import { useNotificationStore } from '@/store/notificationStore'
import { UrgencyBadge } from '@/components/ui/Badge'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { cn } from '@/lib/utils'
import type { SessionDto, SessionDetailDto, MessageDto } from '@/lib/types'

const formatTimeAgo = (timestamp?: string | null) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  if (isNaN(date.getTime())) return ''
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`
  return date.toLocaleDateString()
}

export default function SupportPage() {
  const { unreadCounts, clearSessionMessages, incrementSessionMessage, latestMessagePayload } = useNotificationStore()
  
  const [sessions, setSessions] = useState<SessionDto[]>([])
  const [selectedSession, setSelectedSession] = useState<SessionDetailDto | null>(null)
  const [isLoadingSessions, setIsLoadingSessions] = useState(false)
  const [message, setMessage] = useState('')
  const [sending, setSending] = useState(false)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const selectedSessionIdRef = useRef<number | string | null>(null)

  const fetchSessions = useCallback(async () => {
    setIsLoadingSessions(true)
    try {
      const data = await consultApi.getSupportSessions()
      setSessions(data.sort((a, b) => {
        const timeA = new Date(a.lastMessageAt || a.updatedAt || a.createdAt).getTime()
        const timeB = new Date(b.lastMessageAt || b.updatedAt || b.createdAt).getTime()
        return timeB - timeA
      }))
    } catch {
      toast.error('Failed to load support sessions')
    } finally {
      setIsLoadingSessions(false)
    }
  }, [])

  const handleOpenSession = useCallback(async (sessionId: string | number) => {
    try {
      const details = await consultApi.getSession(String(sessionId))
      setSelectedSession(details)
      clearSessionMessages(Number(sessionId))
    } catch {
      toast.error('Failed to load conversation')
    }
  }, [clearSessionMessages])

  const handleSend = useCallback(async () => {
    if (!message.trim() || !selectedSession || sending) return
    const content = message.trim()
    setMessage('')
    setSending(true)
    try {
      const saved = await consultApi.sendMessage(selectedSession.id, content)
      setSelectedSession(prev => prev ? { ...prev, messages: [...prev.messages, saved] } : null)
      
      // Update session list preview
      setSessions(prev => {
        const updated = [...prev]
        const idx = updated.findIndex(s => s.id == selectedSession.id)
        if (idx !== -1) {
          updated[idx] = { ...updated[idx], lastMessage: content, lastMessageAt: saved.timestamp }
        }
        return updated
      })
    } catch {
      toast.error('Failed to send message')
    } finally {
      setSending(false)
    }
  }, [message, selectedSession, sending])

  useEffect(() => {
    selectedSessionIdRef.current = selectedSession?.id || null
  }, [selectedSession?.id])

  useEffect(() => {
    fetchSessions()
  }, [fetchSessions])

  useEffect(() => {
    if (!latestMessagePayload) return
    const payload = latestMessagePayload
    const sessionId = payload?.sessionId ?? payload?.SessionId
    if (!sessionId) return

    const isChatOpen = String(selectedSessionIdRef.current) === String(sessionId)
    
    if (isChatOpen) {
      const msg: MessageDto = {
        id: 'rt-' + Date.now() + Math.random(),
        role: payload.role || 'user',
        content: payload.message || payload.Content || '',
        timestamp: payload.timestamp || new Date().toISOString(),
        senderName: payload.senderName || payload.SenderName || 'User',
        senderPhotoUrl: payload.senderPhotoUrl || payload.SenderPhotoUrl,
        messageType: payload.messageType || 'text',
        attachmentUrl: payload.attachmentUrl,
        fileName: payload.fileName
      }
      
      setSelectedSession(prev => {
        if (!prev || String(prev.id) !== String(sessionId)) return prev
        const exists = prev.messages.some(m => m.content === msg.content && Math.abs(new Date(m.timestamp).getTime() - new Date(msg.timestamp).getTime()) < 3000)
        if (exists) return prev
        return { ...prev, messages: [...prev.messages, msg] }
      })
      clearSessionMessages(sessionId)
    } else {
      // If it's a support chat message, increment unread for that session
      // We'd need to verify if the session is of type SupportChat in the payload or fetch list
      incrementSessionMessage(sessionId)
      fetchSessions() // Refresh list to show new message
    }
  }, [latestMessagePayload, clearSessionMessages, incrementSessionMessage, fetchSessions])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [selectedSession?.messages])

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-br from-indigo-600 to-indigo-400 rounded-xl blur-lg opacity-30" />
          <div className="relative bg-gradient-to-br from-indigo-600 to-indigo-500 rounded-xl p-3 shadow-lg">
            <LifeBuoy size={28} className="text-white" />
          </div>
        </div>
        <div>
          <h1 className="text-xl font-bold text-gray-800">Support Center</h1>
          <p className="text-sm text-gray-500 mt-0.5">Manage user complaints and technical requests</p>
        </div>
      </div>

      <div className="flex gap-4 h-[calc(100vh-200px)]">
        {/* Sessions list */}
        <motion.div className="w-80 bg-gradient-to-b from-white to-gray-50 rounded-xl border border-gray-100 shadow-sm overflow-y-auto flex-shrink-0"
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
        >
          <div className="p-4 border-b border-gray-100 bg-indigo-50/30">
            <p className="text-sm font-semibold text-gray-700">Support Conversations</p>
            <p className="text-xs text-gray-500 mt-1">{sessions.length} ticket(s) found</p>
          </div>
          {isLoadingSessions ? <PageLoader /> : (
            <div className="divide-y divide-gray-50">
              {sessions.length === 0 ? (
                <div className="p-12 text-center text-gray-400">
                  <LifeBuoy size={40} className="mx-auto mb-3 opacity-20" />
                  <p className="text-sm">No support tickets</p>
                </div>
              ) : sessions.map((s) => (
                <motion.button
                  key={s.id}
                  onClick={() => handleOpenSession(s.id)}
                  whileHover={{ x: 4 }}
                  className={cn(
                    'w-full flex items-start gap-3 p-4 transition-colors',
                    selectedSession?.id === s.id && 'bg-indigo-50/50 border-l-4 border-indigo-600'
                  )}
                >
                  <div className={cn(
                    'w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 overflow-hidden shadow-sm',
                    selectedSession?.id === s.id ? 'bg-indigo-600 text-white' : 'bg-indigo-100 text-indigo-600'
                  )}>
                    {s.patientPhotoUrl ? (
                      <img src={s.patientPhotoUrl} alt="" className="w-full h-full object-cover" />
                    ) : (
                      <User size={18} />
                    )}
                  </div>
                  <div className="flex-1 min-w-0 text-left relative">
                    <div className="flex justify-between items-start">
                      <p className="text-sm font-bold text-gray-800 truncate pr-2">
                        {s.title || `User #${s.userId}`}
                      </p>
                      <p className="text-[10px] text-gray-400 whitespace-nowrap">{formatTimeAgo(s.lastMessageAt || s.updatedAt || s.createdAt)}</p>
                    </div>
                    {s.lastMessage && (
                      <p className="text-xs text-gray-500 truncate mt-1">{s.lastMessage}</p>
                    )}
                    
                    {unreadCounts[Number(s.id)] > 0 && (
                      <div className="absolute -right-1 bottom-0 bg-red-500 text-white min-w-[18px] h-[18px] rounded-full flex items-center justify-center text-[10px] font-bold shadow-sm">
                        {unreadCounts[Number(s.id)]}
                      </div>
                    )}
                  </div>
                </motion.button>
              ))}
            </div>
          )}
        </motion.div>

        {/* Chat area */}
        <motion.div className="flex-1 bg-white rounded-xl border border-gray-100 shadow-sm flex flex-col overflow-hidden"
          initial={{ x: 20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
        >
          {!selectedSession ? (
            <div className="flex-1 flex items-center justify-center text-gray-400 bg-gray-50/30">
              <div className="text-center">
                <MessageSquare size={64} className="mx-auto mb-4 opacity-10" />
                <p className="text-lg font-medium text-gray-400">Select a support ticket to respond</p>
                <p className="text-sm text-gray-400 mt-2">View complaints from users and doctors</p>
              </div>
            </div>
          ) : (
            <>
              {/* Header */}
              <div className="flex items-center justify-between px-6 py-4 border-b bg-white">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-600">
                    <User size={20} />
                  </div>
                  <div>
                    <h3 className="font-bold text-gray-900">{selectedSession.title}</h3>
                    <p className="text-xs text-gray-500">User ID: {selectedSession.userId}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="px-3 py-1 bg-green-100 text-green-700 text-[10px] font-bold rounded-full uppercase tracking-wider">Active Ticket</span>
                </div>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50/50">
                <AnimatePresence mode="popLayout">
                  {selectedSession.messages.map((msg, idx) => {
                    const isAdmin = msg.role === 'admin'
                    return (
                      <motion.div
                        key={msg.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={cn('flex gap-3 max-w-[80%]', isAdmin ? 'ml-auto flex-row-reverse' : '')}
                      >
                        <div className={cn(
                          'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 shadow-sm overflow-hidden',
                          isAdmin ? 'bg-indigo-600 text-white' : 'bg-white text-gray-600'
                        )}>
                          {isAdmin ? <LifeBuoy size={14} /> : (msg.senderPhotoUrl ? <img src={msg.senderPhotoUrl} className="w-full h-full object-cover" /> : <User size={14} />)}
                        </div>
                        <div className={cn(
                          'px-4 py-3 rounded-2xl text-sm shadow-sm',
                          isAdmin ? 'bg-indigo-600 text-white rounded-tr-none' : 'bg-white text-gray-800 rounded-tl-none border border-gray-100'
                        )}>
                          {msg.messageType === 'image' && msg.attachmentUrl ? (
                            <img src={msg.attachmentUrl} alt="attachment" className="max-w-xs rounded-lg mb-2 cursor-pointer hover:opacity-90" onClick={() => window.open(msg.attachmentUrl!, '_blank')} />
                          ) : msg.messageType === 'file' ? (
                            <div className="flex items-center gap-2 mb-2 p-2 bg-black/5 rounded-lg cursor-pointer" onClick={() => window.open(msg.attachmentUrl!, '_blank')}>
                              <FileText size={20} className={isAdmin ? 'text-white' : 'text-indigo-600'} />
                              <div className="flex-1 min-w-0">
                                <p className="text-xs font-bold truncate">{msg.fileName || 'Document'}</p>
                                <p className="text-[10px] opacity-70">Attachment</p>
                              </div>
                            </div>
                          ) : null}
                          <p className="leading-relaxed">{msg.content}</p>
                          <div className={cn('flex items-center justify-between gap-4 mt-2 border-t pt-1', isAdmin ? 'border-white/10' : 'border-gray-50')}>
                            <span className="text-[10px] font-bold opacity-60 uppercase">{msg.senderName || msg.role}</span>
                            <span className="text-[10px] opacity-50">{formatTimeAgo(msg.timestamp)}</span>
                          </div>
                        </div>
                      </motion.div>
                    )
                  })}
                </AnimatePresence>
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div className="p-6 bg-white border-t">
                <div className="flex items-center gap-3">
                  <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                    placeholder="Type your response..."
                    className="flex-1 px-4 py-3 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all"
                  />
                  <motion.button
                    onClick={handleSend}
                    disabled={!message.trim() || sending}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="p-3 bg-indigo-600 text-white rounded-xl shadow-lg hover:bg-indigo-700 disabled:opacity-50 transition-all flex items-center justify-center min-w-[48px]"
                  >
                    {sending ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Send size={18} />}
                  </motion.button>
                </div>
              </div>
            </>
          )}
        </motion.div>
      </div>
    </div>
  )
}
