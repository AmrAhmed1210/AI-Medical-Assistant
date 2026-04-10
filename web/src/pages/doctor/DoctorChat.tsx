import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, AlertTriangle, Brain, User, MessageSquare, MessageCircle } from 'lucide-react'
import { consultApi } from '@/api/consultApi'
import { startConnection } from '@/lib/signalr'
import { useAuthStore } from '@/store/authStore'
import { UrgencyBadge } from '@/components/ui/Badge'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import type { SessionDto, SessionDetailDto, MessageDto } from '@/lib/types'
import { formatTimeAgo, cn } from '@/lib/utils'

export default function DoctorChat() {
  const { token } = useAuthStore()
  const [sessions, setSessions] = useState<SessionDto[]>([])
  const [selectedSession, setSelectedSession] = useState<SessionDetailDto | null>(null)
  const [loading, setLoading] = useState(true)
  const [message, setMessage] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    consultApi.getSessions()
      .then(setSessions)
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!token) return
    let cleanup: (() => void) | undefined

    startConnection(token).then((conn) => {
      conn.on('ReceiveMessage', (msg: MessageDto) => {
        setSelectedSession((prev) => {
          if (!prev || prev.id !== msg.id) return prev
          return { ...prev, messages: [...prev.messages, msg] }
        })
      })
      cleanup = () => conn.off('ReceiveMessage')
    }).catch(console.error)

    return () => cleanup?.()
  }, [token])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [selectedSession?.messages])

  const handleSend = useCallback(async () => {
    if (!message.trim() || !selectedSession) return
    const content = message.trim()
    setMessage('')
    // Optimistic update
    const tempMsg: MessageDto = {
      id: 'temp-' + Date.now(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    }
    setSelectedSession((prev) => prev ? { ...prev, messages: [...prev.messages, tempMsg] } : prev)
  }, [message, selectedSession])

  const hasHighUrgency = selectedSession?.urgencyLevel === 'HIGH'

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-br from-primary-600 to-primary-400 rounded-xl blur-lg opacity-30" />
          <div className="relative bg-gradient-to-br from-primary-600 to-primary-500 rounded-xl p-3 shadow-lg">
            <MessageCircle size={28} className="text-white" />
          </div>
        </div>
        <div>
          <h1 className="text-xl font-bold text-gray-800">Conversations</h1>
          <p className="text-sm text-gray-500 mt-0.5">Connect with your patients in real-time</p>
        </div>
      </div>

      <div className="flex gap-4 h-[calc(100vh-200px)]">
        {/* Sessions list */}
        <motion.div className="w-72 bg-gradient-to-b from-white to-gray-50 rounded-xl border border-gray-100 shadow-sm overflow-y-auto flex-shrink-0"
          initial={{ x: -20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <div className="p-4 border-b border-gray-100 bg-gradient-to-r from-primary-50 to-transparent">
            <p className="text-sm font-semibold text-gray-700">Active Conversations</p>
            <p className="text-xs text-gray-500 mt-1">{sessions.length} session(s)</p>
          </div>
          {loading ? <PageLoader /> : (
            <div className="divide-y divide-gray-50">
              {sessions.length === 0 ? (
                <div className="p-8 text-center text-gray-400">
                  <MessageSquare size={32} className="mx-auto mb-2 opacity-30" />
                  <p className="text-sm">No conversations yet</p>
                </div>
              ) : sessions.map((s) => (
                <motion.button
                  key={s.id}
                  onClick={() => setSelectedSession(s as SessionDetailDto)}
                  whileHover={{ x: 4 }}
                  className={cn(
                    'w-full flex items-start gap-3 p-3 transition-colors',
                    selectedSession?.id === s.id && 'bg-gradient-to-r from-primary-50 to-transparent border-l-2 border-primary-600'
                  )}
                >
                  <div className={cn(
                    'w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0',
                    selectedSession?.id === s.id ? 'bg-gradient-to-br from-primary-600 to-primary-500' : 'bg-primary-100'
                  )}>
                    <User size={16} className={selectedSession?.id === s.id ? 'text-white' : 'text-primary-600'} />
                  </div>
                  <div className="flex-1 min-w-0 text-left">
                    <p className="text-sm font-medium text-gray-800 truncate">{s.title || 'Session ' + s.id.slice(0, 8)}</p>
                    <p className="text-xs text-gray-400">{formatTimeAgo(s.createdAt)}</p>
                    {s.messageCount > 0 && <p className="text-xs text-primary-600 font-medium mt-1">{s.messageCount} messages</p>}
                    {s.urgencyLevel && <UrgencyBadge level={s.urgencyLevel} />}
                  </div>
                </motion.button>
              ))}
            </div>
          )}
        </motion.div>

        {/* Chat area */}
        <motion.div className="flex-1 bg-gradient-to-br from-white via-blue-50/30 to-white rounded-xl border border-gray-100 shadow-sm flex flex-col overflow-hidden"
          initial={{ x: 20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          {!selectedSession ? (
            <div className="flex-1 flex items-center justify-center text-gray-400">
              <motion.div className="text-center"
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 3, repeat: Infinity }}
              >
                <MessageSquare size={48} className="mx-auto mb-3 opacity-20" />
                <p className="text-sm">Select a conversation to start</p>
              </motion.div>
            </div>
          ) : (
            <>
              {/* Header */}
              <div className={cn(
                'flex items-center gap-3 px-5 py-4 border-b backdrop-blur-sm',
                hasHighUrgency ? 'bg-gradient-to-r from-red-50 to-transparent border-red-200' : 'border-gray-100 bg-gradient-to-r from-primary-50/50 to-transparent'
              )}>
                {hasHighUrgency && (
                  <motion.div className="flex items-center gap-2" animate={{ scale: [1, 1.1, 1] }} transition={{ duration: 1.5, repeat: Infinity }}>
                    <AlertTriangle size={16} className="text-red-600" />
                    <span className="text-sm text-red-600 font-medium">High Urgency</span>
                  </motion.div>
                )}
                <div className={cn('flex items-center gap-2', hasHighUrgency && 'ml-auto')}>
                  <div className={cn(
                    'w-8 h-8 rounded-full flex items-center justify-center',
                    hasHighUrgency ? 'bg-red-100' : 'bg-primary-100'
                  )}>
                    <User size={14} className={hasHighUrgency ? 'text-red-600' : 'text-primary-600'} />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-800">
                      {selectedSession.title || 'Session'}
                    </p>
                    {selectedSession.urgencyLevel && <UrgencyBadge level={selectedSession.urgencyLevel} />}
                  </div>
                </div>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-5 space-y-3">
                <AnimatePresence mode="popLayout">
                  {selectedSession.messages.map((msg, idx) => (
                    <motion.div
                      key={msg.id}
                      initial={{ opacity: 0, y: 12, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 8 }}
                      transition={{ duration: 0.3, delay: idx * 0.05 }}
                      className={cn(
                        'flex gap-3 max-w-[85%]',
                        msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''
                      )}
                    >
                      <div className={cn(
                        'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 shadow-sm',
                        msg.role === 'user' ? 'bg-gradient-to-br from-blue-100 to-blue-50' :
                          msg.role === 'assistant' ? 'bg-gradient-to-br from-purple-100 to-purple-50' : 'bg-gradient-to-br from-green-100 to-green-50'
                      )}>
                        {msg.role === 'assistant' ? (
                          <Brain size={14} className="text-purple-600" />
                        ) : (
                          <User size={14} className={msg.role === 'user' ? 'text-blue-600' : 'text-green-600'} />
                        )}
                      </div>
                      <div className={cn(
                        'px-4 py-3 rounded-2xl text-sm shadow-md hover:shadow-lg transition-shadow',
                        msg.role === 'user'
                          ? 'bg-gradient-to-br from-primary-600 to-primary-500 text-white rounded-br-none'
                          : msg.role === 'assistant'
                            ? 'bg-gradient-to-br from-purple-100 to-purple-50 text-gray-800 rounded-bl-none border border-purple-200'
                            : 'bg-gradient-to-br from-gray-100 to-gray-50 text-gray-800 rounded-tl-none'
                      )}>
                        <p className="leading-relaxed">{msg.content}</p>
                        <p className={cn('text-[11px] mt-2 font-medium', msg.role === 'user' ? 'text-white/70' : 'text-gray-500')}>
                          {formatTimeAgo(msg.timestamp)}
                        </p>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div className="p-5 border-t border-gray-100 bg-gradient-to-t from-white to-transparent">
                <motion.div className="flex items-center gap-2"
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                >
                  <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                    placeholder="Type your message..."
                    className="flex-1 px-4 py-3 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30 focus:border-transparent transition-all bg-white"
                  />
                  <motion.button
                    onClick={handleSend}
                    disabled={!message.trim()}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="p-3 bg-gradient-to-br from-primary-600 to-primary-500 text-white rounded-xl hover:shadow-lg disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-md"
                  >
                    <Send size={16} />
                  </motion.button>
                </motion.div>
              </div>
            </>
          )}
        </motion.div>
      </div>
    </div>
  )
}
