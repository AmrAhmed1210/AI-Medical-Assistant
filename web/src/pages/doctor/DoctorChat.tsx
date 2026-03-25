import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, AlertTriangle, Brain, User, MessageSquare } from 'lucide-react'
import { consultApi } from '@/api/consultApi'
import { startConnection } from '@/lib/signalr'
import { useAuthStore } from '@/store/authStore'
import { UrgencyBadge } from '@/components/ui/Badge'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import type { SessionDto, MessageDto } from '@/lib/types'
import { formatTimeAgo, cn } from '@/lib/utils'

export default function DoctorChat() {
  const { token } = useAuthStore()
  const [sessions, setSessions] = useState<SessionDto[]>([])
  const [selectedSession, setSelectedSession] = useState<SessionDto | null>(null)
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
          if (!prev || prev.sessionId !== msg.sessionId) return prev
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
      msgId: 'temp-' + Date.now(),
      sessionId: selectedSession.sessionId,
      role: 'doctor',
      content,
      timestamp: new Date().toISOString(),
    }
    setSelectedSession((prev) => prev ? { ...prev, messages: [...prev.messages, tempMsg] } : prev)
  }, [message, selectedSession])

  const hasEmergency = selectedSession?.finalUrgency === 'EMERGENCY'

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-gray-800">المحادثات</h1>
        <p className="text-sm text-gray-500 mt-0.5">تواصل مع مرضاك في الوقت الحقيقي</p>
      </div>

      <div className="flex gap-4 h-[calc(100vh-200px)]">
        {/* Sessions list */}
        <div className="w-72 bg-white rounded-xl border border-gray-100 shadow-sm overflow-y-auto flex-shrink-0">
          <div className="p-4 border-b border-gray-100">
            <p className="text-sm font-semibold text-gray-700">المحادثات النشطة</p>
          </div>
          {loading ? <PageLoader /> : (
            <div className="divide-y divide-gray-50">
              {sessions.length === 0 ? (
                <div className="p-8 text-center text-gray-400">
                  <MessageSquare size={32} className="mx-auto mb-2 opacity-30" />
                  <p className="text-sm">لا توجد محادثات</p>
                </div>
              ) : sessions.map((s) => (
                <button
                  key={s.sessionId}
                  onClick={() => setSelectedSession(s)}
                  className={cn(
                    'w-full flex items-start gap-3 p-3 hover:bg-gray-50 transition-colors text-right',
                    selectedSession?.sessionId === s.sessionId && 'bg-primary-50'
                  )}
                >
                  <div className="w-9 h-9 rounded-full bg-primary-100 flex items-center justify-center flex-shrink-0">
                    <User size={16} className="text-primary-600" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-800 truncate">{s.patientName}</p>
                    <p className="text-xs text-gray-400">{formatTimeAgo(s.startTime)}</p>
                    {s.finalUrgency && <UrgencyBadge level={s.finalUrgency} />}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Chat area */}
        <div className="flex-1 bg-white rounded-xl border border-gray-100 shadow-sm flex flex-col overflow-hidden">
          {!selectedSession ? (
            <div className="flex-1 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <MessageSquare size={48} className="mx-auto mb-3 opacity-20" />
                <p className="text-sm">اختر محادثة من القائمة</p>
              </div>
            </div>
          ) : (
            <>
              {/* Header */}
              <div className={cn(
                'flex items-center gap-3 px-5 py-4 border-b',
                hasEmergency ? 'bg-red-900 border-red-800' : 'border-gray-100'
              )}>
                {hasEmergency && (
                  <div className="flex items-center gap-2 animate-pulse">
                    <AlertTriangle size={16} className="text-red-200" />
                    <span className="text-sm text-red-200 font-medium">حالة طوارئ!</span>
                  </div>
                )}
                <div className={cn('flex items-center gap-2', hasEmergency && 'mr-auto')}>
                  <div className="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
                    <User size={14} className="text-primary-600" />
                  </div>
                  <div>
                    <p className={cn('text-sm font-medium', hasEmergency ? 'text-white' : 'text-gray-800')}>
                      {selectedSession.patientName}
                    </p>
                    {selectedSession.finalUrgency && <UrgencyBadge level={selectedSession.finalUrgency} />}
                  </div>
                </div>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                <AnimatePresence>
                  {selectedSession.messages.map((msg) => (
                    <motion.div
                      key={msg.msgId}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={cn(
                        'flex gap-2 max-w-[80%]',
                        msg.role === 'doctor' ? 'mr-auto flex-row-reverse' : ''
                      )}
                    >
                      <div className={cn(
                        'w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0',
                        msg.role === 'user' ? 'bg-gray-100' :
                        msg.role === 'assistant' ? 'bg-primary-50' : 'bg-green-100'
                      )}>
                        {msg.role === 'assistant' ? (
                          <Brain size={13} className="text-primary-600" />
                        ) : (
                          <User size={13} className={msg.role === 'doctor' ? 'text-green-600' : 'text-gray-600'} />
                        )}
                      </div>
                      <div className={cn(
                        'px-3.5 py-2.5 rounded-2xl text-sm',
                        msg.role === 'doctor'
                          ? 'bg-primary-600 text-white rounded-tl-sm'
                          : msg.role === 'assistant'
                          ? 'bg-primary-50 text-gray-800 rounded-tr-sm border border-primary-100'
                          : 'bg-gray-100 text-gray-800 rounded-tr-sm'
                      )}>
                        <p>{msg.content}</p>
                        <p className={cn('text-[10px] mt-1', msg.role === 'doctor' ? 'text-white/60' : 'text-gray-400')}>
                          {formatTimeAgo(msg.timestamp)}
                        </p>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div className="p-4 border-t border-gray-100">
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                    placeholder="اكتب رسالتك..."
                    className="flex-1 px-4 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30"
                  />
                  <button
                    onClick={handleSend}
                    disabled={!message.trim()}
                    className="p-2.5 bg-primary-600 text-white rounded-xl hover:bg-primary-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    <Send size={16} />
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
