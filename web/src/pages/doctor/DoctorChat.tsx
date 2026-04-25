import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, AlertTriangle, Activity, User, MessageSquare, Image as ImageIcon, Paperclip, FileText, Trash2 } from 'lucide-react'
import toast from 'react-hot-toast'
import { consultApi } from '@/api/consultApi'
import { useAuthStore } from '@/store/authStore'
import { useNotificationStore } from '@/store/notificationStore'
import { useMessagesStore } from '@/store/messagesStore'
import { UrgencyBadge } from '@/components/ui/Badge'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { cn } from '@/lib/utils'

function isRawSessionTitle(title?: string | null): boolean {
  return !!title && /chat\|p:\d+\|d:\d+\|/i.test(title)
}

function toDisplaySessionTitle(title?: string | null, sessionId?: string | number): string {
  if (!title) return `Session ${String(sessionId ?? '').slice(0, 8)}`
  if (!isRawSessionTitle(title)) return title
  const patientMatch = title.match(/\|p:(\d+)\|/i)
  if (patientMatch) return `Patient #${patientMatch[1]}`
  return `Session ${String(sessionId ?? '').slice(0, 8)}`
}

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

export default function DoctorChat() {
  const { unreadCounts, clearSessionMessages, incrementSessionMessage, latestMessagePayload } = useNotificationStore()
  const { sessions, selectedSession, isLoadingSessions, fetchSessions, openSession, sendMessage, handleIncomingMessage } = useMessagesStore()
  
  const [message, setMessage] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const selectedSessionIdRef = useRef<number | string | null>(null)

  const handleOpenSession = useCallback(async (sessionId: string | number) => {
    try {
      await openSession(sessionId)
      clearSessionMessages(Number(sessionId))
    } catch {
      toast.error('Failed to load conversation')
    }
  }, [openSession, clearSessionMessages])

  const handleSend = useCallback(async () => {
    if (!message.trim() || !selectedSession) return
    const content = message.trim()
    setMessage('')
    try {
      await sendMessage(content)
    } catch {
      toast.error('Failed to send message')
    }
  }, [message, selectedSession, sendMessage])

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !selectedSession) return
    
    const isImage = file.type.startsWith('image/')
    const type = isImage ? 'image' : 'file'
    
    const loadingToast = toast.loading(`Uploading ${type}...`)
    try {
      const { url, fileName } = await consultApi.uploadFile(file)
      await consultApi.sendMessage(selectedSession.id, isImage ? '[Image]' : `[File: ${fileName}]`, type, url, fileName)
      // Refresh session to show new message
      await openSession(selectedSession.id)
      toast.success(`${type} sent`, { id: loadingToast })
    } catch {
      toast.error(`Failed to upload ${type}`, { id: loadingToast })
    }
  }

  useEffect(() => {
    selectedSessionIdRef.current = selectedSession?.id || null
  }, [selectedSession?.id])

  useEffect(() => {
    fetchSessions();
    
    const params = new URLSearchParams(window.location.search);
    const sid = params.get('sessionId');
    if (sid) {
      const numSid = Number(sid);
      if (!isNaN(numSid)) {
        handleOpenSession(numSid);
      }
      // Clear param to avoid re-opening on manual refresh
      window.history.replaceState({}, '', window.location.pathname);
    }
  }, [fetchSessions, handleOpenSession])

  useEffect(() => {
    if (!latestMessagePayload) return
    const payload = latestMessagePayload
    const sessionId = payload?.sessionId ?? payload?.SessionId
    if (!sessionId) return

    const isChatOpen = String(selectedSessionIdRef.current) === String(sessionId)
    
    try {
      handleIncomingMessage(payload, isChatOpen)
    } catch (e) {
      console.error('Error handling incoming message:', e)
    }

    if (isChatOpen) {
      clearSessionMessages(sessionId)
    } else {
      incrementSessionMessage(sessionId)
    }
  }, [latestMessagePayload, clearSessionMessages, incrementSessionMessage, handleIncomingMessage])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [selectedSession?.messages])

  const hasHighUrgency = selectedSession?.urgencyLevel === 'HIGH'

  const handleDeleteSession = async () => {
    if (!selectedSession) return;
    if (!window.confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) return;
    
    try {
      await useMessagesStore.getState().deleteSession(selectedSession.id);
      toast.success('Conversation deleted successfully');
    } catch (e: any) {
      toast.error('Failed to delete conversation');
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-br from-primary-600 to-primary-400 rounded-xl blur-lg opacity-30" />
          <div className="relative bg-gradient-to-br from-primary-600 to-primary-500 rounded-xl p-3 shadow-lg">
            <MessageSquare size={28} className="text-white" />
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
          {isLoadingSessions ? <PageLoader /> : (
            <div className="divide-y divide-gray-50">
              {sessions.length === 0 ? (
                <div className="p-8 text-center text-gray-400">
                  <MessageSquare size={32} className="mx-auto mb-2 opacity-30" />
                  <p className="text-sm">No conversations yet</p>
                </div>
              ) : sessions.map((s) => (
                <motion.button
                  key={s.id}
                  onClick={() => handleOpenSession(s.id)}
                  whileHover={{ x: 4 }}
                  className={cn(
                    'w-full flex items-start gap-3 p-3 transition-colors',
                    selectedSession?.id === s.id && 'bg-gradient-to-r from-primary-50 to-transparent border-l-2 border-primary-600'
                  )}
                >
                  <div className={cn(
                    'w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 overflow-hidden shadow-sm',
                    selectedSession?.id === s.id ? 'bg-gradient-to-br from-primary-600 to-primary-500' : 'bg-primary-100'
                  )}>
                    {s.patientPhotoUrl ? (
                      <img src={s.patientPhotoUrl} alt="" className="w-full h-full object-cover" />
                    ) : (
                      <User size={16} className={selectedSession?.id === s.id ? 'text-white' : 'text-primary-600'} />
                    )}
                  </div>
                  <div className="flex-1 min-w-0 text-left relative">
                    <p className="text-sm font-medium text-gray-800 truncate pr-6">
                      {toDisplaySessionTitle(s.title, s.id)}
                    </p>
                    <p className="text-xs text-gray-400">{formatTimeAgo(s.lastMessageAt || s.updatedAt || s.createdAt)}</p>
                    {s.lastMessage && (
                      <p className="text-xs text-gray-500 truncate mt-1 pr-4">{s.lastMessage}</p>
                    )}
                    {s.urgencyLevel && <UrgencyBadge level={s.urgencyLevel} />}
                    
                    {unreadCounts[Number(s.id)] > 0 && (
                      <div className="absolute top-0 right-0 bg-green-500 text-white min-w-[18px] h-[18px] rounded-full flex items-center justify-center text-[10px] font-bold shadow-sm">
                        {unreadCounts[Number(s.id)] > 9 ? '9+' : unreadCounts[Number(s.id)]}
                      </div>
                    )}
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
                    'w-8 h-8 rounded-full flex items-center justify-center overflow-hidden flex-shrink-0',
                    hasHighUrgency ? 'bg-red-100' : 'bg-primary-100'
                  )}>
                    {selectedSession.patientPhotoUrl ? (
                      <img src={selectedSession.patientPhotoUrl} alt="" className="w-full h-full object-cover" />
                    ) : (
                      <User size={14} className={hasHighUrgency ? 'text-red-600' : 'text-primary-600'} />
                    )}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-800">
                      {toDisplaySessionTitle(selectedSession.title, selectedSession.id)}
                    </p>
                    {selectedSession.urgencyLevel && <UrgencyBadge level={selectedSession.urgencyLevel} />}
                  </div>
                  <button
                    onClick={handleDeleteSession}
                    className="ml-2 p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-full transition-colors"
                    title="Delete Conversation"
                  >
                    <Trash2 size={16} />
                  </button>
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
                        msg.role === 'doctor' ? 'ml-auto flex-row-reverse' : ''
                      )}
                    >
                      <div className={cn(
                        'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 shadow-sm overflow-hidden',
                        msg.role === 'doctor' ? 'bg-gradient-to-br from-blue-100 to-blue-50' :
                          msg.role === 'assistant' ? 'bg-gradient-to-br from-purple-100 to-purple-50' : 'bg-gradient-to-br from-green-100 to-green-50'
                      )}>
                        {msg.role === 'assistant' ? (
                          <Activity size={14} className="text-purple-600" />
                        ) : msg.senderPhotoUrl ? (
                          <img src={msg.senderPhotoUrl} alt="" className="w-full h-full object-cover" />
                        ) : (
                          <User size={14} className={msg.role === 'doctor' ? 'text-blue-600' : 'text-green-600'} />
                        )}
                      </div>
                      <div className={cn(
                        'px-4 py-3 rounded-2xl text-sm shadow-md hover:shadow-lg transition-shadow',
                        msg.role === 'doctor'
                          ? 'bg-gradient-to-br from-primary-600 to-primary-500 text-white rounded-br-none'
                          : msg.role === 'assistant'
                            ? 'bg-gradient-to-br from-purple-100 to-purple-50 text-gray-800 rounded-bl-none border border-purple-200'
                            : 'bg-gradient-to-br from-gray-100 to-gray-50 text-gray-800 rounded-tl-none'
                      )}>
                        {msg.messageType === 'image' && msg.attachmentUrl ? (
                          <img 
                            src={msg.attachmentUrl} 
                            alt="Attachment" 
                            className="max-w-xs rounded-lg mb-2 cursor-pointer hover:opacity-90"
                            onClick={() => window.open(msg.attachmentUrl!, '_blank')}
                          />
                        ) : msg.messageType === 'file' ? (
                          <div 
                            className="flex items-center gap-2 mb-2 p-2 bg-black/5 rounded-lg cursor-pointer"
                            onClick={() => window.open(msg.attachmentUrl!, '_blank')}
                          >
                            <FileText size={20} className={msg.role === 'doctor' ? 'text-white' : 'text-primary-600'} />
                            <div className="flex-1 min-w-0">
                              <p className="text-xs font-bold truncate">{msg.fileName || 'Document'}</p>
                              <p className="text-[10px] opacity-70">Attachment</p>
                            </div>
                          </div>
                        ) : null}
                        <p className="leading-relaxed">{msg.content}</p>
                        <p className={cn('text-[11px] mt-1', msg.role === 'doctor' ? 'text-white/80' : 'text-gray-500')}>
                          {msg.senderName || (msg.role === 'doctor' ? 'You' : toDisplaySessionTitle(selectedSession.title, selectedSession.id))}
                        </p>
                        <p className={cn('text-[11px] mt-2 font-medium', msg.role === 'doctor' ? 'text-white/70' : 'text-gray-500')}>
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
                  <div className="flex gap-1">
                    <label className="p-3 text-gray-500 hover:text-primary-600 hover:bg-primary-50 rounded-xl cursor-pointer transition-all">
                      <ImageIcon size={18} />
                      <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
                    </label>
                    <label className="p-3 text-gray-500 hover:text-primary-600 hover:bg-primary-50 rounded-xl cursor-pointer transition-all">
                      <Paperclip size={18} />
                      <input type="file" className="hidden" onChange={handleFileUpload} />
                    </label>
                  </div>
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
