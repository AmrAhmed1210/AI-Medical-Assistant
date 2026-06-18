import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Bot, Send, Loader2, User, Sparkles, CheckCircle2 } from 'lucide-react'
import { Modal } from '@/components/ui/Modal'
import { Button } from '@/components/ui/Button'
import toast from 'react-hot-toast'

interface Message {
  role: 'user' | 'model'
  content: string
}

interface ReasonForVisitModalProps {
  open: boolean
  onClose: () => void
  onComplete: (reason: string) => void
}

export function ReasonForVisitModal({ open, onClose, onComplete }: ReasonForVisitModalProps) {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'model', content: 'أهلاً بك! لمساعدتك بشكل أفضل، ما هو سبب الزيارة أو الشكوى الرئيسية التي تعاني منها؟' }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isSummarizing, setIsSummarizing] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const aiServerUrl = import.meta.env.VITE_AI_SERVER_URL || 'http://localhost:8000'

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Reset chat when modal closes
  useEffect(() => {
    if (!open) {
      setTimeout(() => {
        setMessages([{ role: 'model', content: 'أهلاً بك! لمساعدتك بشكل أفضل، ما هو سبب الزيارة أو الشكوى الرئيسية التي تعاني منها؟' }])
        setInput('')
      }, 300)
    }
  }, [open])

  const handleSend = async () => {
    if (!input.trim()) return
    
    const userMsg = input.trim()
    setInput('')
    const newHistory: Message[] = [...messages, { role: 'user', content: userMsg }]
    setMessages(newHistory)
    setIsLoading(true)

    try {
      const res = await fetch(`${aiServerUrl}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-internal-token': 'LuxuryMedicalAiSecretKey2026'
        },
        body: JSON.stringify({
          question: userMsg,
          history: messages.map(m => ({ role: m.role, content: m.content }))
        })
      })
      
      if (!res.ok) throw new Error('AI Error')
      const data = await res.json()
      
      setMessages([...newHistory, { role: 'model', content: data.reply }])
    } catch (e) {
      toast.error('Could not connect to AI. Please try again.')
      setMessages([...newHistory, { role: 'model', content: 'عذراً، حدث خطأ في الاتصال. يمكنك تلخيص شكواك الآن للإنهاء.' }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleSummarizeAndComplete = async () => {
    setIsSummarizing(true)
    try {
      const res = await fetch(`${aiServerUrl}/summarize-booking-reason`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-internal-token': 'LuxuryMedicalAiSecretKey2026'
        },
        body: JSON.stringify({
          question: '',
          history: messages.map(m => ({ role: m.role, content: m.content }))
        })
      })
      
      if (!res.ok) throw new Error('Summary Error')
      const data = await res.json()
      onComplete(data.summary)
    } catch (e) {
      console.error('Summary fallback used', e)
      // Fallback if AI fails: just use the last user message
      const lastUserMsg = [...messages].reverse().find(m => m.role === 'user')
      onComplete(lastUserMsg ? lastUserMsg.content : 'General Consultation')
    } finally {
      setIsSummarizing(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="سبب الزيارة (الذكاء الاصطناعي)"
      size="md"
    >
      <div className="flex flex-col h-[60vh] max-h-[500px] font-tajawal bg-slate-50 dark:bg-slate-950/50 rounded-xl overflow-hidden border border-slate-100 dark:border-slate-800">
        
        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <AnimatePresence initial={false}>
            {messages.map((msg, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex gap-3 max-w-[85%] ${msg.role === 'user' ? 'mr-auto flex-row-reverse' : 'ml-auto'}`}
              >
                <div className={`w-8 h-8 rounded-full flex shrink-0 items-center justify-center ${msg.role === 'user' ? 'bg-emerald-100 text-emerald-600' : 'bg-primary-100 text-primary-600'}`}>
                  {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                </div>
                <div className={`px-4 py-2.5 rounded-2xl text-sm ${msg.role === 'user' ? 'bg-emerald-600 text-white rounded-tr-sm' : 'bg-white dark:bg-slate-900 border border-slate-100 dark:border-slate-800 text-slate-800 dark:text-slate-200 rounded-tl-sm shadow-sm'}`}>
                  {msg.content}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {isLoading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3 max-w-[85%]">
               <div className="w-8 h-8 rounded-full bg-primary-100 text-primary-600 flex shrink-0 items-center justify-center">
                  <Bot size={16} />
                </div>
                <div className="px-4 py-2.5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-100 dark:border-slate-800 rounded-tl-sm flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-primary-400 animate-bounce" />
                  <span className="w-1.5 h-1.5 rounded-full bg-primary-400 animate-bounce delay-75" />
                  <span className="w-1.5 h-1.5 rounded-full bg-primary-400 animate-bounce delay-150" />
                </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-3 bg-white dark:bg-slate-900 border-t border-slate-100 dark:border-slate-800">
          <div className="flex items-center gap-2 mb-3">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSend()}
              placeholder="اكتب الأعراض أو سبب الزيارة..."
              className="flex-1 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/30"
              disabled={isLoading || isSummarizing}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading || isSummarizing}
              className="w-11 h-11 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 text-white rounded-xl flex items-center justify-center shrink-0 transition-colors"
            >
              <Send size={18} className="rtl:-scale-x-100" />
            </button>
          </div>
          
          <Button
            onClick={handleSummarizeAndComplete}
            disabled={messages.length <= 1 || isLoading || isSummarizing}
            className="w-full bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-white border-0 shadow-lg shadow-emerald-500/20"
          >
            {isSummarizing ? (
              <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> جاري التلخيص والحجز...</>
            ) : (
              <><CheckCircle2 className="w-4 h-4 mr-2" /> إتمام الحجز والمتابعة</>
            )}
          </Button>
        </div>

      </div>
    </Modal>
  )
}
