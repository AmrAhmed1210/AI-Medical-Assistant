import { useState, useEffect } from 'react'
import { secretaryApi } from '@/api/secretaryApi'
import { useLanguage } from '@/lib/language'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Users, UserPlus, Trash2, Mail, Lock, ShieldCheck } from 'lucide-react'
import toast from 'react-hot-toast'
import type { SecretaryDto } from '@/lib/types'

export default function ManageSecretaries() {
  const [secretaries, setSecretaries] = useState<SecretaryDto[]>([])
  const { t, isRTL } = useLanguage()
  const [loading, setLoading] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  
  // Form state
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  const fetchSecretaries = async () => {
    try {
      const data = await secretaryApi.getMySecretaries()
      console.log('[Staff Data]', data)
      setSecretaries(data)
    } catch {
      toast.error(t('errLoadStaff'))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSecretaries()
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!fullName || !email || !password) {
      toast.error(t('fillAllFields'))
      return
    }

    setIsSubmitting(true)
    try {
      await secretaryApi.addSecretary({ fullName, email, password })
      toast.success(t('secAdded'))
      setFullName('')
      setEmail('')
      setPassword('')
      fetchSecretaries()
    } catch (err: any) {
      toast.error(err.response?.data?.message || t('errAddSec'))
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleDelete = async (id: number) => {
    if (!window.confirm(t('confirmRemoveStaff'))) return
    
    try {
      await secretaryApi.deleteSecretary(id)
      toast.success(t('staffRemoved'))
      fetchSecretaries()
    } catch {
      toast.error(t('errRemoveStaff'))
    }
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center gap-3">
        <div className="bg-primary-600 p-3 rounded-xl shadow-lg">
          <Users size={28} className="text-white" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-gray-800">{t('staffMgmt')}</h1>
          <p className="text-sm text-gray-500">{t('manageStaffDesc')}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form to Add */}
        <Card className="lg:col-span-1 p-6">
          <div className="flex items-center gap-2 mb-6">
            <UserPlus size={20} className="text-primary-600" />
            <h2 className="font-bold">{t('addNewSec')}</h2>
          </div>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="text-xs font-semibold text-gray-500 mb-1 block">{t('fullName')}</label>
              <div className="relative">
                <Users size={16} className={`absolute ${isRTL ? "right-3" : "left-3"} top-1/2 -translate-y-1/2 text-gray-400`} />
                <input 
                  type="text" 
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder={t("egName")}
                  autoComplete="off"
                  className={`w-full ${isRTL ? "pr-10 pl-4" : "pl-10 pr-4"} py-2 border rounded-xl focus:ring-2 focus:ring-primary-500/20 outline-none transition-all`}
                />
              </div>
            </div>
            <div>
              <label className="text-xs font-semibold text-gray-500 mb-1 block">{t('emailAddress')}</label>
              <div className="relative">
                <Mail size={16} className={`absolute ${isRTL ? "right-3" : "left-3"} top-1/2 -translate-y-1/2 text-gray-400`} />
                <input 
                  type="email" 
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="mona@example.com"
                  autoComplete="none"
                  className={`w-full ${isRTL ? "pr-10 pl-4" : "pl-10 pr-4"} py-2 border rounded-xl focus:ring-2 focus:ring-primary-500/20 outline-none transition-all`}
                />
              </div>
            </div>
            <div>
              <label className="text-xs font-semibold text-gray-500 mb-1 block">{t('password')}</label>
              <div className="relative">
                <Lock size={16} className={`absolute ${isRTL ? "right-3" : "left-3"} top-1/2 -translate-y-1/2 text-gray-400`} />
                <input 
                  type="password" 
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  autoComplete="new-password"
                  className={`w-full ${isRTL ? "pr-10 pl-4" : "pl-10 pr-4"} py-2 border rounded-xl focus:ring-2 focus:ring-primary-500/20 outline-none transition-all`}
                />
              </div>
            </div>
            <Button 
              type="submit" 
              className="w-full mt-4" 
              loading={isSubmitting}
            >
              Add Staff Member
            </Button>
          </form>
        </Card>

        {/* List of Staff */}
        <Card className="lg:col-span-2 overflow-hidden">
          <div className="p-4 bg-gray-50 border-b font-bold flex justify-between items-center">
            <span>{t('currentStaff')}</span>
            <span className="text-xs bg-primary-100 text-primary-700 px-2 py-1 rounded-lg">
              {secretaries.length} {t('totalWord')}
            </span>
          </div>
          <div className="divide-y">
            {loading ? (
              <div className="p-10 text-center text-gray-400">{t('loading')}</div>
            ) : secretaries.length === 0 ? (
              <div className="p-10 text-center text-gray-400">{t('noStaffYet')}</div>
            ) : secretaries.map((s) => (
              <div key={s.id} className="p-4 flex items-center justify-between hover:bg-gray-50 transition-colors">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center">
                    <ShieldCheck size={20} className="text-emerald-600" />
                  </div>
                  <div>
                    <h3 className="font-bold text-gray-800">{s.fullName || (s as any).FullName}</h3>
                    <p className="text-xs text-gray-500">{s.email || (s as any).Email}</p>
                  </div>
                </div>
                <button 
                  onClick={() => handleDelete(s.id)}
                  className="p-2 text-red-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                >
                  <Trash2 size={18} />
                </button>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  )
}
