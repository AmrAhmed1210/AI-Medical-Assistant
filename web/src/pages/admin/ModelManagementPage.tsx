import { useLanguage } from '@/lib/language'
import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Cpu, RefreshCw, Zap, Shield, Database, Activity, Loader2 } from 'lucide-react'
import { adminApi } from '@/api/adminApi'
import type { ModelVersionDto } from '@/lib/types'
import { ModelVersionTable } from '@/components/admin/ModelVersionTable'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { Button } from '@/components/ui/Button'
import toast from 'react-hot-toast'

const MOCK_MODELS: ModelVersionDto[] = [
  { id: '1', agentName: 'symptom-extractor' as any, version: 'v2.4.1', filePath: 'models/symptom_extractor.py', isActive: true, loadedAt: new Date().toISOString(), createdAt: new Date().toISOString() },
  { id: '2', agentName: 'rag-retriever' as any, version: 'v1.8.0', filePath: 'models/rag_retriever.py', isActive: true, loadedAt: new Date().toISOString(), createdAt: new Date().toISOString() },
  { id: '3', agentName: 'response-generator' as any, version: 'v0.9.5', filePath: 'models/response_generator.py', isActive: false, loadedAt: null, createdAt: new Date().toISOString() },
]

export default function ModelManagementPage() {
  const { t } = useLanguage()
  const [models, setModels] = useState<ModelVersionDto[]>(MOCK_MODELS)
  const [loading, setLoading] = useState(true)
  const [reloadingAgent, setReloadingAgent] = useState<string | undefined>()

  const loadModels = () => {
    setLoading(true)
    adminApi.getModels()
      .then(setModels)
      .catch(err => {
        console.warn('Models: API failed, using mock data.', err)
        setModels(MOCK_MODELS)
      })
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    loadModels()
  }, [])

  const handleReload = async (agentName: string) => {
    setReloadingAgent(agentName)
    try {
      await adminApi.reloadModel(agentName)
      toast.success(`Deployment request sent for "${agentName}"`)
    } catch {
      toast.error('Failed to initiate reload')
    } finally {
      setReloadingAgent(undefined)
    }
  }

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-8 p-4 md:p-8"
    >
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="flex items-center gap-4">
          <div className="p-4 bg-blue-600 rounded-3xl shadow-xl shadow-blue-500/20 text-white">
            <Cpu size={28} strokeWidth={2.5} />
          </div>
          <div>
            <h1 className="text-3xl font-black text-slate-800 dark:text-white tracking-tight">{t('aiModelManagement')}</h1>
            <p className="text-slate-500 dark:text-slate-400 font-medium">{t('monitorManageModels')}</p>
          </div>
        </div>
        <div className="flex gap-3">
           <Button variant="outline" onClick={loadModels} className="gap-2">
             <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />{t('refresh')}</Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="border-0 shadow-xl rounded-3xl bg-emerald-50/50 dark:bg-emerald-950/20 border border-emerald-100 dark:border-emerald-900/30">
          <CardContent className="p-6 flex items-center gap-4">
            <div className="p-3 bg-emerald-500 rounded-2xl text-white shadow-lg"><Zap size={20} /></div>
            <div>
              <p className="text-sm font-bold text-emerald-800 dark:text-emerald-400 uppercase tracking-wider">{t('systemStatus')}</p>
              <p className="text-xl font-black text-emerald-950 dark:text-emerald-300">{t('optimal')}</p>
            </div>
          </CardContent>
        </Card>
        <Card className="border-0 shadow-xl rounded-3xl bg-blue-50/50 dark:bg-blue-950/20 border border-blue-100 dark:border-blue-900/30">
          <CardContent className="p-6 flex items-center gap-4">
            <div className="p-3 bg-blue-500 rounded-2xl text-white shadow-lg"><Shield size={20} /></div>
            <div>
              <p className="text-sm font-bold text-blue-800 dark:text-blue-400 uppercase tracking-wider">{t('security')}</p>
              <p className="text-xl font-black text-blue-950 dark:text-blue-300">{t('encrypted')}</p>
            </div>
          </CardContent>
        </Card>
        <Card className="border-0 shadow-xl rounded-3xl bg-indigo-50/50 dark:bg-indigo-950/20 border border-indigo-100 dark:border-indigo-900/30">
          <CardContent className="p-6 flex items-center gap-4">
            <div className="p-3 bg-indigo-500 rounded-2xl text-white shadow-lg"><Database size={20} /></div>
            <div>
              <p className="text-sm font-bold text-indigo-800 dark:text-indigo-400 uppercase tracking-wider">{t('storage')}</p>
              <p className="text-xl font-black text-indigo-950 dark:text-indigo-300">{t('stable')}</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="border border-slate-100 dark:border-slate-800 shadow-2xl rounded-3xl overflow-hidden bg-white dark:bg-slate-900">
        <CardHeader className="p-6 border-b border-slate-100 dark:border-slate-800/80 flex flex-row items-center justify-between bg-white/40 dark:bg-slate-900/40">
          <div>
            <CardTitle className="text-xl font-bold">{t('deployedModels')}</CardTitle>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 font-medium">{t('inventoryAgentModels')}</p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 dark:bg-slate-950 rounded-xl text-xs font-bold text-slate-600 dark:text-slate-300">
             <Activity size={14} className="text-blue-500" /> {models.length} Models Found
          </div>
        </CardHeader>
        <div className="p-0">
          {loading && !models.length ? (
            <div className="p-20 flex flex-col items-center justify-center space-y-4">
               <Loader2 className="animate-spin text-blue-500" size={40} />
               <p className="text-slate-400 font-bold">{t('initializingEnv')}</p>
            </div>
          ) : (
            <ModelVersionTable 
              models={models} 
              onReload={handleReload} 
              reloadingAgent={reloadingAgent} 
            />
          )}
        </div>
      </Card>
    </motion.div>
  )
}
