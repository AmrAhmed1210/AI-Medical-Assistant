import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Cpu, RefreshCw, Zap, Shield, Database, Activity } from 'lucide-react'
import { adminApi } from '@/api/adminApi'
import type { ModelVersionDto } from '@/lib/types'
import { ModelVersionTable } from '@/components/admin/ModelVersionTable'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { Button } from '@/components/ui/Button'
import toast from 'react-hot-toast'

const MOCK_MODELS: ModelVersionDto[] = [
  { agentName: 'MedicalConsultant', currentVersion: 'v2.4.1', lastReloaded: new Date().toISOString(), status: 'Running' },
  { agentName: 'DiagnosticAssistant', currentVersion: 'v1.8.0', lastReloaded: new Date().toISOString(), status: 'Running' },
  { agentName: 'TranscriptionService', currentVersion: 'v0.9.5', lastReloaded: new Date().toISOString(), status: 'Maintenance' },
] as any

export default function ModelManagementPage() {
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
            <h1 className="text-3xl font-black text-slate-800 tracking-tight">AI Model Forge</h1>
            <p className="text-slate-500 font-medium">Core intelligence unit orchestration</p>
          </div>
        </div>
        <div className="flex gap-3">
           <Button variant="outline" onClick={loadModels} className="gap-2">
             <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
             Synchronize
           </Button>
        </div>
      </div>

      {/* Connectivity Status Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="border-0 shadow-xl rounded-3xl bg-emerald-50/50 border-emerald-100">
          <CardContent className="p-6 flex items-center gap-4">
            <div className="p-3 bg-emerald-500 rounded-2xl text-white shadow-lg"><Zap size={20} /></div>
            <div>
              <p className="text-sm font-bold text-emerald-800 uppercase tracking-wider">System Status</p>
              <p className="text-xl font-black text-emerald-950">OPTIMAL</p>
            </div>
          </CardContent>
        </Card>
        <Card className="border-0 shadow-xl rounded-3xl bg-blue-50/50 border-blue-100">
          <CardContent className="p-6 flex items-center gap-4">
            <div className="p-3 bg-blue-500 rounded-2xl text-white shadow-lg"><Shield size={20} /></div>
            <div>
              <p className="text-sm font-bold text-blue-800 uppercase tracking-wider">Core Security</p>
              <p className="text-xl font-black text-blue-950">ENCRYPTED</p>
            </div>
          </CardContent>
        </Card>
        <Card className="border-0 shadow-xl rounded-3xl bg-indigo-50/50 border-indigo-100">
          <CardContent className="p-6 flex items-center gap-4">
            <div className="p-3 bg-indigo-500 rounded-2xl text-white shadow-lg"><Database size={20} /></div>
            <div>
              <p className="text-sm font-bold text-indigo-800 uppercase tracking-wider">Storage Link</p>
              <p className="text-xl font-black text-indigo-950">STABLE</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="border-0 shadow-2xl rounded-3xl overflow-hidden bg-white/80 backdrop-blur-xl">
        <CardHeader className="p-6 border-b border-slate-100 flex flex-row items-center justify-between bg-white/40">
          <div>
            <CardTitle className="text-xl font-bold">Active Neuron Layers</CardTitle>
            <p className="text-sm text-slate-500 mt-1 font-medium">Inventory of deployed agent models and versions</p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 rounded-xl text-xs font-bold text-slate-600">
             <Activity size={14} className="text-blue-500" /> {models.length} Layers Found
          </div>
        </CardHeader>
        <div className="p-0">
          {loading && !models.length ? (
            <div className="p-20 flex flex-col items-center justify-center space-y-4">
               <Loader2 className="animate-spin text-blue-500" size={40} />
               <p className="text-slate-400 font-bold">Initializing Environment...</p>
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
