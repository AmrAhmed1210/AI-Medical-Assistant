import { useEffect, useState } from 'react'
import { Cpu, RefreshCw } from 'lucide-react'
import { adminApi } from '@/api/adminApi'
import type { ModelVersionDto } from '@/lib/types'
import { ModelVersionTable } from '@/components/admin/ModelVersionTable'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import toast from 'react-hot-toast'

export default function ModelManagementPage() {
  const [models, setModels] = useState<ModelVersionDto[]>([])
  const [loading, setLoading] = useState(true)
  const [reloadingAgent, setReloadingAgent] = useState<string | undefined>()

  useEffect(() => {
    adminApi.getModels().then(setModels).finally(() => setLoading(false))
  }, [])

  const handleReload = async (agentName: string) => {
    setReloadingAgent(agentName)
    try {
      await adminApi.reloadModel(agentName)
      toast.success(`تم إرسال طلب إعادة تحميل "${agentName}"`)
    } catch {
      toast.error('فشل طلب إعادة التحميل')
    } finally {
      setReloadingAgent(undefined)
    }
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-primary-50 rounded-xl">
          <Cpu size={20} className="text-primary-600" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-gray-800">إدارة نماذج AI</h1>
          <p className="text-sm text-gray-500">إدارة وتحديث نماذج الذكاء الاصطناعي</p>
        </div>
      </div>

      <Card padding="none">
        <CardHeader className="px-5 pt-5 pb-0">
          <CardTitle>نسخ النماذج المتاحة</CardTitle>
          <button
            onClick={() => adminApi.getModels().then(setModels)}
            className="flex items-center gap-1.5 text-xs text-primary-600 hover:text-primary-700"
          >
            <RefreshCw size={12} />
            تحديث
          </button>
        </CardHeader>
        <div className="p-5">
          {loading ? <PageLoader /> : (
            <ModelVersionTable models={models} onReload={handleReload} reloadingAgent={reloadingAgent} />
          )}
        </div>
      </Card>
    </div>
  )
}
