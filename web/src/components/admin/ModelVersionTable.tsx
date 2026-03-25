import { RefreshCw, CheckCircle } from 'lucide-react'
import type { ModelVersionDto } from '@/lib/types'
import { formatDateTime } from '@/lib/utils'
import { cn } from '@/lib/utils'

interface ModelVersionTableProps {
  models: ModelVersionDto[]
  onReload: (agentName: string) => void
  reloadingAgent?: string
}

export function ModelVersionTable({ models, onReload, reloadingAgent }: ModelVersionTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-gray-50 border-b border-gray-100">
            <th className="px-4 py-3 text-right font-semibold text-gray-600">اسم الوكيل</th>
            <th className="px-4 py-3 text-right font-semibold text-gray-600">الإصدار</th>
            <th className="px-4 py-3 text-right font-semibold text-gray-600">مسار الملف</th>
            <th className="px-4 py-3 text-right font-semibold text-gray-600">الحالة</th>
            <th className="px-4 py-3 text-right font-semibold text-gray-600">تاريخ النشر</th>
            <th className="px-4 py-3 text-right font-semibold text-gray-600">إعادة تحميل</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-50">
          {models.map((m) => (
            <tr key={m.modelId} className="hover:bg-gray-50">
              <td className="px-4 py-3 font-medium text-gray-800">{m.agentName}</td>
              <td className="px-4 py-3">
                <span className="font-mono text-xs bg-gray-100 px-2 py-0.5 rounded">{m.version}</span>
              </td>
              <td className="px-4 py-3 text-gray-500 font-mono text-xs truncate max-w-xs">{m.filePath}</td>
              <td className="px-4 py-3">
                {m.isActive ? (
                  <span className="inline-flex items-center gap-1.5 text-xs font-medium text-green-700 bg-green-100 px-2.5 py-1 rounded-full">
                    <CheckCircle size={12} />
                    نشط
                  </span>
                ) : (
                  <span className="inline-flex items-center text-xs font-medium text-gray-500 bg-gray-100 px-2.5 py-1 rounded-full">
                    غير نشط
                  </span>
                )}
              </td>
              <td className="px-4 py-3 text-gray-500 text-xs">{formatDateTime(m.deployedAt)}</td>
              <td className="px-4 py-3">
                <button
                  onClick={() => onReload(m.agentName)}
                  disabled={reloadingAgent === m.agentName}
                  className={cn(
                    'flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg font-medium transition-colors',
                    'm isActive ? "bg-primary-50 text-primary-600 hover:bg-primary-100" : "bg-gray-100 text-gray-500 hover:bg-gray-200"'
                  )}
                >
                  <RefreshCw size={12} className={reloadingAgent === m.agentName ? 'animate-spin' : ''} />
                  {reloadingAgent === m.agentName ? 'جاري...' : 'تحميل ساخن'}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
