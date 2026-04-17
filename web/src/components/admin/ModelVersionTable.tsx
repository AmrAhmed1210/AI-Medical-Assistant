import { RefreshCw, CheckCircle, AlertCircle } from 'lucide-react'
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
          <tr className="bg-slate-50 border-b border-slate-100">
            <th className="px-6 py-4 text-left font-bold text-slate-600 uppercase tracking-wider">Agent Intelligence</th>
            <th className="px-6 py-4 text-left font-bold text-slate-600 uppercase tracking-wider">Core Version</th>
            <th className="px-6 py-4 text-left font-bold text-slate-600 uppercase tracking-wider">Storage Path</th>
            <th className="px-6 py-4 text-left font-bold text-slate-600 uppercase tracking-wider">Status</th>
            <th className="px-6 py-4 text-left font-bold text-slate-600 uppercase tracking-wider">Deployment Date</th>
            <th className="px-6 py-4 text-center font-bold text-slate-600 uppercase tracking-wider">Operations</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-50">
          {models.map((m) => (
            <tr key={m.modelId} className="hover:bg-slate-50 transition-colors group">
              <td className="px-6 py-4 font-bold text-slate-800">{m.agentName}</td>
              <td className="px-6 py-4">
                <span className="font-mono text-xs bg-slate-100 text-slate-600 px-2.5 py-1 rounded-lg border border-slate-200">{m.version}</span>
              </td>
              <td className="px-6 py-4">
                <code className="text-[10px] text-slate-400 font-mono truncate max-w-[150px] inline-block">{m.filePath}</code>
              </td>
              <td className="px-6 py-4">
                {m.isActive ? (
                  <span className="inline-flex items-center gap-1.5 text-xs font-bold text-emerald-700 bg-emerald-50 px-3 py-1.5 rounded-full border border-emerald-100">
                    <CheckCircle size={14} className="text-emerald-500" />
                    Operational
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 text-xs font-bold text-slate-400 bg-slate-50 px-3 py-1.5 rounded-full border border-slate-200">
                    <AlertCircle size={14} />
                    Standby
                  </span>
                )}
              </td>
              <td className="px-6 py-4 text-slate-500 text-xs font-medium">{formatDateTime(m.deployedAt)}</td>
              <td className="px-6 py-4">
                <div className="flex justify-center">
                  <button
                    onClick={() => onReload(m.agentName)}
                    disabled={reloadingAgent === m.agentName}
                    className={cn(
                      'flex items-center gap-2 px-4 py-2 text-xs rounded-xl font-bold transition-all shadow-sm',
                      reloadingAgent === m.agentName 
                        ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                        : 'bg-white border border-slate-200 text-blue-600 hover:bg-blue-600 hover:text-white hover:border-blue-600 hover:shadow-lg'
                    )}
                  >
                    <RefreshCw size={14} className={reloadingAgent === m.agentName ? 'animate-spin' : ''} />
                    {reloadingAgent === m.agentName ? 'Hot Reloading...' : 'Hot Reload'}
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
