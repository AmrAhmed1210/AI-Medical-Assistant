import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { adminApi } from '@/api/adminApi'
import toast from 'react-hot-toast'
import {
  Users, CheckCircle2, XCircle, Clock, Eye, ChevronDown,
  ChevronUp, Mail, Phone, Briefcase, FileText, MessageSquare
} from 'lucide-react'

interface DoctorApplication {
  id: number
  name: string
  email: string
  phone: string
  specialtyId: number
  specialtyName: string
  experience: number
  bio: string
  licenseNumber: string
  message: string
  documentUrl: string
  photoUrl?: string
  status: string
  submittedAt: string
  processedAt: string | null
}

type TabStatus = 'Pending' | 'Approved' | 'Rejected'

export default function ApplicationsPage() {
  const [activeTab, setActiveTab] = useState<TabStatus>('Pending')
  const [applications, setApplications] = useState<DoctorApplication[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [expandedId, setExpandedId] = useState<number | null>(null)
  const [actionLoading, setActionLoading] = useState<number | null>(null)
  const [rejectModal, setRejectModal] = useState<{ id: number; name: string } | null>(null)
  const [rejectReason, setRejectReason] = useState('')

  const fetchApplications = async (status: TabStatus) => {
    setIsLoading(true)
    try {
      const data = await adminApi.getApplications(status)
      setApplications(data)
    } catch {
      toast.error('Failed to load applications')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchApplications(activeTab)
    setExpandedId(null)
  }, [activeTab])

  const handleApprove = async (id: number, name: string) => {
    if (!window.confirm(`Approve application from ${name}?\n\nThis will create a doctor account with a temporary password.`)) return
    setActionLoading(id)
    try {
      await adminApi.approveApplication(id)
      toast.success(`✅ ${name}'s application approved! Doctor account created.`)
      fetchApplications(activeTab)
    } catch (err: any) {
      toast.error(err?.response?.data?.message || 'Failed to approve application')
    } finally {
      setActionLoading(null)
    }
  }

  const handleRejectConfirm = async () => {
    if (!rejectModal) return
    setActionLoading(rejectModal.id)
    try {
      await adminApi.rejectApplication(rejectModal.id, rejectReason)
      toast.success(`Application from ${rejectModal.name} rejected.`)
      setRejectModal(null)
      setRejectReason('')
      fetchApplications(activeTab)
    } catch (err: any) {
      toast.error(err?.response?.data?.message || 'Failed to reject application')
    } finally {
      setActionLoading(null)
    }
  }

  const tabs: { label: string; value: TabStatus; icon: React.ReactNode; color: string }[] = [
    { label: 'Pending', value: 'Pending', icon: <Clock size={16} />, color: 'text-amber-600' },
    { label: 'Approved', value: 'Approved', icon: <CheckCircle2 size={16} />, color: 'text-emerald-600' },
    { label: 'Rejected', value: 'Rejected', icon: <XCircle size={16} />, color: 'text-red-600' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-br from-primary-600 to-primary-400 rounded-xl blur-lg opacity-30" />
          <div className="relative bg-gradient-to-br from-primary-600 to-primary-500 rounded-xl p-3 shadow-lg">
            <Users size={28} className="text-white" />
          </div>
        </div>
        <div>
          <h1 className="text-xl font-bold text-gray-800">Doctor Applications</h1>
          <p className="text-sm text-gray-500 mt-0.5">Review and manage applicants</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
        <div className="flex border-b border-gray-100">
          {tabs.map(tab => (
            <button
              key={tab.value}
              onClick={() => setActiveTab(tab.value)}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-4 text-sm font-semibold transition-colors border-b-2 ${
                activeTab === tab.value
                  ? `border-primary-600 text-primary-600 bg-primary-50/40`
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
              }`}
            >
              <span className={activeTab === tab.value ? tab.color : ''}>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>

        {/* List */}
        <div className="divide-y divide-gray-50">
          {isLoading ? (
            <div className="py-16 text-center text-gray-400 text-sm">Loading applications...</div>
          ) : applications.length === 0 ? (
            <div className="py-16 text-center">
              <Users size={40} className="mx-auto text-gray-200 mb-3" />
              <p className="text-gray-400 text-sm">No {activeTab.toLowerCase()} applications</p>
            </div>
          ) : applications.map(app => (
            <ApplicationRow
              key={app.id}
              app={app}
              isExpanded={expandedId === app.id}
              onToggle={() => setExpandedId(expandedId === app.id ? null : app.id)}
              onApprove={() => handleApprove(app.id, app.name)}
              onReject={() => { setRejectModal({ id: app.id, name: app.name }); setRejectReason('') }}
              actionLoading={actionLoading === app.id}
              showActions={activeTab === 'Pending'}
            />
          ))}
        </div>
      </div>

      {/* Reject Modal */}
      <AnimatePresence>
        {rejectModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6"
            >
              <h3 className="text-lg font-bold text-gray-900 mb-1">Reject Application</h3>
              <p className="text-sm text-gray-500 mb-4">
                You are about to reject <span className="font-semibold text-gray-700">{rejectModal.name}'s</span> application.
              </p>
              <textarea
                rows={3}
                value={rejectReason}
                onChange={e => setRejectReason(e.target.value)}
                placeholder="Reason for rejection (optional)..."
                className="w-full text-sm bg-white text-gray-900 placeholder-gray-400 border border-gray-200 rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-red-400/30 shadow-sm resize-none"
              />
              <div className="flex gap-3 mt-4">
                <button
                  onClick={() => setRejectModal(null)}
                  className="flex-1 py-2.5 border border-gray-200 rounded-xl text-sm font-medium text-gray-600 hover:bg-gray-50 transition"
                >
                  Cancel
                </button>
                <button
                  onClick={handleRejectConfirm}
                  disabled={actionLoading !== null}
                  className="flex-1 py-2.5 bg-red-600 text-white rounded-xl text-sm font-bold hover:bg-red-700 disabled:opacity-50 transition"
                >
                  {actionLoading !== null ? 'Rejecting...' : 'Confirm Reject'}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function ApplicationRow({
  app, isExpanded, onToggle, onApprove, onReject, actionLoading, showActions
}: {
  app: DoctorApplication
  isExpanded: boolean
  onToggle: () => void
  onApprove: () => void
  onReject: () => void
  actionLoading: boolean
  showActions: boolean
}) {
  const statusColors: Record<string, string> = {
    Pending: 'bg-amber-100 text-amber-700',
    Approved: 'bg-emerald-100 text-emerald-700',
    Rejected: 'bg-red-100 text-red-600',
  }

  return (
    <div className="hover:bg-gray-50/50 transition-colors">
      {/* Summary row */}
      <div className="flex items-center gap-4 px-5 py-4 cursor-pointer" onClick={onToggle}>
        {app.photoUrl ? (
          <img src={app.photoUrl} alt={app.name} className="w-10 h-10 rounded-full object-cover shadow border border-gray-100 flex-shrink-0" />
        ) : (
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-500 to-primary-400 flex items-center justify-center flex-shrink-0 text-white font-bold text-sm shadow">
            {app.name.charAt(0).toUpperCase()}
          </div>
        )}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-gray-800">{app.name}</p>
          <p className="text-xs text-gray-500">{app.email} · {app.specialtyName}</p>
        </div>
        <span className={`text-xs font-semibold px-2.5 py-1 rounded-full ${statusColors[app.status] || ''}`}>
          {app.status}
        </span>
        <span className="text-xs text-gray-400 hidden sm:block">
          {new Date(app.submittedAt).toLocaleDateString()}
        </span>
        {isExpanded ? <ChevronUp size={16} className="text-gray-400 flex-shrink-0" /> : <ChevronDown size={16} className="text-gray-400 flex-shrink-0" />}
      </div>

      {/* Expanded detail */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 space-y-4 border-t border-gray-100 pt-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                <InfoCell icon={<Phone size={14} />} label="Phone" value={app.phone} />
                <InfoCell icon={<Briefcase size={14} />} label="Experience" value={`${app.experience} years`} />
                <InfoCell icon={<FileText size={14} />} label="License / ID" value={app.licenseNumber || '—'} />
              </div>

              <div>
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">Bio</p>
                <p className="text-sm text-gray-600 leading-relaxed">{app.bio || '—'}</p>
              </div>

              {app.message && (
                <div>
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1 flex items-center gap-1">
                    <MessageSquare size={12} /> Message to Admin
                  </p>
                  <p className="text-sm text-gray-600 italic bg-gray-50 rounded-xl px-3 py-2">{app.message}</p>
                </div>
              )}

              {app.documentUrl && (
                <div>
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">Document</p>
                  {app.documentUrl.startsWith('data:') ? (
                    <a
                      href={app.documentUrl}
                      download={`application-${app.id}-doc`}
                      className="inline-flex items-center gap-2 text-sm text-primary-600 hover:text-primary-700 font-medium"
                    >
                      <FileText size={14} /> Download Document
                    </a>
                  ) : (
                    <a href={app.documentUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 text-sm text-primary-600 hover:text-primary-700 font-medium">
                      <Eye size={14} /> View Document
                    </a>
                  )}
                </div>
              )}

              {app.processedAt && (
                <p className="text-xs text-gray-400">Processed: {new Date(app.processedAt).toLocaleString()}</p>
              )}

              {showActions && (
                <div className="flex gap-3 pt-2">
                  <button
                    onClick={e => { e.stopPropagation(); onApprove() }}
                    disabled={actionLoading}
                    className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-5 py-2.5 bg-emerald-600 text-white rounded-xl text-sm font-bold hover:bg-emerald-700 disabled:opacity-50 transition-all"
                  >
                    <CheckCircle2 size={15} />
                    {actionLoading ? 'Processing...' : 'Approve'}
                  </button>
                  <button
                    onClick={e => { e.stopPropagation(); onReject() }}
                    disabled={actionLoading}
                    className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-5 py-2.5 border border-red-200 text-red-600 rounded-xl text-sm font-semibold hover:bg-red-50 disabled:opacity-50 transition-all"
                  >
                    <XCircle size={15} />
                    Reject
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function InfoCell({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="bg-gray-50 rounded-xl px-3 py-2.5">
      <p className="text-xs text-gray-400 flex items-center gap-1 mb-0.5">{icon}{label}</p>
      <p className="text-sm font-medium text-gray-700">{value}</p>
    </div>
  )
}
