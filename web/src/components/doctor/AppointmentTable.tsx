import { useState } from 'react'
import { CheckCircle, XCircle, CheckCheck, Eye, Undo2, Trash2 } from 'lucide-react'
import type { AppointmentDto } from '@/lib/types'
import { StatusBadge } from '@/components/ui/Badge'
import { ConfirmDialog, Modal } from '@/components/ui/Modal'
import { formatDateTime } from '@/lib/utils'

interface AppointmentTableProps {
  appointments: AppointmentDto[]
  onConfirm?: (id: string) => Promise<void>
  onUnconfirm?: (id: string) => Promise<void>
  onCancel?: (id: string) => Promise<void>
  onComplete?: (id: string) => Promise<void>
  onDelete?: (id: string) => Promise<void>
}

const formatPaymentMethod = (value?: string | null) => {
  if (!value) return 'Not specified'
  const lowered = value.toLowerCase()
  if (lowered.includes('cash')) return 'Cash'
  if (lowered.includes('arrival')) return 'On arrival'
  if (lowered.includes('card') || lowered.includes('visa') || lowered.includes('master')) return 'Card'
  return value
}

export function AppointmentTable({
  appointments,
  onConfirm,
  onUnconfirm,
  onCancel,
  onComplete,
  onDelete,
}: AppointmentTableProps) {
  const [actionItem, setActionItem] = useState<{ appointment: AppointmentDto; action: 'confirm' | 'unconfirm' | 'cancel' | 'complete' | 'delete' } | null>(null)
  const [detailItem, setDetailItem] = useState<AppointmentDto | null>(null)
  const [loadingId, setLoadingId] = useState<string | null>(null)

  const handleAction = async () => {
    if (!actionItem) return
    const { appointment, action } = actionItem
    setLoadingId(appointment.id)
    try {
      if (action === 'confirm') await onConfirm?.(appointment.id)
      if (action === 'unconfirm') await onUnconfirm?.(appointment.id)
      if (action === 'cancel') await onCancel?.(appointment.id)
      if (action === 'complete') await onComplete?.(appointment.id)
      if (action === 'delete') await onDelete?.(appointment.id)
    } finally {
      setLoadingId(null)
      setActionItem(null)
    }
  }

  const actionLabels = {
    confirm: { title: 'Confirm Appointment', msg: 'Are you sure you want to confirm this appointment?', btn: 'Confirm', variant: 'primary' as const },
    unconfirm: { title: 'Move Back To Pending', msg: 'Do you want to move this appointment back to pending?', btn: 'Unconfirm', variant: 'danger' as const },
    cancel: { title: 'Cancel Appointment', msg: 'Are you sure you want to cancel this appointment?', btn: 'Cancel', variant: 'danger' as const },
    complete: { title: 'Complete Appointment', msg: 'Are you sure you want to mark this appointment as completed?', btn: 'Complete', variant: 'primary' as const },
    delete: { title: 'Delete Appointment', msg: 'Are you sure you want to delete/cancel this appointment?', btn: 'Delete', variant: 'danger' as const },
  }

  return (
    <>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-100">
              <th className="px-4 py-3 text-right font-semibold text-gray-600">Patient</th>
              <th className="px-4 py-3 text-right font-semibold text-gray-600">Date/Time</th>
              <th className="px-4 py-3 text-right font-semibold text-gray-600">Payment</th>
              <th className="px-4 py-3 text-right font-semibold text-gray-600">Status</th>
              <th className="px-4 py-3 text-right font-semibold text-gray-600">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {appointments.length === 0 ? (
              <tr>
                <td colSpan={5} className="py-16 text-center text-gray-400 text-sm">
                  No appointments
                </td>
              </tr>
            ) : appointments.map((appt) => (
              <tr key={appt.id} className="hover:bg-gray-50/80 transition-colors">
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-full bg-primary-100 flex items-center justify-center">
                      <span className="text-primary-700 text-xs font-semibold">{appt.patientName.charAt(0)}</span>
                    </div>
                    <span className="font-medium text-gray-800">{appt.patientName}</span>
                  </div>
                </td>
                <td className="px-4 py-3 text-gray-600 whitespace-nowrap">{formatDateTime(appt.scheduledAt)}</td>
                <td className="px-4 py-3 text-gray-600 whitespace-nowrap">{formatPaymentMethod(appt.paymentMethod)}</td>
                <td className="px-4 py-3"><StatusBadge status={appt.status} /></td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-1">
                    <button onClick={() => setDetailItem(appt)} className="p-1.5 rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-600" title="Details">
                      <Eye size={14} />
                    </button>
                    {appt.status === 'Pending' && (
                      <>
                        <button
                          onClick={() => setActionItem({ appointment: appt, action: 'confirm' })}
                          disabled={loadingId === appt.id}
                          className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl text-green-600 bg-green-50 hover:bg-green-100 transition-all shadow-sm active:scale-95"
                          title="Confirm"
                        >
                          <CheckCircle size={14} />
                          <span className="text-[10px] font-extrabold tracking-wider">CONFIRM</span>
                        </button>
                        <button
                          onClick={() => setActionItem({ appointment: appt, action: 'cancel' })}
                          disabled={loadingId === appt.id}
                          className="p-1.5 rounded-xl text-red-400 hover:bg-red-50 hover:text-red-600 transition-colors"
                          title="Cancel"
                        >
                          <XCircle size={14} />
                        </button>
                        <button
                          onClick={() => setActionItem({ appointment: appt, action: 'complete' })}
                          disabled={loadingId === appt.id}
                          className="p-1.5 rounded-lg text-blue-500 hover:bg-blue-50"
                          title="Complete"
                        >
                          <CheckCheck size={14} />
                        </button>
                      </>
                    )}
                    {appt.status === 'Confirmed' && (
                      <>
                        <button
                          onClick={() => setActionItem({ appointment: appt, action: 'unconfirm' })}
                          disabled={loadingId === appt.id}
                          className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl text-amber-600 bg-amber-50 hover:bg-amber-100 transition-all shadow-sm active:scale-95"
                          title="Unconfirm"
                        >
                          <Undo2 size={14} />
                          <span className="text-[10px] font-extrabold tracking-wider uppercase">Unconfirm</span>
                        </button>
                        <button
                          onClick={() => setActionItem({ appointment: appt, action: 'complete' })}
                          disabled={loadingId === appt.id}
                          className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl text-blue-600 bg-blue-50 hover:bg-blue-100 transition-all shadow-sm active:scale-95"
                          title="Complete"
                        >
                          <CheckCheck size={14} />
                          <span className="text-[10px] font-extrabold tracking-wider uppercase">Complete</span>
                        </button>
                      </>
                    )}
                    {appt.status !== 'Cancelled' && appt.status !== 'Completed' && (
                      <button
                        onClick={() => setActionItem({ appointment: appt, action: 'delete' })}
                        disabled={loadingId === appt.id}
                        className="p-1.5 rounded-lg text-red-500 hover:bg-red-50"
                        title="Delete"
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {actionItem && (
        <ConfirmDialog
          open={!!actionItem}
          onClose={() => setActionItem(null)}
          onConfirm={handleAction}
          title={actionLabels[actionItem.action].title}
          message={`${actionLabels[actionItem.action].msg}\nPatient: ${actionItem.appointment.patientName}`}
          confirmLabel={actionLabels[actionItem.action].btn}
          variant={actionLabels[actionItem.action].variant}
        />
      )}

      <Modal open={!!detailItem} onClose={() => setDetailItem(null)} title="Appointment Details" size="md">
        {detailItem && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div><p className="text-xs text-gray-400 mb-1">Patient</p><p className="font-medium">{detailItem.patientName}</p></div>
              <div><p className="text-xs text-gray-400 mb-1">Doctor</p><p className="font-medium">{detailItem.doctorName}</p></div>
              <div><p className="text-xs text-gray-400 mb-1">Date/Time</p><p className="font-medium">{formatDateTime(detailItem.scheduledAt)}</p></div>
              <div><p className="text-xs text-gray-400 mb-1">Status</p><StatusBadge status={detailItem.status} /></div>
              <div><p className="text-xs text-gray-400 mb-1">Payment</p><p className="font-medium">{formatPaymentMethod(detailItem.paymentMethod)}</p></div>
            </div>
            {detailItem.notes && (
              <div>
                <p className="text-xs text-gray-400 mb-1">Notes</p>
                <p className="text-sm text-gray-700 bg-gray-50 rounded-lg p-3">{detailItem.notes}</p>
              </div>
            )}
          </div>
        )}
      </Modal>
    </>
  )
}
