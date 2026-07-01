import { useState, useMemo } from 'react'
import { CheckCircle, XCircle, CheckCheck, Eye, Undo2, Trash2, CalendarClock, UserX } from 'lucide-react'
import type { AppointmentDto } from '@/lib/types'
import { StatusBadge } from '@/components/ui/Badge'
import { ConfirmDialog, Modal } from '@/components/ui/Modal'
import { formatDateTime } from '@/lib/utils'
import { useLanguage } from '@/lib/language'

interface AppointmentTableProps {
  appointments: AppointmentDto[]
  onConfirm?: (id: string) => Promise<void> | void
  onUnconfirm?: (id: string) => Promise<void> | void
  onCancel?: (id: string) => Promise<void> | void
  onComplete?: (id: string) => Promise<void> | void
  onDelete?: (id: string) => Promise<void> | void
  onNoShow?: (id: string) => Promise<void> | void
  onReschedule?: (id: string) => Promise<void> | void
  showSecretaryActions?: boolean
}

export function AppointmentTable({
  appointments,
  onConfirm,
  onUnconfirm,
  onCancel,
  onComplete,
  onDelete,
  onNoShow,
  onReschedule,
  showSecretaryActions,
}: AppointmentTableProps) {
  const { t, isRTL } = useLanguage()
  const [actionItem, setActionItem] = useState<{ appointment: AppointmentDto; action: 'confirm' | 'unconfirm' | 'cancel' | 'complete' | 'delete' | 'noShow' | 'reschedule' } | null>(null)
  const [detailItem, setDetailItem] = useState<AppointmentDto | null>(null)
  const [loadingId, setLoadingId] = useState<string | null>(null)

  const formatPaymentMethod = (value?: string | null) => {
    if (!value) return t('notSpecified')
    const lowered = value.toLowerCase()
    if (lowered.includes('cash')) return t('cash')
    if (lowered.includes('arrival')) return t('onArrival')
    if (lowered.includes('card') || lowered.includes('visa') || lowered.includes('master')) return t('card')
    return value
  }

  const actionLabels = useMemo(() => ({
    confirm: { title: t('confirmApptTitle'), msg: t('confirmApptMsg'), btn: t('confirm'), variant: 'primary' as const },
    unconfirm: { title: t('unconfirmApptTitle'), msg: t('unconfirmApptMsg'), btn: t('unconfirm'), variant: 'danger' as const },
    cancel: { title: t('cancelApptTitle'), msg: t('cancelApptMsg'), btn: t('cancel'), variant: 'danger' as const },
    complete: { title: t('completeApptTitle'), msg: t('completeApptMsg'), btn: t('complete'), variant: 'primary' as const },
    delete: { title: t('deleteApptTitle'), msg: t('deleteApptMsg'), btn: t('delete'), variant: 'danger' as const },
    noShow: { title: t('noShowTitle'), msg: t('noShowMsg'), btn: t('noShow'), variant: 'danger' as const },
    reschedule: { title: t('rescheduleApptTitle'), msg: t('rescheduleApptMsg'), btn: t('reschedule'), variant: 'primary' as const },
  }), [t])

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
      if (action === 'noShow') await onNoShow?.(appointment.id)
      if (action === 'reschedule') await onReschedule?.(appointment.id)
    } finally {
      setLoadingId(null)
      setActionItem(null)
    }
  }

  return (
    <>
      <div className="overflow-x-auto" dir={isRTL ? 'rtl' : 'ltr'}>
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-100">
              <th className={`px-4 py-3 font-semibold text-gray-600 ${isRTL ? 'text-right' : 'text-left'}`}>{t('patientName')}</th>
              <th className={`px-4 py-3 font-semibold text-gray-600 ${isRTL ? 'text-right' : 'text-left'}`}>{t('dateTime')}</th>
              <th className={`px-4 py-3 font-semibold text-gray-600 ${isRTL ? 'text-right' : 'text-left'}`}>{t('paymentMethod')}</th>
              <th className={`px-4 py-3 font-semibold text-gray-600 ${isRTL ? 'text-right' : 'text-left'}`}>{t('statusLabel')}</th>
              <th className={`px-4 py-3 font-semibold text-gray-600 ${isRTL ? 'text-right' : 'text-left'}`}>{t('actions')}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {appointments.length === 0 ? (
              <tr>
                <td colSpan={5} className="py-16 text-center text-gray-400 text-sm">
                  {t('noAppts')}
                </td>
              </tr>
            ) : appointments.map((appt) => (
              <tr key={appt.id} className="hover:bg-gray-50/80 transition-colors">
                <td className="px-4 py-3">
                  <div className={`flex items-center gap-2 ${isRTL ? 'flex-row-reverse' : ''}`}>
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
                  <div className={`flex items-center gap-1 ${isRTL ? 'flex-row-reverse' : ''}`}>
                    <button onClick={() => setDetailItem(appt)} className="p-1.5 rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-600" title={t('details')}>
                      <Eye size={14} />
                    </button>
                    {appt.status === 'Pending' && (
                      <>
                        <button
                          onClick={() => setActionItem({ appointment: appt, action: 'confirm' })}
                          disabled={loadingId === appt.id}
                          className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl text-green-600 bg-green-50 hover:bg-green-100 transition-all shadow-sm active:scale-95"
                          title={t('confirm')}
                        >
                          <CheckCircle size={14} />
                          <span className="text-[10px] font-extrabold tracking-wider uppercase">{t('confirm')}</span>
                        </button>
                        <button
                          onClick={() => setActionItem({ appointment: appt, action: 'cancel' })}
                          disabled={loadingId === appt.id}
                          className="p-1.5 rounded-xl text-red-400 hover:bg-red-50 hover:text-red-600 transition-colors"
                          title={t('cancel')}
                        >
                          <XCircle size={14} />
                        </button>
                        {onComplete && (
                          <button
                            onClick={() => setActionItem({ appointment: appt, action: 'complete' })}
                            disabled={loadingId === appt.id}
                            className="p-1.5 rounded-lg text-blue-500 hover:bg-blue-50"
                            title={t('complete')}
                          >
                            <CheckCheck size={14} />
                          </button>
                        )}
                      </>
                    )}
                    {appt.status === 'Confirmed' && (
                      <>
                        <button
                          onClick={() => setActionItem({ appointment: appt, action: 'unconfirm' })}
                          disabled={loadingId === appt.id}
                          className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl text-amber-600 bg-amber-50 hover:bg-amber-100 transition-all shadow-sm active:scale-95"
                          title={t('unconfirm')}
                        >
                          <Undo2 size={14} />
                          <span className="text-[10px] font-extrabold tracking-wider uppercase">{t('unconfirm')}</span>
                        </button>
                        {onComplete && (
                          <button
                            onClick={() => setActionItem({ appointment: appt, action: 'complete' })}
                            disabled={loadingId === appt.id}
                            className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl text-blue-600 bg-blue-50 hover:bg-blue-100 transition-all shadow-sm active:scale-95"
                            title={t('complete')}
                          >
                            <CheckCheck size={14} />
                            <span className="text-[10px] font-extrabold tracking-wider uppercase">{t('complete')}</span>
                          </button>
                        )}
                        {showSecretaryActions && (
                          <>
                            <button
                              onClick={() => setActionItem({ appointment: appt, action: 'noShow' })}
                              disabled={loadingId === appt.id}
                              className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl text-red-600 bg-red-50 hover:bg-red-100 transition-all shadow-sm active:scale-95"
                              title={t('noShow')}
                            >
                              <UserX size={14} />
                              <span className="text-[10px] font-extrabold tracking-wider uppercase">{t('noShow')}</span>
                            </button>
                            <button
                              onClick={() => setActionItem({ appointment: appt, action: 'reschedule' })}
                              disabled={loadingId === appt.id}
                              className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl text-purple-600 bg-purple-50 hover:bg-purple-100 transition-all shadow-sm active:scale-95"
                              title={t('reschedule')}
                            >
                              <CalendarClock size={14} />
                              <span className="text-[10px] font-extrabold tracking-wider uppercase">{t('reschedule')}</span>
                            </button>
                          </>
                        )}
                      </>
                    )}
                    {appt.status !== 'Cancelled' && appt.status !== 'Completed' && onDelete && (
                      <button
                        onClick={() => setActionItem({ appointment: appt, action: 'delete' })}
                        disabled={loadingId === appt.id}
                        className="p-1.5 rounded-lg text-red-500 hover:bg-red-50"
                        title={t('delete')}
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
          message={`${actionLabels[actionItem.action].msg}\n${t('patientName')}: ${actionItem.appointment.patientName}`}
          confirmLabel={actionLabels[actionItem.action].btn}
          variant={actionLabels[actionItem.action].variant}
        />
      )}

      <Modal open={!!detailItem} onClose={() => setDetailItem(null)} title={t('apptDetails')} size="md">
        {detailItem && (
          <div className="space-y-4" dir={isRTL ? 'rtl' : 'ltr'}>
            <div className="grid grid-cols-2 gap-4">
              <div><p className="text-xs text-gray-400 mb-1">{t('patientName')}</p><p className="font-medium">{detailItem.patientName}</p></div>
              <div><p className="text-xs text-gray-400 mb-1">{t('doctor')}</p><p className="font-medium">{detailItem.doctorName}</p></div>
              {detailItem.patientPhone && (
                <div><p className="text-xs text-gray-400 mb-1">{t('phone')}</p><p className="font-bold text-primary-600">{detailItem.patientPhone}</p></div>
              )}
              <div><p className="text-xs text-gray-400 mb-1">{t('dateTime')}</p><p className="font-medium">{formatDateTime(detailItem.scheduledAt)}</p></div>
              <div><p className="text-xs text-gray-400 mb-1">{t('statusLabel')}</p><StatusBadge status={detailItem.status} /></div>
              <div><p className="text-xs text-gray-400 mb-1">{t('paymentMethod')}</p><p className="font-medium">{formatPaymentMethod(detailItem.paymentMethod)}</p></div>
            </div>
            {detailItem.notes && (
              <div>
                <p className="text-xs text-gray-400 mb-1">{t('notes')}</p>
                <p className="text-sm text-gray-700 bg-gray-50 rounded-lg p-3">{detailItem.notes}</p>
              </div>
            )}
          </div>
        )}
      </Modal>
    </>
  )
}
