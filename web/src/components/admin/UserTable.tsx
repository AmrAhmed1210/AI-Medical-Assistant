import { useState } from 'react'
import { motion } from 'framer-motion'
import { UserCheck, UserX, Trash2, Shield, Stethoscope, User } from 'lucide-react'
import type { UserDto } from '@/lib/types'
import { formatDate } from '@/lib/utils'
import { Badge } from '@/components/ui/Badge'
import { ConfirmDialog } from '@/components/ui/Modal'
import { cn } from '@/lib/utils'

interface UserTableProps {
  users: UserDto[]
  onToggle: (id: string) => void
  onDelete: (id: string) => void
  isLoading?: boolean
}

const roleIcon = {
  Admin: <Shield size={13} />,
  Doctor: <Stethoscope size={13} />,
  Patient: <User size={13} />,
}
const roleColor = {
  Admin: 'purple' as const,
  Doctor: 'blue' as const,
  Patient: 'green' as const,
}
const roleLabel = {
  Admin: 'مدير',
  Doctor: 'طبيب',
  Patient: 'مريض',
}

export function UserTable({ users, onToggle, onDelete }: UserTableProps) {
  const [confirmToggle, setConfirmToggle] = useState<UserDto | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<UserDto | null>(null)

  return (
    <>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-100">
              <th className="px-4 py-3 text-right font-semibold text-gray-600">المستخدم</th>
              <th className="px-4 py-3 text-right font-semibold text-gray-600">الدور</th>
              <th className="px-4 py-3 text-right font-semibold text-gray-600">الحالة</th>
              <th className="px-4 py-3 text-right font-semibold text-gray-600">تاريخ الإنشاء</th>
              <th className="px-4 py-3 text-right font-semibold text-gray-600">الإجراءات</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {users.map((user, i) => (
              <motion.tr
                key={user.userId}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.03 }}
                className="hover:bg-gray-50/80 transition-colors"
              >
                <td className="px-4 py-3">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center flex-shrink-0">
                      <span className="text-primary-700 text-xs font-semibold">
                        {user.fullName.charAt(0)}
                      </span>
                    </div>
                    <div>
                      <p className="font-medium text-gray-800">{user.fullName}</p>
                      <p className="text-xs text-gray-400">{user.email}</p>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <Badge color={roleColor[user.role]}>
                    <span className="flex items-center gap-1">
                      {roleIcon[user.role]}
                      {roleLabel[user.role]}
                    </span>
                  </Badge>
                </td>
                <td className="px-4 py-3">
                  <span className={cn(
                    'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium',
                    user.isActive ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
                  )}>
                    <span className={cn('w-1.5 h-1.5 rounded-full', user.isActive ? 'bg-green-500' : 'bg-gray-400')} />
                    {user.isActive ? 'نشط' : 'معطل'}
                  </span>
                </td>
                <td className="px-4 py-3 text-gray-500">{formatDate(user.createdAt)}</td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setConfirmToggle(user)}
                      title={user.isActive ? 'تعطيل' : 'تفعيل'}
                      className={cn(
                        'p-1.5 rounded-lg transition-colors',
                        user.isActive
                          ? 'text-amber-500 hover:bg-amber-50'
                          : 'text-green-500 hover:bg-green-50'
                      )}
                    >
                      {user.isActive ? <UserX size={15} /> : <UserCheck size={15} />}
                    </button>
                    <button
                      onClick={() => setConfirmDelete(user)}
                      title="حذف"
                      className="p-1.5 rounded-lg text-red-400 hover:bg-red-50 transition-colors"
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      <ConfirmDialog
        open={!!confirmToggle}
        onClose={() => setConfirmToggle(null)}
        onConfirm={() => { if (confirmToggle) onToggle(confirmToggle.userId) }}
        title={confirmToggle?.isActive ? 'تعطيل المستخدم' : 'تفعيل المستخدم'}
        message={`هل تريد ${confirmToggle?.isActive ? 'تعطيل' : 'تفعيل'} حساب "${confirmToggle?.fullName}"؟`}
        confirmLabel={confirmToggle?.isActive ? 'تعطيل' : 'تفعيل'}
        variant={confirmToggle?.isActive ? 'danger' : 'primary'}
      />
      <ConfirmDialog
        open={!!confirmDelete}
        onClose={() => setConfirmDelete(null)}
        onConfirm={() => { if (confirmDelete) onDelete(confirmDelete.userId) }}
        title="حذف المستخدم"
        message={`هل تريد حذف حساب "${confirmDelete?.fullName}" نهائياً؟ لا يمكن التراجع عن هذا الإجراء.`}
        confirmLabel="حذف نهائياً"
      />
    </>
  )
}
