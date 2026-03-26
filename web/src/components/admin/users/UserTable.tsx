import { motion, AnimatePresence } from 'framer-motion'
import { Mail, User, Trash2, ToggleLeft, ToggleRight, Check, X, Crown, Stethoscope } from 'lucide-react'
import type { UserDto, UserRole } from '@/lib/types'
import { Badge } from '@/components/ui/Badge'
import { Table, TableHeader, TableRow, TableHead, TableCell } from '@/components/ui/Table'
import { ROLE_CONFIG } from './constants'

interface UserTableProps {
  users: UserDto[]
  onToggle: (id: string) => Promise<void>
  onDelete: (id: string) => void
}

export const UserTable = ({ users, onToggle, onDelete }: UserTableProps) => {
  
  const getRoleBadge = (role: string) => {
    const config = ROLE_CONFIG[role as UserRole] || { variant: 'info', label: 'غير معروف', icon: null }
    const Icon = role === 'Admin' ? Crown : role === 'Doctor' ? Stethoscope : User
    return (
      <Badge variant={config.variant}>
        <Icon className="w-3 h-3" />
        {config.label}
      </Badge>
    )
  }
  
  const getStatusBadge = (isActive: boolean) => (
    <Badge variant={isActive ? 'success' : 'danger'}>
      {isActive ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
      {isActive ? 'نشط' : 'معطل'}
    </Badge>
  )
  
  if (!users || users.length === 0) {
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center py-16 text-center">
        <div className="p-4 rounded-2xl bg-gray-100/80 mb-4">
          <User className="w-8 h-8 text-gray-400" />
        </div>
        <p className="text-sm text-gray-500 font-medium">لا توجد مستخدمين لعرضهم</p>
      </motion.div>
    )
  }
  
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="rounded-r-2xl">المستخدم</TableHead>
          <TableHead>البريد الإلكتروني</TableHead>
          <TableHead>الدور</TableHead>
          <TableHead>الحالة</TableHead>
          <TableHead>تاريخ الإنشاء</TableHead>
          <TableHead className="rounded-l-2xl text-center">الإجراءات</TableHead>
        </TableRow>
      </TableHeader>
      <tbody>
        <AnimatePresence mode="popLayout">
          {users.map((user, index) => (
            <motion.tr
              key={user.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ delay: index * 0.05 }}
              className="group"
            >
              <TableCell className="font-medium">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-bold shadow-lg shadow-blue-500/20">
                    {user.name.charAt(0)}
                  </div>
                  <span className="text-gray-900 font-semibold">{user.name}</span>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2 text-gray-600">
                  <Mail className="w-4 h-4 text-gray-400" />
                  {user.email}
                </div>
              </TableCell>
              <TableCell>{getRoleBadge(user.role)}</TableCell>
              <TableCell>{getStatusBadge(user.isActive)}</TableCell>
              <TableCell className="text-gray-500">{new Date().toLocaleDateString('ar-EG')}</TableCell>
              <TableCell>
                <div className="flex items-center justify-center gap-2">
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => onToggle(user.id)}
                    className={`p-2.5 rounded-xl transition-all duration-200 ${
                      user.isActive 
                        ? 'bg-amber-100 text-amber-600 hover:bg-amber-200' 
                        : 'bg-green-100 text-green-600 hover:bg-green-200'
                    }`}
                    title={user.isActive ? 'تعطيل المستخدم' : 'تفعيل المستخدم'}
                  >
                    {user.isActive ? <ToggleRight className="w-5 h-5" /> : <ToggleLeft className="w-5 h-5" />}
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => { if(confirm('هل أنت متأكد من حذف هذا المستخدم؟')) onDelete(user.id) }}
                    className="p-2.5 rounded-xl bg-red-100 text-red-600 hover:bg-red-200 transition-all duration-200"
                    title="حذف المستخدم"
                  >
                    <Trash2 className="w-5 h-5" />
                  </motion.button>
                </div>
              </TableCell>
            </motion.tr>
          ))}
        </AnimatePresence>
      </tbody>
    </Table>
  )
}