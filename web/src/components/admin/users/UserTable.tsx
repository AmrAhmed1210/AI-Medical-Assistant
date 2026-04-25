import { motion, AnimatePresence } from 'framer-motion'
import { Mail, User, Trash2, ToggleLeft, ToggleRight, Check, X, Crown, Stethoscope, ShieldCheck } from 'lucide-react'
import type { UserDto, UserRole } from '@/lib/types'
import { Badge } from '@/components/ui/Badge'
import { Table, TableHeader, TableRow, TableHead, TableCell } from '@/components/ui/Table'
import { ROLE_CONFIG } from './constants'
import { formatDateTime } from '@/lib/utils'

interface UserTableProps {
  users: UserDto[]
  onToggle: (id: number) => Promise<void>
  onDelete: (id: number, role: string) => void
}

export const UserTable = ({ users, onToggle, onDelete }: UserTableProps) => {
  
  const getRoleBadge = (role: string) => {
    const config = ROLE_CONFIG[role as UserRole] || { variant: 'info', label: 'Unknown', icon: null }
    const Icon = role === 'Admin' ? Crown : role === 'Doctor' ? Stethoscope : User
    return (
      <Badge variant={config.variant} className="px-3 py-1 text-[10px] font-black uppercase tracking-tighter shadow-sm">
        <Icon className="w-3 h-3" />
        {config.label}
      </Badge>
    )
  }
  
  const getStatusBadge = (isActive: boolean) => (
    <div className={`flex items-center gap-1.5 ${isActive ? 'text-emerald-500' : 'text-slate-300'}`}>
      <div className={`w-2 h-2 rounded-full animate-pulse ${isActive ? 'bg-emerald-500' : 'bg-slate-300'}`} />
      <span className="text-[11px] font-bold uppercase">{isActive ? 'Online' : 'Offline'}</span>
    </div>
  )
  
  if (!users || users.length === 0) {
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center py-24 text-center">
        <div className="p-8 rounded-[40px] bg-slate-50 mb-6 shadow-inner">
          <User className="w-12 h-12 text-slate-300" />
        </div>
        <h4 className="text-xl font-bold text-slate-800">No Entities Found</h4>
        <p className="text-slate-400 mt-1">Refine your search parameters or add a new record.</p>
      </motion.div>
    )
  }
  
  return (
    <Table>
      <TableHeader>
        <TableRow className="bg-slate-50/50 border-0">
          <TableHead className="px-8 py-5 text-[11px] font-black uppercase text-slate-400 tracking-widest rounded-tr-3xl">User / المستخدم</TableHead>
          <TableHead className="px-8 py-5 text-[11px] font-black uppercase text-slate-400 tracking-widest">Role & Status / الدور والحالة</TableHead>
          <TableHead className="px-8 py-5 text-[11px] font-black uppercase text-slate-400 tracking-widest">Created At / تاريخ الإنشاء</TableHead>
          <TableHead className="px-8 py-5 text-[11px] font-black uppercase text-slate-400 tracking-widest text-center rounded-tl-3xl">Actions / إجراءات</TableHead>
        </TableRow>
      </TableHeader>
      <tbody>
        <AnimatePresence mode="popLayout">
          {users.map((user, index) => (
            <motion.tr
              key={user.id}
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, x: -30 }}
              transition={{ delay: index * 0.05 }}
              className="group border-b border-slate-50/80 hover:bg-blue-50/30 transition-all duration-500"
            >
              <TableCell className="px-8 py-6">
                <div className="flex items-center gap-4">
                  {/* Legendary Avatar */}
                  <div className="relative">
                    <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white text-xl font-black shadow-xl shadow-blue-500/30 group-hover:scale-110 transition-transform duration-500 overflow-hidden">
                      {user.photoUrl ? (
                        <img src={user.photoUrl} alt={user.name} className="w-full h-full object-cover" />
                      ) : (
                        user.name.charAt(0)
                      )}
                    </div>
                    {user.isActive && <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-emerald-500 border-4 border-white shadow-lg" />}
                  </div>
                  
                  {/* STACKED INFO: Name under each other */}
                  <div className="flex flex-col gap-0.5">
                    <span className="text-lg font-black text-slate-800 tracking-tight group-hover:text-blue-600 transition-colors">{user.name}</span>
                    <div className="flex items-center gap-2 text-slate-400 text-xs font-medium">
                      <Mail size={12} className="shrink-0" />
                      {user.email}
                    </div>
                  </div>
                </div>
              </TableCell>

              <TableCell className="px-8 py-6">
                <div className="flex flex-col items-start gap-2">
                  {getRoleBadge(user.role)}
                  {getStatusBadge(user.isActive)}
                </div>
              </TableCell>

              <TableCell className="px-8 py-6">
                <div className="flex flex-col gap-1">
                  <span className="text-xs font-bold text-slate-700">Registered</span>
                  <div className="flex items-center gap-1.5 text-[10px] text-slate-400 font-black uppercase">
                     <ShieldCheck size={12} className="text-blue-400" /> 
                     {user.createdAt ? formatDateTime(user.createdAt) : '—'}
                  </div>
                </div>
              </TableCell>

              <TableCell className="px-8 py-6">
                <div className="flex items-center justify-center gap-3">
                  <motion.button
                    whileHover={{ scale: 1.12, rotate: 5 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => onToggle(user.id)}
                    className={`p-3 rounded-2xl transition-all duration-500 shadow-md hover:shadow-xl ${
                      user.isActive 
                        ? 'bg-white border border-amber-100 text-amber-500 hover:bg-amber-500 hover:text-white' 
                        : 'bg-white border border-emerald-100 text-emerald-500 hover:bg-emerald-500 hover:text-white'
                    }`}
                  >
                    {user.isActive ? <ToggleRight size={20} /> : <ToggleLeft size={20} />}
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.12, rotate: -5 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => { if(confirm('Permanently wipe this entry?')) onDelete(user.id, user.role) }}
                    className="p-3 rounded-2xl bg-white border border-rose-100 text-rose-500 hover:bg-rose-500 hover:text-white shadow-md hover:shadow-xl transition-all duration-500"
                  >
                    <Trash2 size={20} />
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