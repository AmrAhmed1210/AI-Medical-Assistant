import { useEffect, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Search, UserPlus, Filter, Check, AlertCircle, Loader2,
  Mail, Lock, User, Stethoscope, Activity, RefreshCw,
  Database, WifiOff, X, Hash, ChevronRight, Layers, Crown
} from 'lucide-react'
import type { UserDto, UserRole, CreateUserRequest } from '@/lib/types'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import { UserModal } from '@/components/admin/users/UserModal'
import { Skeleton } from '@/components/ui/Skeleton'
import { UserTable } from '@/components/admin/users/UserTable'
import { useUsers } from '@/components/admin/users/useUsers'
import { Pagination } from '@/components/ui/Pagination'
import toast from 'react-hot-toast'

const PAGE_SIZE = 10

export default function AdminUsers() {
  const [search, setSearch] = useState('')
  const [roleFilter, setRoleFilter] = useState<UserRole | ''>('')
  const [showAddModal, setShowAddModal] = useState(false)

  const [form, setForm] = useState<CreateUserRequest>({
    fullName: '', email: '', password: '', role: 'Patient',
    specialityName: '', specialityNameAr: '', consultationFee: 0, yearsExperience: 0, bio: '',
  })
  const [formErrors, setFormErrors] = useState<Partial<Record<keyof CreateUserRequest, string>>>({})
  const [addLoading, setAddLoading] = useState(false)

  const {
    users, total, loading, fetchUsers, handleSearch, handleRoleFilter,
    handleToggle, handleDelete, handleAddUser, setPage, page,
    handleRetry, handleUseMockData, connectionStatus, error
  } = useUsers()

  const setField = (key: keyof CreateUserRequest, val: unknown) => {
    setForm(p => ({ ...p, [key]: val })); setFormErrors(p => ({ ...p, [key]: undefined }))
  }

  if (connectionStatus === 'error' && users.length === 0 && !loading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center p-8">
        <Card className="max-w-lg w-full border-0 shadow-2xl bg-white/80 backdrop-blur-xl">
          <CardContent className="p-10 text-center space-y-6">
            <div className="p-6 rounded-3xl bg-rose-50 border-2 border-rose-100 shadow-xl shadow-rose-500/10 inline-block font-black">
               <WifiOff size={48} className="text-rose-500" />
            </div>
             <h3 className="text-3xl font-black text-slate-900">Connection Lost</h3>
            <Button onClick={handleRetry} className="w-full h-14 rounded-2xl">Try Reconnecting</Button>
            <Button onClick={handleUseMockData} variant="outline" className="w-full h-14 rounded-2xl border-2">Use Demo Data</Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  const handleSubmitAddUser = async () => {
    const errors: Partial<Record<keyof CreateUserRequest, string>> = {}
    if (!form.fullName?.trim()) errors.fullName = 'Required'
    if (!form.email?.trim()) errors.email = 'Required'
    if (!form.password || form.password.length < 8) errors.password = '8+ characters'
    
    if (form.role === 'Doctor') {
      if (!form.specialityName?.trim()) errors.specialityName = 'Required'
      if (!form.specialityNameAr?.trim()) errors.specialityNameAr = 'Required'
      if (form.yearsExperience === undefined || form.yearsExperience < 0 || form.yearsExperience > 60) {
        errors.yearsExperience = 'Must be 0-60'
      }
      if (form.consultationFee === undefined || form.consultationFee < 0) {
        errors.consultationFee = 'Must be >= 0'
      }
    }

    if (Object.keys(errors).length > 0) {
      setFormErrors(errors)
      return
    }

    setAddLoading(true)
    await handleAddUser(form)
    setAddLoading(false)
    setShowAddModal(false)
    setForm({ fullName: '', email: '', password: '', role: 'Patient'})
  }

  return (
    <motion.div initial={{opacity:0}} animate={{opacity:1}} className="min-h-screen bg-[#f1f5f9] p-4 md:p-10 space-y-10">
      
      {/* Header with Glassmorphism */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-8">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
             <div className="p-3 bg-white rounded-2xl shadow-lg text-blue-600"><Layers size={24} /></div>
             <h1 className="text-4xl font-black text-slate-900 tracking-tighter">User Management</h1>
          </div>
          <p className="text-slate-500 font-bold ml-14">Manage all doctors, patients and admin accounts in one place</p>
        </div>
        <div className="flex gap-4">
          <Button variant="primary" onClick={() => setShowAddModal(true)} className="h-14 px-8 rounded-2xl gap-3 shadow-2xl shadow-blue-500/40 text-lg bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700">
            <UserPlus size={22} /> Add User
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-10">
        
        {/* SIDEBAR FILTERS: Vertical Stacking */}
        <div className="xl:col-span-3 space-y-8">
          <div className="bg-white/60 backdrop-blur-xl border border-white p-2 rounded-[32px] shadow-xl">
             <div className="p-6 pb-2"><h3 className="text-xs font-black uppercase tracking-widest text-slate-400">Filter by Role</h3></div>
             <div className="space-y-1.5 p-2">
                {(['', 'Admin', 'Doctor', 'Patient'] as const).map((role) => {
                  const labelMap: Record<string, string> = {
                    '': 'All Users',
                    'Admin': 'Admins',
                    'Doctor': 'Doctors',
                    'Patient': 'Patients'
                  }
                  return (
                  <motion.button
                    key={role}
                    whileHover={{ x: 5 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => handleRoleFilter(role)}
                    className={`w-full flex items-center justify-between px-5 py-4 rounded-2xl transition-all duration-300 font-black text-sm ${
                      roleFilter === role 
                      ? 'bg-blue-600 text-white shadow-xl shadow-blue-500/30' 
                      : 'text-slate-500 hover:bg-white hover:text-slate-800'
                    }`}
                  >
                    <div className="flex items-center gap-4">
                       {role === '' ? <Layers size={18} /> : role === 'Admin' ? <Crown size={18} /> : role === 'Doctor' ? <Stethoscope size={18} /> : <User size={18} />}
                       {labelMap[role]}
                    </div>
                    {roleFilter === role && <ChevronRight size={18} />}
                  </motion.button>
                  )
                })}
             </div>
          </div>

          {/* Quick Stats Block */}
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-[32px] p-8 text-white shadow-2xl relative overflow-hidden">
             <div className="relative z-10 space-y-4">
                <p className="text-slate-400 text-xs font-black uppercase tracking-widest">Total Registered Users</p>
                <h4 className="text-5xl font-black tracking-tighter">{total.toLocaleString()}</h4>
                <div className="h-1.5 w-full bg-slate-700 rounded-full overflow-hidden">
                   <motion.div initial={{width:0}} animate={{width:'75%'}} className="h-full bg-gradient-to-r from-blue-400 to-cyan-400" />
                </div>
                <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Platform capacity: 75% utilized</p>
             </div>
             <div className="absolute -bottom-10 -right-10 opacity-10 rotate-12"><Activity size={180} /></div>
          </div>
        </div>

        {/* MAIN CONTENT AREA */}
        <div className="xl:col-span-9 space-y-8">
          <Card className="border-0 shadow-2xl rounded-[40px] overflow-hidden bg-white/90 backdrop-blur-2xl">
            {/* Horizontal Search Only */}
            <div className="p-8 border-b bg-white/40 flex items-center justify-between gap-6">
              <div className="relative flex-1">
                <Search className="absolute left-6 top-1/2 -translate-y-1/2 text-slate-300" size={22} />
                <Input 
                  value={search}
                  onChange={(e) => handleSearch(e.target.value)}
                  placeholder="Search users by name or email..."
                  className="h-16 pl-16 pr-6 bg-slate-50 border-0 focus:ring-blue-500/20 text-lg rounded-3xl"
                />
              </div>
              <Button variant="outline" onClick={handleRetry} className="h-16 w-16 rounded-3xl p-0 shrink-0">
                 <RefreshCw size={24} className={loading ? 'animate-spin' : ''} />
              </Button>
            </div>

            <CardContent className="p-0">
               {loading ? (
                 <div className="p-10 space-y-6">
                   {[...Array(4)].map((_, i) => <Skeleton key={i} className="h-24 w-full rounded-[32px]" />)}
                 </div>
               ) : (
                 <motion.div initial={{opacity:0}} animate={{opacity:1}} transition={{duration:0.5}}>
                    <UserTable users={users} onToggle={handleToggle} onDelete={handleDelete} />
                 </motion.div>
               )}
            </CardContent>

            {!loading && total > PAGE_SIZE && (
              <div className="p-8 bg-slate-50 border-t flex justify-end">
                <Pagination total={total} page={page} pageSize={PAGE_SIZE} onChange={setPage} />
              </div>
            )}
          </Card>
        </div>
      </div>

      <UserModal
        open={showAddModal}
        onClose={() => setShowAddModal(false)}
        form={form}
        errors={formErrors as Record<keyof CreateUserRequest, string>}
        loading={addLoading}
        onFieldChange={setField}
        onSubmit={handleSubmitAddUser}
      />
    </motion.div>
  )
}