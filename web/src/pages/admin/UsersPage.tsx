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
import { Modal } from '@/components/ui/Modal'
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
    fullName: '', email: '', passwordHash: '', role: 'Patient',
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
            <h3 className="text-3xl font-black text-slate-900">System Link Broken</h3>
            <Button onClick={handleRetry} className="w-full h-14 rounded-2xl">Re-establish Link</Button>
            <Button onClick={handleUseMockData} variant="outline" className="w-full h-14 rounded-2xl border-2">Switch to Demo Mode</Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  const handleSubmitAddUser = async () => {
    const errors: Partial<Record<keyof CreateUserRequest, string>> = {}
    if (!form.fullName.trim()) errors.fullName = 'Required'
    if (!form.email.trim()) errors.email = 'Required'
    if (!form.passwordHash || form.passwordHash.length < 8) errors.passwordHash = '8+ characters'
    
    if (Object.keys(errors).length > 0) {
      setFormErrors(errors)
      return
    }

    setAddLoading(true)
    await handleAddUser(form)
    setAddLoading(false)
    setShowAddModal(false)
    setForm({ fullName: '', email: '', passwordHash: '', role: 'Patient'})
  }

  return (
    <motion.div initial={{opacity:0}} animate={{opacity:1}} className="min-h-screen bg-[#f1f5f9] p-4 md:p-10 space-y-10">
      
      {/* Header with Glassmorphism */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-8">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
             <div className="p-3 bg-white rounded-2xl shadow-lg text-blue-600"><Layers size={24} /></div>
             <h1 className="text-4xl font-black text-slate-900 tracking-tighter">Account Ecosystem</h1>
          </div>
          <p className="text-slate-500 font-bold ml-14">Orchestrate and monitor all medical personnel and client profiles.</p>
        </div>
        <div className="flex gap-4">
          <Button variant="primary" onClick={() => setShowAddModal(true)} className="h-14 px-8 rounded-2xl gap-3 shadow-2xl shadow-blue-500/40 text-lg">
            <UserPlus size={22} /> Provision New Account
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-10">
        
        {/* SIDEBAR FILTERS: Vertical Stacking */}
        <div className="xl:col-span-3 space-y-8">
          <div className="bg-white/60 backdrop-blur-xl border border-white p-2 rounded-[32px] shadow-xl">
             <div className="p-6 pb-2"><h3 className="text-xs font-black uppercase tracking-widest text-slate-400">Filter Spectrum</h3></div>
             <div className="space-y-1.5 p-2">
                {(['', 'Admin', 'Doctor', 'Patient'] as const).map(role => (
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
                       {role === '' ? 'Omni View' : `${role}s`}
                    </div>
                    {roleFilter === role && <ChevronRight size={18} />}
                  </motion.button>
                ))}
             </div>
          </div>

          {/* Quick Stats Block */}
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-[32px] p-8 text-white shadow-2xl relative overflow-hidden">
             <div className="relative z-10 space-y-4">
                <p className="text-slate-400 text-xs font-black uppercase tracking-widest">Total Registry</p>
                <h4 className="text-5xl font-black tracking-tighter">{total.toLocaleString()}</h4>
                <div className="h-1.5 w-full bg-slate-700 rounded-full overflow-hidden">
                   <motion.div initial={{width:0}} animate={{width:'75%'}} className="h-full bg-blue-500" />
                </div>
                <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Storage allocation: 75% capacity</p>
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
                  placeholder="Global search into the matrix..."
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

      <Modal open={showAddModal} onClose={() => setShowAddModal(false)} title="System Provisioning" size="lg">
        <div className="space-y-8 py-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-xs font-black text-slate-400 uppercase tracking-widest ml-1">Full Legal Entity Name</label>
              <Input value={form.fullName} onChange={e => setField('fullName', e.target.value)} icon={<User size={18} />} error={formErrors.fullName} className="h-14 rounded-2xl" />
            </div>
            <div className="space-y-2">
              <label className="text-xs font-black text-slate-400 uppercase tracking-widest ml-1">Electronic Mail</label>
              <Input type="email" value={form.email} onChange={e => setField('email', e.target.value)} icon={<Mail size={18} />} error={formErrors.email} className="h-14 rounded-2xl" />
            </div>
            <div className="space-y-2">
              <label className="text-xs font-black text-slate-400 uppercase tracking-widest ml-1">Secure Passkey</label>
              <Input type="password" value={form.passwordHash} onChange={e => setField('passwordHash', e.target.value)} icon={<Lock size={18} />} error={formErrors.passwordHash} className="h-14 rounded-2xl" />
            </div>
            <div className="space-y-2">
              <label className="text-xs font-black text-slate-400 uppercase tracking-widest ml-1">Operational Authority</label>
              <Select value={form.role} onChange={e => setField('role', e.target.value as UserRole)} className="h-14 rounded-2xl font-bold">
                <option value="Patient">Patient</option>
                <option value="Doctor">Doctor</option>
                <option value="Admin">Administrator</option>
              </Select>
            </div>
          </div>

          <AnimatePresence>
            {form.role === 'Doctor' && (
              <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }} className="p-8 rounded-[32px] bg-slate-50 border border-slate-100 space-y-6">
                <div className="flex items-center gap-2 text-blue-600 font-black uppercase text-xs tracking-widest"><Stethoscope size={18} /> Medical Certification</div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Input value={form.specialityName} onChange={e => setField('specialityName', e.target.value)} placeholder="Speciality (EN)" className="h-12 rounded-xl" />
                  <Input value={form.specialityNameAr} onChange={e => setField('specialityNameAr', e.target.value)} placeholder="الخصص (AR)" className="h-12 rounded-xl" />
                  <Input type="number" value={form.yearsExperience} onChange={e => setField('yearsExperience', Number(e.target.value))} placeholder="Experience Years" className="h-12 rounded-xl" />
                  <Input type="number" value={form.consultationFee} onChange={e => setField('consultationFee', Number(e.target.value))} placeholder="Fee (EGP)" className="h-12 rounded-xl" />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="flex justify-end gap-3 pt-6">
            <Button variant="outline" onClick={() => setShowAddModal(false)} className="h-14 px-8 rounded-2xl font-bold">Cancel</Button>
            <Button variant="primary" onClick={handleSubmitAddUser} className="h-14 px-10 rounded-2xl font-black shadow-xl shadow-blue-500/20">Finalize Creation</Button>
          </div>
        </div>
      </Modal>
    </motion.div>
  )
}