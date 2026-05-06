import { useEffect, useMemo, useState } from 'react'
import { useAppointmentStore } from '@/store/appointmentStore'
import { appointmentApi } from '@/api/appointmentApi'
import { secretaryApi } from '@/api/secretaryApi'
import { AppointmentTable } from '@/components/doctor/AppointmentTable'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Modal } from '@/components/ui/Modal'
import type { AppointmentStatus, AppointmentDto, DoctorDetailDto, PatientSummaryDto } from '@/lib/types'
import toast from 'react-hot-toast'
import { RefreshCw, Calendar, Search, ClipboardList, Plus, Loader2, Settings } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

const STATUS_FILTERS: { label: string; value: AppointmentStatus | '' }[] = [
  { label: 'All', value: '' },
  { label: 'Pending', value: 'Pending' },
  { label: 'Confirmed', value: 'Confirmed' },
  { label: 'Completed', value: 'Completed' },
  { label: 'Cancelled', value: 'Cancelled' },
]

export default function SecretaryDashboard() {
  const navigate = useNavigate()
  const { appointments, isLoading, fetchSecretaryAppointments, confirm, cancel, updateLocal } = useAppointmentStore()

  const [activeTab, setActiveTab] = useState<'active' | 'history'>('active')
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<AppointmentStatus | ''>('')
  const [selectedDayKey, setSelectedDayKey] = useState('')

  // Reschedule modal
  const [rescheduleOpen, setRescheduleOpen] = useState(false)
  const [rescheduleAppt, setRescheduleAppt] = useState<AppointmentDto | null>(null)
  const [rescheduleDate, setRescheduleDate] = useState('')
  const [rescheduleTime, setRescheduleTime] = useState('')
  const [rescheduleReason, setRescheduleReason] = useState('')
  const [rescheduleLoading, setRescheduleLoading] = useState(false)

  // Book modal
  const [bookOpen, setBookOpen] = useState(false)
  const [myDoctor, setMyDoctor] = useState<DoctorDetailDto | null>(null)
  const [patientMode, setPatientMode] = useState<'existing' | 'new'>('existing')
  const [patientSearchQuery, setPatientSearchQuery] = useState('')
  const [patientResults, setPatientResults] = useState<PatientSummaryDto[]>([])
  const [selectedPatient, setSelectedPatient] = useState<PatientSummaryDto | null>(null)
  const [showPatientDropdown, setShowPatientDropdown] = useState(false)
  const [patientSearchLoading, setPatientSearchLoading] = useState(false)
  const [newPatientName, setNewPatientName] = useState('')
  const [newPatientEmail, setNewPatientEmail] = useState('')
  const [newPatientPhone, setNewPatientPhone] = useState('')
  const [bookDate, setBookDate] = useState('')
  const [bookTime, setBookTime] = useState('')
  const [bookPayment, setBookPayment] = useState('cash')
  const [bookNotes, setBookNotes] = useState('')
  const [bookLoading, setBookLoading] = useState(false)

  useEffect(() => {
    fetchSecretaryAppointments()
  }, [])

  useEffect(() => {
    secretaryApi.getMyDoctor()
      .then(setMyDoctor)
      .catch(() => toast.error('Failed to load doctor info'))
  }, [])

  useEffect(() => {
    if (!bookOpen) {
      setPatientSearchQuery('')
      setPatientResults([])
      setSelectedPatient(null)
      setShowPatientDropdown(false)
      setPatientMode('existing')
      setNewPatientName('')
      setNewPatientEmail('')
      setNewPatientPhone('')
      return
    }
    // Book modal opened; doctor already loaded
  }, [bookOpen])

  useEffect(() => {
    if (!patientSearchQuery.trim() || !bookOpen) {
      setPatientResults([])
      setShowPatientDropdown(false)
      return
    }
    const timer = setTimeout(() => {
      setPatientSearchLoading(true)
      secretaryApi.searchMyDoctorPatients(patientSearchQuery)
        .then((patients) => {
          setPatientResults(patients)
          setShowPatientDropdown(true)
        })
        .catch(() => toast.error('Failed to search patients'))
        .finally(() => setPatientSearchLoading(false))
    }, 300)
    return () => clearTimeout(timer)
  }, [patientSearchQuery, bookOpen])

  const handleConfirm = async (id: string) => {
    try {
      await confirm(id)
      toast.success('Appointment confirmed')
    } catch { toast.error('Failed to confirm appointment') }
  }

  const handleUnconfirm = async (id: string) => {
    try {
      await appointmentApi.setPending(id)
      updateLocal(id, { status: 'Pending' })
      toast.success('Appointment moved to pending')
    } catch { toast.error('Failed to unconfirm appointment') }
  }

  const handleCancel = async (id: string) => {
    try {
      await cancel(id)
      toast.success('Appointment cancelled')
    } catch { toast.error('Failed to cancel appointment') }
  }

  const handleDelete = async (id: string) => {
    try {
      await appointmentApi.delete(id)
      updateLocal(id, { status: 'Cancelled' })
      toast.success('Appointment deleted')
    } catch { toast.error('Failed to delete appointment') }
  }

  const handleNoShow = async (id: string) => {
    try {
      await appointmentApi.noShow(id)
      updateLocal(id, { status: 'NoShow' })
      toast.success('Marked as no-show')
    } catch { toast.error('Failed to mark no-show') }
  }

  const openReschedule = (appt: AppointmentDto) => {
    setRescheduleAppt(appt)
    const d = new Date(appt.scheduledAt)
    setRescheduleDate(d.toISOString().split('T')[0])
    setRescheduleTime(d.toTimeString().slice(0, 5))
    setRescheduleReason('')
    setRescheduleOpen(true)
  }

  const handleRescheduleSubmit = async () => {
    if (!rescheduleAppt) return
    if (!rescheduleDate || !rescheduleTime) {
      toast.error('Please select date and time')
      return
    }
    try {
      setRescheduleLoading(true)
      await appointmentApi.reschedule(rescheduleAppt.id, rescheduleDate, rescheduleTime, rescheduleReason)
      toast.success('Appointment rescheduled')
      await fetchSecretaryAppointments()
      setRescheduleOpen(false)
    } catch {
      toast.error('Failed to reschedule')
    } finally {
      setRescheduleLoading(false)
    }
  }

  const handleBookSubmit = async () => {
    if (!myDoctor || !bookDate || !bookTime) {
      toast.error('Please fill all required fields')
      return
    }
    try {
      setBookLoading(true)
      let patientId: number
      if (patientMode === 'existing') {
        if (!selectedPatient) {
          toast.error('Please select a patient')
          return
        }
        patientId = Number(selectedPatient.id)
        console.log('Booking for existing patient:', { patientId, doctorId: myDoctor.id, date: bookDate, time: bookTime })
      } else {
        if (!newPatientName || !newPatientPhone) {
          toast.error('Please fill patient name and phone')
          return
        }
        console.log('Creating walk-in patient:', { fullName: newPatientName, email: newPatientEmail, phoneNumber: newPatientPhone })
        const newPatient = await secretaryApi.createWalkInPatient({
          fullName: newPatientName,
          email: newPatientEmail,
          phoneNumber: newPatientPhone,
        })
        patientId = newPatient.id
        console.log('Created new patient:', newPatient)
      }
      const appointmentData = {
        doctorId: String(myDoctor.id),
        patientId,
        date: bookDate,
        time: bookTime,
        paymentMethod: bookPayment,
        notes: bookNotes,
        scheduledAt: `${bookDate}T${bookTime}`,
      }
      console.log('Creating appointment with data:', appointmentData)
      console.log('Raw bookTime value:', bookTime)
      console.log('ScheduledAt string:', `${bookDate}T${bookTime}`)
      const result = await appointmentApi.create(appointmentData)
      console.log('Appointment created:', result)
      toast.success('Appointment booked successfully')
      await fetchSecretaryAppointments()
      console.log('Appointments after booking:', appointments)
      setBookOpen(false)
      setPatientSearchQuery('')
      setSelectedPatient(null)
      setPatientResults([])
      setNewPatientName('')
      setNewPatientEmail('')
      setNewPatientPhone('')
      setBookDate('')
      setBookTime('')
      setBookPayment('cash')
      setBookNotes('')
    } catch (error: any) {
      console.error('Failed to book appointment:', error)
      if (error.response?.data) {
        console.error('Backend error:', error.response.data)
        toast.error(`Failed to book: ${error.response.data.message || 'Unknown error'}`)
      } else {
        toast.error('Failed to book appointment')
      }
    } finally {
      setBookLoading(false)
    }
  }

  const toDayKey = (value: string | Date) => {
    const date = value instanceof Date ? value : new Date(value)
    if (Number.isNaN(date.getTime())) return 'unknown'
    const year = date.getFullYear()
    const month = `${date.getMonth() + 1}`.padStart(2, '0')
    const day = `${date.getDate()}`.padStart(2, '0')
    return `${year}-${month}-${day}`
  }

  const displayAppointments = useMemo(() => {
    let filtered = appointments
    if (activeTab === 'active') {
      filtered = filtered.filter(a =>
        a.status === 'Pending' || a.status === 'Confirmed'
      )
      if (searchQuery) {
        const q = searchQuery.toLowerCase()
        filtered = filtered.filter(a =>
          a.patientName?.toLowerCase().includes(q) ||
          a.scheduledAt?.toLowerCase().includes(q) ||
          a.status?.toLowerCase().includes(q)
        )
      }
    } else {
      filtered = filtered.filter(a =>
        a.status === 'Completed' || a.status === 'Cancelled'
      )
    }

    if (statusFilter) {
      filtered = filtered.filter(a => a.status === statusFilter)
    }

    return filtered
  }, [appointments, activeTab, searchQuery, statusFilter])

  const daySections = useMemo(() => {
    const grouped = displayAppointments.reduce<Record<string, AppointmentDto[]>>((acc, appt) => {
      const key = toDayKey(appt.scheduledAt)
      if (!acc[key]) acc[key] = []
      acc[key].push(appt)
      return acc
    }, {})

    return Object.entries(grouped)
      .sort(([a], [b]) => (a > b ? 1 : -1))
      .map(([key, dayAppointments]) => ({
        key,
        label: key === 'unknown'
          ? 'No Date'
          : new Date(`${key}T00:00:00`).toLocaleDateString(undefined, {
            weekday: 'short',
            day: '2-digit',
            month: 'short',
            year: 'numeric',
          }),
        appointments: dayAppointments,
      }))
  }, [displayAppointments])

  useEffect(() => {
    if (daySections.length === 0) {
      setSelectedDayKey('')
      return
    }
    const today = toDayKey(new Date())
    const hasToday = daySections.some((section) => section.key === today)
    const nextSelected = hasToday ? today : daySections[0].key
    if (!selectedDayKey || !daySections.some((section) => section.key === selectedDayKey)) {
      setSelectedDayKey(nextSelected)
    }
  }, [daySections, selectedDayKey])

  const selectedDayAppointments = daySections.find((section) => section.key === selectedDayKey)?.appointments ?? []

  return (
    <div className="space-y-5 p-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative bg-gradient-to-br from-blue-600 to-blue-500 rounded-xl p-3 shadow-lg">
            <Calendar size={28} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-800">Secretary Portal</h1>
            <p className="text-sm text-gray-500 mt-0.5">Manage doctor's appointments</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Button variant="outline" icon={<Settings size={14} />} onClick={() => navigate('/secretary/schedule')} size="sm">
            Doctor Schedule
          </Button>
          <Button variant="outline" icon={<RefreshCw size={14} />} onClick={fetchSecretaryAppointments} size="sm">
            Refresh
          </Button>
          <Button icon={<Plus size={14} />} onClick={() => setBookOpen(true)} size="sm">
            Book Appointment
          </Button>
        </div>
      </div>

      <Card>
        <div className="flex flex-col sm:flex-row border-b border-gray-100">
          <div className="flex">
            <button
              onClick={() => setActiveTab('active')}
              className={`px-6 py-4 text-sm font-medium transition-colors border-b-2 ${
                activeTab === 'active'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Active
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`px-6 py-4 text-sm font-medium transition-colors border-b-2 ${
                activeTab === 'history'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              History
            </button>
          </div>

          <div className="flex-1 p-2 flex items-center justify-end gap-3 sm:border-l border-gray-100 bg-gray-50/50">
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search patient..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8 pr-4 py-1.5 text-sm border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none w-64 bg-white"
              />
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 p-4 border-b border-gray-100 flex-wrap bg-white">
          {STATUS_FILTERS.map((f) => (
            <button
              key={f.value}
              onClick={() => setStatusFilter(f.value)}
              className={`px-3 py-1.5 text-xs rounded-xl font-medium transition-colors ${statusFilter === f.value
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
            >
              {f.label}
            </button>
          ))}
        </div>

        {daySections.length === 0 ? (
          <div className="py-20 text-center">
            {isLoading ? <p>Loading appointments...</p> : <p className="text-gray-400">No appointments found</p>}
          </div>
        ) : (
          <div className="p-4 space-y-4">
            <div className="flex items-center gap-2 flex-wrap">
              {daySections.map((section) => {
                const isSelected = selectedDayKey === section.key
                return (
                  <button
                    key={section.key}
                    onClick={() => setSelectedDayKey(section.key)}
                    className={`px-4 py-2 text-xs rounded-xl font-semibold transition-all duration-200 ${
                      isSelected ? 'bg-primary-600 text-white shadow-lg shadow-primary-500/25' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {section.label}
                    <span className="ml-2 text-[10px] opacity-70">({section.appointments.length})</span>
                  </button>
                )
              })}
            </div>

            <div className="rounded-2xl border border-gray-100 overflow-hidden bg-white">
              <AppointmentTable
                appointments={selectedDayAppointments}
                onConfirm={handleConfirm}
                onUnconfirm={handleUnconfirm}
                onCancel={handleCancel}
                onDelete={handleDelete}
                onNoShow={handleNoShow}
                onReschedule={(id) => {
                  const appt = selectedDayAppointments.find(a => a.id === id)
                  if (appt) openReschedule(appt)
                }}
                showSecretaryActions
              />
            </div>
          </div>
        )}
      </Card>

      {/* Reschedule Modal */}
      <Modal
        open={rescheduleOpen}
        onClose={() => setRescheduleOpen(false)}
        title="Reschedule Appointment"
        footer={
          <>
            <Button variant="outline" onClick={() => setRescheduleOpen(false)} disabled={rescheduleLoading}>Cancel</Button>
            <Button onClick={handleRescheduleSubmit} disabled={rescheduleLoading}>
              {rescheduleLoading && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
              Reschedule
            </Button>
          </>
        }
      >
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Patient</label>
            <p className="text-sm text-gray-500">{rescheduleAppt?.patientName}</p>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">New Date</label>
            <input
              type="date"
              value={rescheduleDate}
              onChange={(e) => setRescheduleDate(e.target.value)}
              className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">New Time</label>
            <input
              type="time"
              value={rescheduleTime}
              onChange={(e) => setRescheduleTime(e.target.value)}
              className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Reason (optional)</label>
            <input
              type="text"
              value={rescheduleReason}
              onChange={(e) => setRescheduleReason(e.target.value)}
              placeholder="e.g. Patient request"
              className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 outline-none text-sm"
            />
          </div>
        </div>
      </Modal>

      {/* Book Appointment Modal */}
      <Modal
        open={bookOpen}
        onClose={() => setBookOpen(false)}
        title="Book New Appointment"
        size="lg"
        footer={
          <>
            <Button variant="outline" onClick={() => setBookOpen(false)} disabled={bookLoading}>Cancel</Button>
            <Button onClick={handleBookSubmit} disabled={bookLoading}>
              {bookLoading && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
              Book Appointment
            </Button>
          </>
        }
      >
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Patient Type</label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setPatientMode('existing')}
                className={`flex-1 px-3 py-2 text-sm rounded-lg border transition-colors ${
                  patientMode === 'existing'
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
                }`}
              >
                Existing Patient
              </button>
              <button
                type="button"
                onClick={() => setPatientMode('new')}
                className={`flex-1 px-3 py-2 text-sm rounded-lg border transition-colors ${
                  patientMode === 'new'
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
                }`}
              >
                New Walk-in
              </button>
            </div>
          </div>

          {patientMode === 'existing' ? (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Search Patient *</label>
              <div className="relative">
                <input
                  type="text"
                  value={selectedPatient ? selectedPatient.fullName : patientSearchQuery}
                  onChange={(e) => {
                    if (selectedPatient) {
                      setSelectedPatient(null)
                    }
                    setPatientSearchQuery(e.target.value)
                  }}
                  onFocus={() => {
                    if (patientResults.length > 0) setShowPatientDropdown(true)
                  }}
                  placeholder="Search patient by name..."
                  className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none text-sm"
                />
                {patientSearchLoading && (
                  <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 animate-spin text-gray-400" />
                )}
                {showPatientDropdown && patientResults.length > 0 && (
                  <div className="absolute z-[9999] mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-48 overflow-auto">
                    {patientResults.map((patient) => (
                      <button
                        key={patient.id}
                        type="button"
                        onClick={() => {
                          setSelectedPatient(patient)
                          setShowPatientDropdown(false)
                          setPatientSearchQuery(patient.fullName)
                        }}
                        className="w-full text-left px-3 py-2 hover:bg-gray-50 text-sm border-b border-gray-50 last:border-0"
                      >
                        <span className="font-medium">{patient.fullName}</span>
                        <span className="text-gray-400 text-xs ml-2">{patient.email}</span>
                      </button>
                    ))}
                  </div>
                )}
                {showPatientDropdown && !patientSearchLoading && patientSearchQuery.trim() && patientResults.length === 0 && (
                  <div className="absolute z-[9999] mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg px-3 py-2 text-sm text-gray-400">
                    No patients found
                  </div>
                )}
              </div>
            </div>
          ) : (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Full Name *</label>
                <input
                  type="text"
                  value={newPatientName}
                  onChange={(e) => setNewPatientName(e.target.value)}
                  placeholder="Enter patient name"
                  className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none text-sm"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email *</label>
                <input
                  type="email"
                  value={newPatientEmail}
                  onChange={(e) => setNewPatientEmail(e.target.value)}
                  placeholder="Enter patient email"
                  className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none text-sm"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Phone *</label>
                <input
                  type="tel"
                  value={newPatientPhone}
                  onChange={(e) => setNewPatientPhone(e.target.value)}
                  placeholder="Enter patient phone"
                  className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none text-sm"
                />
              </div>
            </>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Doctor</label>
            <div className="px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm text-gray-700">
              {myDoctor ? `${myDoctor.fullName} – ${myDoctor.specialty}` : 'Loading doctor...'}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Date *</label>
              <input
                type="date"
                value={bookDate}
                onChange={(e) => setBookDate(e.target.value)}
                className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none text-sm"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Time *</label>
              <input
                type="time"
                value={bookTime}
                onChange={(e) => {
                  console.log('Time input changed to:', e.target.value)
                  setBookTime(e.target.value)
                }}
                className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none text-sm"
              />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Payment Method</label>
            <select
              value={bookPayment}
              onChange={(e) => setBookPayment(e.target.value)}
              className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none text-sm bg-white"
            >
              <option value="cash">Cash on Arrival</option>
              <option value="visa">Visa (Online)</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Notes</label>
            <textarea
              value={bookNotes}
              onChange={(e) => setBookNotes(e.target.value)}
              placeholder="Optional notes..."
              rows={3}
              className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none text-sm resize-none"
            />
          </div>
        </div>
      </Modal>
    </div>
  )
}
