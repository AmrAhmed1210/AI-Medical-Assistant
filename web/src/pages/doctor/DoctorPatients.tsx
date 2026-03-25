import { useState } from 'react'
import { Search, User } from 'lucide-react'
import { useDoctorPatients } from '@/hooks/useDoctor'
import { Modal } from '@/components/ui/Modal'
import { UrgencyBadge } from '@/components/ui/Badge'
import { Card } from '@/components/ui/Card'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import type { PatientSummaryDto } from '@/lib/types'
import { formatDate } from '@/lib/utils'

export default function DoctorPatients() {
  const [search, setSearch] = useState('')
  const { patients, isLoading } = useDoctorPatients(search)
  const [selected, setSelected] = useState<PatientSummaryDto | null>(null)

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-bold text-gray-800">المرضى</h1>
        <p className="text-sm text-gray-500 mt-0.5">{patients.length} مريض مسجل</p>
      </div>

      <Card padding="none">
        <div className="p-4 border-b border-gray-100">
          <div className="relative max-w-sm">
            <Search size={15} className="absolute top-1/2 -translate-y-1/2 right-3 text-gray-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="بحث باسم المريض..."
              className="w-full pr-9 pl-4 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/30"
            />
          </div>
        </div>

        {isLoading ? <PageLoader /> : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-100">
                  <th className="px-4 py-3 text-right font-semibold text-gray-600">المريض</th>
                  <th className="px-4 py-3 text-right font-semibold text-gray-600">العمر / الجنس</th>
                  <th className="px-4 py-3 text-right font-semibold text-gray-600">آخر زيارة</th>
                  <th className="px-4 py-3 text-right font-semibold text-gray-600">عدد الجلسات</th>
                  <th className="px-4 py-3 text-right font-semibold text-gray-600">مستوى الخطورة</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {patients.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-16 text-center text-gray-400 text-sm">لا توجد نتائج</td>
                  </tr>
                ) : patients.map((p) => (
                  <tr
                    key={p.patientId}
                    onClick={() => setSelected(p)}
                    className="hover:bg-gray-50/80 cursor-pointer transition-colors"
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center">
                          <User size={14} className="text-green-600" />
                        </div>
                        <div>
                          <p className="font-medium text-gray-800">{p.fullName}</p>
                          {p.phone && <p className="text-xs text-gray-400">{p.phone}</p>}
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-gray-600">
                      {p.age ? `${p.age} سنة` : '-'} / {p.gender === 'male' ? 'ذكر' : p.gender === 'female' ? 'أنثى' : '-'}
                    </td>
                    <td className="px-4 py-3 text-gray-600">{p.lastVisit ? formatDate(p.lastVisit) : '-'}</td>
                    <td className="px-4 py-3 text-gray-600">{p.totalSessions}</td>
                    <td className="px-4 py-3"><UrgencyBadge level={p.urgencyTrend} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Modal open={!!selected} onClose={() => setSelected(null)} title="تفاصيل المريض" size="md">
        {selected && (
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center">
                <User size={28} className="text-green-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-800">{selected.fullName}</h3>
                {selected.email && <p className="text-sm text-gray-500">{selected.email}</p>}
                {selected.phone && <p className="text-sm text-gray-500">{selected.phone}</p>}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 border-t border-gray-100 pt-4">
              <div><p className="text-xs text-gray-400">العمر</p><p className="font-medium">{selected.age ? `${selected.age} سنة` : 'غير محدد'}</p></div>
              <div><p className="text-xs text-gray-400">الجنس</p><p className="font-medium">{selected.gender === 'male' ? 'ذكر' : selected.gender === 'female' ? 'أنثى' : 'غير محدد'}</p></div>
              <div><p className="text-xs text-gray-400">إجمالي الجلسات</p><p className="font-medium">{selected.totalSessions}</p></div>
              <div><p className="text-xs text-gray-400">آخر زيارة</p><p className="font-medium">{selected.lastVisit ? formatDate(selected.lastVisit) : 'لا توجد'}</p></div>
              <div><p className="text-xs text-gray-400 mb-1">مستوى الخطورة</p><UrgencyBadge level={selected.urgencyTrend} /></div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}
