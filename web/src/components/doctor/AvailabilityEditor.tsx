import { useState } from 'react'
import type { AvailabilityDto } from '@/lib/types'
import { DAY_NAMES_AR } from '@/lib/utils'
import { Button } from '@/components/ui/Button'
import { Save } from 'lucide-react'
import { cn } from '@/lib/utils'

interface AvailabilityEditorProps {
  availability: AvailabilityDto[]
  onSave: (data: AvailabilityDto[]) => Promise<void>
  isSaving?: boolean
}

const DEFAULT_AVAILABILITY: AvailabilityDto[] = DAY_NAMES_AR.map((dayName, i) => ({
  dayOfWeek: i,
  dayName,
  startTime: '09:00',
  endTime: '17:00',
  slotDurationMinutes: 30,
  isActive: i >= 0 && i <= 4, // Sun-Thu active by default
}))

export function AvailabilityEditor({ availability, onSave, isSaving }: AvailabilityEditorProps) {
  const [slots, setSlots] = useState<AvailabilityDto[]>(
    availability.length > 0 ? availability : DEFAULT_AVAILABILITY
  )

  const update = (dayOfWeek: number, field: keyof AvailabilityDto, value: unknown) => {
    setSlots((prev) =>
      prev.map((s) => (s.dayOfWeek === dayOfWeek ? { ...s, [field]: value } : s))
    )
  }

  return (
    <div className="space-y-3">
      {slots.map((slot) => (
        <div
          key={slot.dayOfWeek}
          className={cn(
            'flex flex-wrap items-center gap-4 p-4 rounded-xl border transition-colors',
            slot.isActive ? 'border-primary-100 bg-primary-50/30' : 'border-gray-100 bg-gray-50/50 opacity-60'
          )}
        >
          <div className="flex items-center gap-3 w-28">
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={slot.isActive}
                onChange={(e) => update(slot.dayOfWeek, 'isActive', e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-9 h-5 bg-gray-200 peer-focus:ring-2 peer-focus:ring-primary-300 rounded-full peer peer-checked:bg-primary-600 after:content-[''] after:absolute after:top-0.5 after:right-0.5 after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:after:translate-x-[-16px]" />
            </label>
            <span className="text-sm font-medium text-gray-700">{slot.dayName}</span>
          </div>

          {slot.isActive && (
            <>
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-500">من</label>
                <input
                  type="time"
                  value={slot.startTime}
                  onChange={(e) => update(slot.dayOfWeek, 'startTime', e.target.value)}
                  className="text-sm border border-gray-200 rounded-lg px-2 py-1 focus:outline-none focus:ring-2 focus:ring-primary-400/30"
                />
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-500">إلى</label>
                <input
                  type="time"
                  value={slot.endTime}
                  onChange={(e) => update(slot.dayOfWeek, 'endTime', e.target.value)}
                  className="text-sm border border-gray-200 rounded-lg px-2 py-1 focus:outline-none focus:ring-2 focus:ring-primary-400/30"
                />
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-500">مدة الحجز</label>
                <select
                  value={slot.slotDurationMinutes}
                  onChange={(e) => update(slot.dayOfWeek, 'slotDurationMinutes', Number(e.target.value))}
                  className="text-sm border border-gray-200 rounded-lg px-2 py-1 focus:outline-none focus:ring-2 focus:ring-primary-400/30"
                >
                  <option value={15}>15 دقيقة</option>
                  <option value={20}>20 دقيقة</option>
                  <option value={30}>30 دقيقة</option>
                  <option value={45}>45 دقيقة</option>
                  <option value={60}>60 دقيقة</option>
                </select>
              </div>
            </>
          )}
        </div>
      ))}

      <div className="flex justify-end pt-2">
        <Button
          variant="primary"
          onClick={() => onSave(slots)}
          loading={isSaving}
          icon={<Save size={15} />}
        >
          حفظ الجدول
        </Button>
      </div>
    </div>
  )
}
