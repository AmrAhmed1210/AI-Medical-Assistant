import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { AvailabilityDto, DayOfWeek } from '@/lib/types'
import { DAY_NAMES_AR } from '@/lib/utils'
import { Button } from '@/components/ui/Button'
import { Clock, Save } from 'lucide-react'
import { cn } from '@/lib/utils'

interface AvailabilityEditorProps {
  availability: AvailabilityDto[]
  onSave: (data: AvailabilityDto[]) => Promise<void>
  isSaving?: boolean
}

const DEFAULT_AVAILABILITY: AvailabilityDto[] = DAY_NAMES_AR.map((dayName, i) => ({
  dayOfWeek: i as DayOfWeek,
  dayName,
  startTime: '09:00',
  endTime: '17:00',
  slotDurationMinutes: 30,
  isAvailable: false, // Inactive by default
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
    <div className="space-y-4 p-2">
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="flex items-center gap-2 p-3 bg-gradient-to-r from-blue-50 to-cyan-50 rounded-xl border border-blue-200"
      >
        <Clock size={16} className="text-blue-600" />
        <p className="text-xs text-gray-600"><strong>Tip:</strong> Toggle the checkbox to enable/disable a day, then set your available time slots</p>
      </motion.div>

      <div className="space-y-3">
        {slots.map((slot, idx) => (
          <motion.div
            key={slot.dayOfWeek}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.05 }}
            className={cn(
              'flex flex-wrap items-center gap-4 p-4 rounded-xl border-2 transition-all shadow-sm',
              slot.isAvailable
                ? 'border-primary-300 bg-gradient-to-r from-primary-50 to-blue-50'
                : 'border-gray-200 bg-gray-50 opacity-50'
            )}
          >
            <div className="flex items-center gap-3 w-32">
              <motion.label
                className="relative inline-flex items-center cursor-pointer"
                whileHover={{ scale: 1.05 }}
              >
                <input
                  type="checkbox"
                  checked={slot.isAvailable}
                  onChange={(e) => update(slot.dayOfWeek, 'isAvailable', e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-gray-300 peer-focus:ring-2 peer-focus:ring-primary-300 rounded-full peer peer-checked:bg-gradient-to-r peer-checked:from-primary-600 peer-checked:to-primary-500 after:content-[''] after:absolute after:top-0.5 after:right-0.5 after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:after:translate-x-[-20px] shadow-sm" />
              </motion.label>
              <span className={cn('text-sm font-semibold', slot.isAvailable ? 'text-gray-800' : 'text-gray-500')}>
                {slot.dayName}
              </span>
            </div>

            <AnimatePresence>
              {slot.isAvailable && (
                <>
                  <motion.div
                    className="flex items-center gap-2"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                  >
                    <label className="text-xs font-medium text-gray-600 whitespace-nowrap">From</label>
                    <input
                      type="time"
                      value={slot.startTime}
                      onChange={(e) => update(slot.dayOfWeek, 'startTime', e.target.value)}
                      className="text-sm border-2 border-primary-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-400/50 focus:border-primary-400 bg-white font-medium"
                    />
                  </motion.div>

                  <motion.div
                    className="flex items-center gap-2"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ delay: 0.05 }}
                  >
                    <label className="text-xs font-medium text-gray-600 whitespace-nowrap">To</label>
                    <input
                      type="time"
                      value={slot.endTime}
                      onChange={(e) => update(slot.dayOfWeek, 'endTime', e.target.value)}
                      className="text-sm border-2 border-primary-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-400/50 focus:border-primary-400 bg-white font-medium"
                    />
                  </motion.div>

                  <motion.div
                    className="flex items-center gap-2"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    transition={{ delay: 0.1 }}
                  >
                    <label className="text-xs font-medium text-gray-600 whitespace-nowrap">Slot Duration</label>
                    <select
                      value={slot.slotDurationMinutes}
                      onChange={(e) => update(slot.dayOfWeek, 'slotDurationMinutes', Number(e.target.value))}
                      className="text-sm border-2 border-primary-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary-400/50 focus:border-primary-400 bg-white font-medium"
                    >
                      <option value={15}>15 min</option>
                      <option value={20}>20 min</option>
                      <option value={30}>30 min</option>
                      <option value={45}>45 min</option>
                      <option value={60}>60 min</option>
                    </select>
                  </motion.div>
                </>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>

      <motion.div
        className="flex justify-end pt-4"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Button
          variant="primary"
          onClick={() => onSave(slots)}
          loading={isSaving}
          icon={<Save size={16} />}
        >
          Save Schedule
        </Button>
      </motion.div>
    </div>
  )
}
