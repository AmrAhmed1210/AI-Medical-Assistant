import { describe, it, expect } from 'vitest'
import { formatDate, formatCurrency, getInitials, generateId, cn } from '@/lib/utils'

describe('formatDate', () => {
  it('formats date string correctly', () => {
    const result = formatDate('2024-01-15')
    expect(result).toContain('15')
  })

  it('returns original string for invalid date', () => {
    const result = formatDate('invalid-date')
    expect(result).toBe('invalid-date')
  })

  it('returns empty string for empty input', () => {
    expect(formatDate('')).toBe('')
  })
})

describe('formatCurrency', () => {
  it('formats number as USD', () => {
    const result = formatCurrency(100)
    expect(result).toContain('$')
  })
})

describe('getInitials', () => {
  it('returns first letter of each word', () => {
    expect(getInitials('John Doe')).toBe('JD')
  })

  it('returns max 2 letters', () => {
    expect(getInitials('John Doe Smith')).toBe('JD')
  })
})

describe('generateId', () => {
  it('generates random string of length 7', () => {
    const id = generateId()
    expect(id.length).toBe(7)
  })

  it('generates unique ids', () => {
    const ids = new Set(Array.from({ length: 100 }, () => generateId()))
    expect(ids.size).toBe(100)
  })
})

describe('cn (classnames)', () => {
  it('merges classnames correctly', () => {
    const result = cn('base-class', 'additional-class')
    expect(result).toContain('base-class')
    expect(result).toContain('additional-class')
  })

  it('handles conditional classes', () => {
    const result = cn('base', false && 'hidden', 'visible')
    expect(result).toContain('base')
    expect(result).toContain('visible')
  })
})