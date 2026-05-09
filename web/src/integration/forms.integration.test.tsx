import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { Input } from '@/components/ui/Input'
import { Select } from '@/components/ui/Select'
import { Button } from '@/components/ui/Button'

describe('Form Components Integration', () => {
  describe('Input + Select Integration', () => {
    it('should render input with label', () => {
      render(
        <div>
          <label>Email</label>
          <Input placeholder="Enter email" />
        </div>
      )
      expect(screen.getByPlaceholderText('Enter email')).toBeInTheDocument()
    })

    it('should show validation error for input', () => {
      render(<Input error="Email is required" placeholder="Enter email" />)
      expect(screen.getByText('Email is required')).toBeInTheDocument()
    })

    it('should render select with options', () => {
      render(
        <Select>
          <option value="doctor">Doctor</option>
          <option value="patient">Patient</option>
        </Select>
      )
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })

    it('should show validation error for select', () => {
      render(
        <Select error="Please select a role">
          <option value="">Select role</option>
          <option value="doctor">Doctor</option>
        </Select>
      )
      expect(screen.getByText('Please select a role')).toBeInTheDocument()
    })
  })

  describe('Button States', () => {
    it('should render enabled button', () => {
      render(<Button>Submit</Button>)
      expect(screen.getByRole('button')).toBeEnabled()
    })

    it('should render loading button', () => {
      render(<Button loading>Submit</Button>)
      expect(screen.getByRole('button')).toBeDisabled()
    })

    it('should render disabled button', () => {
      render(<Button disabled>Submit</Button>)
      expect(screen.getByRole('button')).toBeDisabled()
    })
  })

  describe('Form Validation', () => {
    it('should validate email format', () => {
      const isValidEmail = (email: string) => {
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)
      }
      
      expect(isValidEmail('test@test.com')).toBe(true)
      expect(isValidEmail('invalid-email')).toBe(false)
      expect(isValidEmail('')).toBe(false)
    })

    it('should validate required fields', () => {
      const isRequired = (value: string) => value.trim().length > 0
      
      expect(isRequired('test')).toBe(true)
      expect(isRequired('')).toBe(false)
      expect(isRequired('   ')).toBe(false)
    })
  })
})