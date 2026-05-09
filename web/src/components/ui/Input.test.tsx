import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Input } from '@/components/ui/Input'

describe('Input Component', () => {
  it('renders input element', () => {
    render(<Input />)
    expect(screen.getByRole('textbox')).toBeInTheDocument()
  })

  it('accepts placeholder text', () => {
    render(<Input placeholder="Enter name" />)
    expect(screen.getByPlaceholderText('Enter name')).toBeInTheDocument()
  })

  it('accepts custom className', () => {
    const { container } = render(<Input className="custom-input" />)
    expect(screen.getByRole('textbox')).toHaveClass('custom-input')
  })

  it('shows error message when error prop is provided', () => {
    render(<Input error="This field is required" />)
    expect(screen.getByText('This field is required')).toBeInTheDocument()
  })

  it('applies error styling when error is present', () => {
    const { container } = render(<Input error="Error" />)
    expect(screen.getByRole('textbox')).toHaveClass('border-red-300')
    expect(screen.getByRole('textbox')).toHaveClass('bg-red-50')
  })

  it('applies normal styling when no error', () => {
    const { container } = render(<Input />)
    expect(screen.getByRole('textbox')).toHaveClass('border-gray-200')
    expect(screen.getByRole('textbox')).toHaveClass('bg-white')
  })

  it('renders with icon', () => {
    const icon = <span data-testid="input-icon">🔔</span>
    render(<Input icon={icon} />)
    expect(screen.getByTestId('input-icon')).toBeInTheDocument()
  })

  it('applies icon styling when icon is present', () => {
    const icon = <span>🔔</span>
    render(<Input icon={icon} />)
    expect(screen.getByRole('textbox')).toHaveClass('pr-10')
  })

  it('handles typed input', () => {
    render(<Input />)
    const input = screen.getByRole('textbox')
    input.focus()
  })
})