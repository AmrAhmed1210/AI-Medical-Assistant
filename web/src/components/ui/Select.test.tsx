import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Select } from '@/components/ui/Select'

describe('Select Component', () => {
  it('renders select element', () => {
    render(<Select><option>Option 1</option></Select>)
    expect(screen.getByRole('combobox')).toBeInTheDocument()
  })

  it('renders options', () => {
    render(
      <Select>
        <option value="1">Option 1</option>
        <option value="2">Option 2</option>
      </Select>
    )
    expect(screen.getByText('Option 1')).toBeInTheDocument()
    expect(screen.getByText('Option 2')).toBeInTheDocument()
  })

  it('accepts custom className', () => {
    const { container } = render(
      <Select className="custom-select">
        <option>Option</option>
      </Select>
    )
    expect(screen.getByRole('combobox')).toHaveClass('custom-select')
  })

  it('shows error message when error prop is provided', () => {
    render(<Select error="Select is required"><option>Option</option></Select>)
    expect(screen.getByText('Select is required')).toBeInTheDocument()
  })

  it('applies error styling when error is present', () => {
    render(
      <Select error="Error">
        <option>Option</option>
      </Select>
    )
    expect(screen.getByRole('combobox')).toHaveClass('border-red-300')
  })

  it('applies normal styling when no error', () => {
    render(<Select><option>Option</option></Select>)
    expect(screen.getByRole('combobox')).toHaveClass('border-gray-200')
  })
})