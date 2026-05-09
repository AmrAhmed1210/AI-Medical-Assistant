import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Badge, StatusBadge, UrgencyBadge } from '@/components/ui/Badge'

describe('Badge Component', () => {
  it('renders with children', () => {
    render(<Badge>Test Badge</Badge>)
    expect(screen.getByText('Test Badge')).toBeInTheDocument()
  })

  it('renders with default variant', () => {
    const { container } = render(<Badge>Default</Badge>)
    expect(container.firstChild).toHaveClass('bg-gray-100/80')
  })

  it('renders with success variant', () => {
    const { container } = render(<Badge variant="success">Success</Badge>)
    expect(container.firstChild).toHaveClass('bg-gradient-to-r')
    expect(container.firstChild).toHaveClass('from-green-400')
  })

  it('renders with warning variant', () => {
    const { container } = render(<Badge variant="warning">Warning</Badge>)
    expect(container.firstChild).toHaveClass('bg-gradient-to-r')
    expect(container.firstChild).toHaveClass('from-amber-400')
  })

  it('renders with danger variant', () => {
    const { container } = render(<Badge variant="danger">Danger</Badge>)
    expect(container.firstChild).toHaveClass('bg-gradient-to-r')
    expect(container.firstChild).toHaveClass('from-red-400')
  })

  it('renders with info variant', () => {
    const { container } = render(<Badge variant="info">Info</Badge>)
    expect(container.firstChild).toHaveClass('bg-gradient-to-r')
    expect(container.firstChild).toHaveClass('from-blue-400')
  })

  it('renders with admin variant', () => {
    const { container } = render(<Badge variant="admin">Admin</Badge>)
    expect(container.firstChild).toHaveClass('bg-gradient-to-r')
    expect(container.firstChild).toHaveClass('from-violet-500')
  })

  it('applies custom className', () => {
    const { container } = render(<Badge className="custom-class">Custom</Badge>)
    expect(container.firstChild).toHaveClass('custom-class')
  })
})

describe('StatusBadge Component', () => {
  it('renders Pending status', () => {
    render(<StatusBadge status="Pending" />)
    expect(screen.getByText('Pending')).toBeInTheDocument()
  })

  it('renders Confirmed status', () => {
    render(<StatusBadge status="Confirmed" />)
    expect(screen.getByText('Confirmed')).toBeInTheDocument()
  })

  it('renders Completed status', () => {
    render(<StatusBadge status="Completed" />)
    expect(screen.getByText('Completed')).toBeInTheDocument()
  })

  it('renders Cancelled status', () => {
    render(<StatusBadge status="Cancelled" />)
    expect(screen.getByText('Cancelled')).toBeInTheDocument()
  })

  it('renders NoShow status', () => {
    render(<StatusBadge status="NoShow" />)
    expect(screen.getByText('No Show')).toBeInTheDocument()
  })

  it('renders Rescheduled status', () => {
    render(<StatusBadge status="Rescheduled" />)
    expect(screen.getByText('Rescheduled')).toBeInTheDocument()
  })

  it('renders unknown status with default badge', () => {
    render(<StatusBadge status="UnknownStatus" />)
    expect(screen.getByText('UnknownStatus')).toBeInTheDocument()
  })
})

describe('UrgencyBadge Component', () => {
  it('renders LOW level', () => {
    render(<UrgencyBadge level="LOW" />)
    expect(screen.getByText('Low')).toBeInTheDocument()
  })

  it('renders MEDIUM level', () => {
    render(<UrgencyBadge level="MEDIUM" />)
    expect(screen.getByText('Medium')).toBeInTheDocument()
  })

  it('renders HIGH level', () => {
    render(<UrgencyBadge level="HIGH" />)
    expect(screen.getByText('High')).toBeInTheDocument()
  })
})