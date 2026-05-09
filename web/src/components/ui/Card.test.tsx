import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card'

describe('Card Component', () => {
  it('renders with children', () => {
    render(<Card>Card Content</Card>)
    expect(screen.getByText('Card Content')).toBeInTheDocument()
  })

  it('applies custom className', () => {
    const { container } = render(<Card className="custom-class">Content</Card>)
    expect(container.firstChild).toHaveClass('custom-class')
  })

  it('has proper styling classes', () => {
    const { container } = render(<Card>Content</Card>)
    expect(container.firstChild).toHaveClass('rounded-3xl')
    expect(container.firstChild).toHaveClass('shadow-xl')
    expect(container.firstChild).toHaveClass('bg-white/80')
  })
})

describe('CardHeader Component', () => {
  it('renders children', () => {
    render(<CardHeader>Header Content</CardHeader>)
    expect(screen.getByText('Header Content')).toBeInTheDocument()
  })

  it('has proper spacing classes', () => {
    const { container } = render(<CardHeader />)
    expect(container.firstChild).toHaveClass('flex')
    expect(container.firstChild).toHaveClass('flex-col')
    expect(container.firstChild).toHaveClass('space-y-2')
  })
})

describe('CardTitle Component', () => {
  it('renders title text', () => {
    render(<CardTitle>Test Title</CardTitle>)
    expect(screen.getByText('Test Title')).toBeInTheDocument()
  })

  it('has proper typography classes', () => {
    const { container } = render(<CardTitle>Title</CardTitle>)
    expect(container.firstChild).toHaveClass('text-xl')
    expect(container.firstChild).toHaveClass('font-bold')
  })
})

describe('CardDescription Component', () => {
  it('renders description text', () => {
    render(<CardDescription>Test Description</CardDescription>)
    expect(screen.getByText('Test Description')).toBeInTheDocument()
  })

  it('has proper styling for muted text', () => {
    const { container } = render(<CardDescription>Desc</CardDescription>)
    expect(container.firstChild).toHaveClass('text-sm')
    expect(container.firstChild).toHaveClass('text-gray-500')
  })
})

describe('CardContent Component', () => {
  it('renders children', () => {
    render(<CardContent>Content Area</CardContent>)
    expect(screen.getByText('Content Area')).toBeInTheDocument()
  })

  it('has proper padding classes', () => {
    const { container } = render(<CardContent />)
    expect(container.firstChild).toHaveClass('p-6')
    expect(container.firstChild).toHaveClass('pt-0')
  })
})