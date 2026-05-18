import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { LoadingSpinner, PageLoader, FullPageLoader } from '@/components/ui/LoadingSpinner'

describe('LoadingSpinner Component', () => {
  it('renders with default size', () => {
    const { container } = render(<LoadingSpinner />)
    expect(container.firstChild).toBeInTheDocument()
    expect(container.firstChild).toHaveClass('w-6')
    expect(container.firstChild).toHaveClass('h-6')
  })

  it('renders with sm size', () => {
    const { container } = render(<LoadingSpinner size="sm" />)
    expect(container.firstChild).toHaveClass('w-4')
    expect(container.firstChild).toHaveClass('h-4')
  })

  it('renders with lg size', () => {
    const { container } = render(<LoadingSpinner size="lg" />)
    expect(container.firstChild).toHaveClass('w-8')
    expect(container.firstChild).toHaveClass('h-8')
  })

  it('renders with xl size', () => {
    const { container } = render(<LoadingSpinner size="xl" />)
    expect(container.firstChild).toHaveClass('w-12')
    expect(container.firstChild).toHaveClass('h-12')
    expect(container.firstChild).toHaveClass('border-4')
  })

  it('has animate-spin class', () => {
    const { container } = render(<LoadingSpinner />)
    expect(container.firstChild).toHaveClass('animate-spin')
  })

  it('accepts custom className', () => {
    const { container } = render(<LoadingSpinner className="custom-spinner" />)
    expect(container.firstChild).toHaveClass('custom-spinner')
  })
})

describe('PageLoader Component', () => {
  it('renders spinner and loading text', () => {
    render(<PageLoader />)
    const spinner = document.querySelector('.animate-spin')
    expect(spinner).toBeInTheDocument()
    expect(screen.getByText('Loading...')).toBeInTheDocument()
  })

  it('centers content', () => {
    const { container } = render(<PageLoader />)
    expect(container.firstChild).toHaveClass('flex')
    expect(container.firstChild).toHaveClass('items-center')
    expect(container.firstChild).toHaveClass('justify-center')
  })
})

describe('FullPageLoader Component', () => {
  it('renders logo and spinner', () => {
    render(<FullPageLoader />)
    const spinner = document.querySelector('.animate-spin')
    expect(spinner).toBeInTheDocument()
  })

  it('shows loading text', () => {
    render(<FullPageLoader />)
    expect(screen.getByText('Loading MedBook...')).toBeInTheDocument()
  })

  it('has fixed positioning', () => {
    const { container } = render(<FullPageLoader />)
    expect(container.firstChild).toHaveClass('fixed')
    expect(container.firstChild).toHaveClass('inset-0')
  })
})