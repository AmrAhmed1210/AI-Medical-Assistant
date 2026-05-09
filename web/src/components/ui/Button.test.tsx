import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Button } from '@/components/ui/Button'

describe('Button Component', () => {
  it('renders with children', () => {
    render(<Button>Click me</Button>)
    expect(screen.getByRole('button', { name: /click me/i })).toBeInTheDocument()
  })

  it('renders with different variants', () => {
    const variants = ['primary', 'outline', 'ghost', 'destructive', 'glass', 'success'] as const
    variants.forEach((variant) => {
      const { container } = render(<Button variant={variant}>Button</Button>)
      expect(container.firstChild).toBeInTheDocument()
    })
  })

  it('renders with different sizes', () => {
    const sizes = ['sm', 'md', 'lg'] as const
    sizes.forEach((size) => {
      const { container } = render(<Button size={size}>Button</Button>)
      expect(container.firstChild).toBeInTheDocument()
    })
  })

  it('shows loading spinner when loading', () => {
    render(<Button loading>Loading</Button>)
    const spinner = document.querySelector('.animate-spin')
    expect(spinner).toBeInTheDocument()
  })

  it('is disabled when disabled prop is true', () => {
    render(<Button disabled>Disabled</Button>)
    expect(screen.getByRole('button')).toBeDisabled()
  })

  it('is disabled when loading', () => {
    render(<Button loading>Loading</Button>)
    expect(screen.getByRole('button')).toBeDisabled()
  })

  it('renders with icon', () => {
    const icon = <span data-testid="icon">🔔</span>
    const { getByTestId } = render(<Button icon={icon}>With Icon</Button>)
    expect(getByTestId('icon')).toBeInTheDocument()
  })
})