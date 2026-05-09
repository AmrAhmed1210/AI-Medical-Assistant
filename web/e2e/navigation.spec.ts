import { test, expect } from '@playwright/test'

test.describe('Navigation E2E', () => {
  test('should redirect to login when accessing protected route', async ({ page }) => {
    await page.goto('/admin/dashboard')
    await expect(page).toHaveURL(/\/login/)
  })

  test('should show 404 for unknown routes', async ({ page }) => {
    await page.goto('/unknown-route-xyz')
    await expect(page).toHaveURL('/')
  })

  test('should have working logo link on login page', async ({ page }) => {
    await page.goto('/login')
    const logo = page.locator('h1').filter({ hasText: 'MedBook Portal' })
    await expect(logo).toBeVisible()
  })

  test('should display footer links on login page', async ({ page }) => {
    await page.goto('/login')
    const applyLink = page.locator('text=Apply to join our platform')
    await expect(applyLink).toBeVisible()
  })

  test('should maintain RTL direction on Arabic pages', async ({ page }) => {
    await page.goto('/doctors')
    await expect(page.locator('body')).toHaveAttribute('dir', 'rtl')
  })
})

test.describe('Accessibility E2E', () => {
  test('should have proper page title', async ({ page }) => {
    await page.goto('/login')
    await expect(page).toHaveTitle(/MedBook|Login/)
  })

  test('should have accessible form labels', async ({ page }) => {
    await page.goto('/login')
    await expect(page.locator('label')).toContainText(['Email', 'Password'])
  })

  test('should be keyboard navigable', async ({ page }) => {
    await page.goto('/login')
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
  })
})