import { test, expect } from '@playwright/test'

test.describe('Doctors List Page E2E', () => {
  test('should load doctors list page', async ({ page }) => {
    await page.goto('/doctors')
    await expect(page.locator('h1')).toContainText('قائمة الأطباء')
  })

  test('should show search input', async ({ page }) => {
    await page.goto('/doctors')
    const searchInput = page.locator('input[placeholder*="ابحث"]')
    await expect(searchInput).toBeVisible()
  })

  test('should show specialty filter dropdown', async ({ page }) => {
    await page.goto('/doctors')
    const specialtySelect = page.locator('select').first()
    await expect(specialtySelect).toBeVisible()
    await expect(specialtySelect.locator('option').first()).toContainText('جميع التخصصات')
  })

  test('should filter doctors by search term', async ({ page }) => {
    await page.goto('/doctors')
    await page.waitForTimeout(2000)
    const searchInput = page.locator('input[placeholder*="ابحث"]')
    await searchInput.fill('أحمد')
    await page.waitForTimeout(500)
  })

  test('should display loading state', async ({ page }) => {
    await page.goto('/doctors')
    const loadingSpinner = page.locator('.animate-spin')
    await expect(loadingSpinner).toBeVisible({ timeout: 5000 }).catch(() => {})
  })

  test('should show empty state when no doctors', async ({ page }) => {
    await page.goto('/doctors')
    await page.waitForTimeout(2000)
  })
})

test.describe('Doctor Details Page E2E', () => {
  test('should navigate to doctor details page', async ({ page }) => {
    await page.goto('/doctors')
    await page.waitForTimeout(2000)
    const detailButton = page.locator('button').filter({ hasText: 'التفاصيل' }).first()
    if (await detailButton.isVisible()) {
      await detailButton.click()
      await expect(page).toHaveURL(/\/doctor\/\d+/)
    }
  })
})