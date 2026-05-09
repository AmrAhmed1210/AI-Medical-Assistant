import { test, expect } from '@playwright/test'

test.describe('Login Page E2E', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login')
  })

  test('should load login page successfully', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('MedBook Portal')
    await expect(page.locator('h2')).toContainText('Sign in to your account')
  })

  test('should show email and password fields', async ({ page }) => {
    await expect(page.locator('#login-email')).toBeVisible()
    await expect(page.locator('#login-password')).toBeVisible()
  })

  test('should toggle password visibility', async ({ page }) => {
    const passwordInput = page.locator('#login-password')
    await expect(passwordInput).toHaveAttribute('type', 'password')
    
    await page.locator('button').filter({ has: page.locator('svg') }).nth(1).click()
    await expect(passwordInput).toHaveAttribute('type', 'text')
  })

  test('should show validation error for empty email', async ({ page }) => {
    await page.locator('#login-password').fill('password123')
    await page.locator('button[type="submit"]').click()
  })

  test('should show error for invalid credentials', async ({ page }) => {
    await page.locator('#login-email').fill('invalid@test.com')
    await page.locator('#login-password').fill('wrongpassword')
    await page.locator('button[type="submit"]').click()
    
    await page.waitForTimeout(2000)
    await expect(page.locator('text=Invalid email or password')).toBeVisible({ timeout: 5000 })
  })

  test('should navigate to apply page', async ({ page }) => {
    await page.locator('text=Apply to join our platform').click()
    await expect(page).toHaveURL(/\/apply/)
  })
})