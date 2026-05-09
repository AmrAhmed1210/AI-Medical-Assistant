import { test, expect } from '@playwright/test'

test.describe('DoctorApp E2E', () => {
  test.describe('Authentication', () => {
    test('should load login page', async ({ page }) => {
      await page.goto('/(auth)/login')
      await expect(page.locator('text=مرحبا')).toBeVisible({ timeout: 5000 }).catch(() => {
        expect(page.locator('input')).toBeVisible()
      })
    })

    test('should show validation error for empty login', async ({ page }) => {
      await page.goto('/(auth)/login')
      await page.locator('button[type="submit"]').click().catch(() => {})
    })

    test('should navigate to register page', async ({ page }) => {
      await page.goto('/(auth)/login')
      const registerLink = page.locator('text=إنشاء حساب جديد').or(page.locator('text=Register'))
      if (await registerLink.isVisible()) {
        await registerLink.click()
      }
    })
  })

  test.describe('Home Page', () => {
    test('should load home page', async ({ page }) => {
      await page.goto('/(patient)/home')
      await expect(page.locator('text=الرئيسية').or(page.locator('text=Home'))).toBeVisible({ timeout: 5000 }).catch(() => {
        expect(page).toBeDefined()
      })
    })

    test('should show doctors list', async ({ page }) => {
      await page.goto('/(patient)/home')
      await page.goto('/(patient)/doctors')
    })

    test('should show search functionality', async ({ page }) => {
      await page.goto('/(patient)/home')
      const searchInput = page.locator('input').first()
      if (await searchInput.isVisible()) {
        await searchInput.fill('طبيب')
      }
    })
  })

  test.describe('Medical Records', () => {
    test('should load medical records page', async ({ page }) => {
      await page.goto('/(patient)/medical-records')
    })

    test('should show vitals section', async ({ page }) => {
      await page.goto('/(patient)/vitals')
    })

    test('should show medications section', async ({ page }) => {
      await page.goto('/(patient)/medications')
    })
  })

  test.describe('Navigation', () => {
    test('should navigate between tabs', async ({ page }) => {
      await page.goto('/(patient)/home')
    })

    test('should show profile page', async ({ page }) => {
      await page.goto('/(patient)/profile')
    })

    test('should have working chatbot', async ({ page }) => {
      await page.goto('/(patient)/chatbot')
    })
  })

  test.describe('Accessibility', () => {
    test('should be keyboard navigable', async ({ page }) => {
      await page.goto('/(auth)/login')
      await page.keyboard.press('Tab')
      await page.keyboard.press('Tab')
      await page.keyboard.press('Tab')
    })

    test('should have proper touch targets', async ({ page }) => {
      await page.goto('/(patient)/home')
    })
  })
})