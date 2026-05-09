# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: navigation.spec.ts >> Navigation E2E >> should maintain RTL direction on Arabic pages
- Location: e2e\navigation.spec.ts:26:3

# Error details

```
Error: expect(locator).toHaveAttribute(expected) failed

Locator:  locator('body')
Expected: "rtl"
Received: ""
Timeout:  5000ms

Call log:
  - Expect "toHaveAttribute" with timeout 5000ms
  - waiting for locator('body')
    9 × locator resolved to <body>…</body>
      - unexpected value "null"

```

# Test source

```ts
  1  | import { test, expect } from '@playwright/test'
  2  | 
  3  | test.describe('Navigation E2E', () => {
  4  |   test('should redirect to login when accessing protected route', async ({ page }) => {
  5  |     await page.goto('/admin/dashboard')
  6  |     await expect(page).toHaveURL(/\/login/)
  7  |   })
  8  | 
  9  |   test('should show 404 for unknown routes', async ({ page }) => {
  10 |     await page.goto('/unknown-route-xyz')
  11 |     await expect(page).toHaveURL('/')
  12 |   })
  13 | 
  14 |   test('should have working logo link on login page', async ({ page }) => {
  15 |     await page.goto('/login')
  16 |     const logo = page.locator('h1').filter({ hasText: 'MedBook Portal' })
  17 |     await expect(logo).toBeVisible()
  18 |   })
  19 | 
  20 |   test('should display footer links on login page', async ({ page }) => {
  21 |     await page.goto('/login')
  22 |     const applyLink = page.locator('text=Apply to join our platform')
  23 |     await expect(applyLink).toBeVisible()
  24 |   })
  25 | 
  26 |   test('should maintain RTL direction on Arabic pages', async ({ page }) => {
  27 |     await page.goto('/doctors')
> 28 |     await expect(page.locator('body')).toHaveAttribute('dir', 'rtl')
     |                                        ^ Error: expect(locator).toHaveAttribute(expected) failed
  29 |   })
  30 | })
  31 | 
  32 | test.describe('Accessibility E2E', () => {
  33 |   test('should have proper page title', async ({ page }) => {
  34 |     await page.goto('/login')
  35 |     await expect(page).toHaveTitle(/MedBook|Login/)
  36 |   })
  37 | 
  38 |   test('should have accessible form labels', async ({ page }) => {
  39 |     await page.goto('/login')
  40 |     await expect(page.locator('label')).toContainText(['Email', 'Password'])
  41 |   })
  42 | 
  43 |   test('should be keyboard navigable', async ({ page }) => {
  44 |     await page.goto('/login')
  45 |     await page.keyboard.press('Tab')
  46 |     await page.keyboard.press('Tab')
  47 |   })
  48 | })
```