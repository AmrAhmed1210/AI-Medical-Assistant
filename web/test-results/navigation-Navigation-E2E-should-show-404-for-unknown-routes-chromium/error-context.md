# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: navigation.spec.ts >> Navigation E2E >> should show 404 for unknown routes
- Location: e2e\navigation.spec.ts:9:3

# Error details

```
Error: expect(page).toHaveURL(expected) failed

Expected: "http://localhost:3000/"
Received: "http://localhost:3000/login"
Timeout:  5000ms

Call log:
  - Expect "toHaveURL" with timeout 5000ms
    8 × unexpected value "http://localhost:3000/login"

```

# Page snapshot

```yaml
- generic [ref=e4]:
  - generic [ref=e5]:
    - img [ref=e7]
    - heading "MedBook Portal" [level=1] [ref=e10]
    - paragraph [ref=e11]: Secure staff access
  - generic [ref=e12]:
    - heading "Sign in to your account" [level=2] [ref=e13]
    - generic [ref=e14]:
      - generic [ref=e15]:
        - generic [ref=e16]: Email address
        - generic [ref=e17]:
          - img [ref=e18]
          - textbox "you@example.com" [ref=e21]
      - generic [ref=e22]:
        - generic [ref=e23]: Password
        - generic [ref=e24]:
          - img [ref=e25]
          - textbox "••••••••" [ref=e28]
          - button [ref=e29] [cursor=pointer]:
            - img [ref=e30]
      - button "Sign In" [ref=e33] [cursor=pointer]
  - paragraph [ref=e34]:
    - text: Are you a doctor?
    - button "Apply to join our platform" [ref=e35] [cursor=pointer]
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
> 11 |     await expect(page).toHaveURL('/')
     |                        ^ Error: expect(page).toHaveURL(expected) failed
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
  28 |     await expect(page.locator('body')).toHaveAttribute('dir', 'rtl')
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