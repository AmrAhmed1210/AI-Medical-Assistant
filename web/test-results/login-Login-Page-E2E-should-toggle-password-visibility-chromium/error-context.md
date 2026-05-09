# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: login.spec.ts >> Login Page E2E >> should toggle password visibility
- Location: e2e\login.spec.ts:18:3

# Error details

```
Test timeout of 30000ms exceeded.
```

```
Error: locator.click: Test timeout of 30000ms exceeded.
Call log:
  - waiting for locator('button').filter({ has: locator('svg') }).nth(1)

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
  3  | test.describe('Login Page E2E', () => {
  4  |   test.beforeEach(async ({ page }) => {
  5  |     await page.goto('/login')
  6  |   })
  7  | 
  8  |   test('should load login page successfully', async ({ page }) => {
  9  |     await expect(page.locator('h1')).toContainText('MedBook Portal')
  10 |     await expect(page.locator('h2')).toContainText('Sign in to your account')
  11 |   })
  12 | 
  13 |   test('should show email and password fields', async ({ page }) => {
  14 |     await expect(page.locator('#login-email')).toBeVisible()
  15 |     await expect(page.locator('#login-password')).toBeVisible()
  16 |   })
  17 | 
  18 |   test('should toggle password visibility', async ({ page }) => {
  19 |     const passwordInput = page.locator('#login-password')
  20 |     await expect(passwordInput).toHaveAttribute('type', 'password')
  21 |     
> 22 |     await page.locator('button').filter({ has: page.locator('svg') }).nth(1).click()
     |                                                                              ^ Error: locator.click: Test timeout of 30000ms exceeded.
  23 |     await expect(passwordInput).toHaveAttribute('type', 'text')
  24 |   })
  25 | 
  26 |   test('should show validation error for empty email', async ({ page }) => {
  27 |     await page.locator('#login-password').fill('password123')
  28 |     await page.locator('button[type="submit"]').click()
  29 |   })
  30 | 
  31 |   test('should show error for invalid credentials', async ({ page }) => {
  32 |     await page.locator('#login-email').fill('invalid@test.com')
  33 |     await page.locator('#login-password').fill('wrongpassword')
  34 |     await page.locator('button[type="submit"]').click()
  35 |     
  36 |     await page.waitForTimeout(2000)
  37 |     await expect(page.locator('text=Invalid email or password')).toBeVisible({ timeout: 5000 })
  38 |   })
  39 | 
  40 |   test('should navigate to apply page', async ({ page }) => {
  41 |     await page.locator('text=Apply to join our platform').click()
  42 |     await expect(page).toHaveURL(/\/apply/)
  43 |   })
  44 | })
```