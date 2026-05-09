# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: login.spec.ts >> Login Page E2E >> should show error for invalid credentials
- Location: e2e\login.spec.ts:31:3

# Error details

```
Error: expect(locator).toBeVisible() failed

Locator: locator('text=Invalid email or password')
Expected: visible
Error: strict mode violation: locator('text=Invalid email or password') resolved to 2 elements:
    1) <div role="status" aria-live="polite" class="go3958317564">Invalid email or password.</div> aka getByRole('status')
    2) <div class="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-2.5">Invalid email or password.</div> aka locator('form').getByText('Invalid email or password.')

Call log:
  - Expect "toBeVisible" with timeout 5000ms
  - waiting for locator('text=Invalid email or password')

```

# Page snapshot

```yaml
- generic [ref=e2]:
  - status [ref=e8]: Invalid email or password.
  - generic [ref=e10]:
    - generic [ref=e11]:
      - img [ref=e13]
      - heading "MedBook Portal" [level=1] [ref=e16]
      - paragraph [ref=e17]: Secure staff access
    - generic [ref=e18]:
      - heading "Sign in to your account" [level=2] [ref=e19]
      - generic [ref=e20]:
        - generic [ref=e21]:
          - generic [ref=e22]: Email address
          - generic [ref=e23]:
            - img [ref=e24]
            - textbox "you@example.com" [ref=e27]: invalid@test.com
        - generic [ref=e28]:
          - generic [ref=e29]: Password
          - generic [ref=e30]:
            - img [ref=e31]
            - textbox "••••••••" [ref=e34]: wrongpassword
            - button [ref=e35] [cursor=pointer]:
              - img [ref=e36]
        - generic [ref=e39]: Invalid email or password.
        - button "Sign In" [ref=e40] [cursor=pointer]
    - paragraph [ref=e41]:
      - text: Are you a doctor?
      - button "Apply to join our platform" [ref=e42] [cursor=pointer]
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
  22 |     await page.locator('button').filter({ has: page.locator('svg') }).nth(1).click()
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
> 37 |     await expect(page.locator('text=Invalid email or password')).toBeVisible({ timeout: 5000 })
     |                                                                  ^ Error: expect(locator).toBeVisible() failed
  38 |   })
  39 | 
  40 |   test('should navigate to apply page', async ({ page }) => {
  41 |     await page.locator('text=Apply to join our platform').click()
  42 |     await expect(page).toHaveURL(/\/apply/)
  43 |   })
  44 | })
```