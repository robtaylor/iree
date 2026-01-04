// @ts-check
const { defineConfig, devices } = require('@playwright/test');

/**
 * Playwright configuration for IREE WebGPU testing.
 * @see https://playwright.dev/docs/test-configuration
 */
module.exports = defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',

  use: {
    // Base URL for the sample server
    baseURL: 'http://localhost:8000',

    // Collect trace on first retry
    trace: 'on-first-retry',

    // Screenshot on failure
    screenshot: 'only-on-failure',
  },

  projects: [
    {
      name: 'chromium-webgpu',
      use: {
        ...devices['Desktop Chrome'],
        // Enable WebGPU in Chromium
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-vulkan',
            '--disable-dawn-features=disallow_unsafe_apis',
          ],
        },
      },
    },
  ],

  // Run the sample server before starting tests
  webServer: {
    command: 'cd ../sample_webgpu && python3 -m http.server 8000',
    url: 'http://localhost:8000',
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
});
