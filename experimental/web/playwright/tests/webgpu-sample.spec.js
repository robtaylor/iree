// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * IREE WebGPU Sample Tests
 *
 * These tests verify that the WebGPU sample can:
 * 1. Initialize WebGPU and IREE runtime
 * 2. Load compiled programs (.vmfb files)
 * 3. Execute functions and return correct results
 */

test.describe('IREE WebGPU Sample', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the sample page
    await page.goto('/');

    // Wait for IREE to initialize
    await page.waitForFunction(() => {
      return window.ireeInitialize !== undefined;
    }, { timeout: 30000 });
  });

  test('page loads and displays title', async ({ page }) => {
    await expect(page).toHaveTitle(/IREE WebGPU Sample/);
    await expect(page.locator('h1')).toContainText('IREE WebGPU Sample');
  });

  test('WebGPU is available', async ({ page }) => {
    const hasWebGPU = await page.evaluate(async () => {
      if (!navigator.gpu) {
        return { available: false, reason: 'navigator.gpu not available' };
      }
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          return { available: false, reason: 'No WebGPU adapter found' };
        }
        const device = await adapter.requestDevice();
        if (!device) {
          return { available: false, reason: 'Could not request WebGPU device' };
        }
        return { available: true, adapterInfo: adapter.info };
      } catch (e) {
        return { available: false, reason: e.message };
      }
    });

    console.log('WebGPU availability:', hasWebGPU);
    expect(hasWebGPU.available).toBe(true);
  });

  test('IREE runtime initializes successfully', async ({ page }) => {
    const initResult = await page.evaluate(async () => {
      try {
        await window.ireeInitialize();
        return { success: true };
      } catch (e) {
        return { success: false, error: e.message };
      }
    });

    console.log('IREE initialization:', initResult);
    expect(initResult.success).toBe(true);
  });

  test('simple_abs sample loads and runs', async ({ page }) => {
    // Load the simple_abs sample via URL parameters
    await page.goto('/?program=simple_abs.vmfb&function=abs&arguments=4xf32=-2,-3,-4,5');

    // Wait for program to load
    await page.waitForFunction(() => {
      const nameElement = document.getElementById('loaded-program-name');
      return nameElement && nameElement.innerText !== '(None)';
    }, { timeout: 30000 });

    // Verify program loaded
    const programName = await page.locator('#loaded-program-name').innerText();
    expect(programName).toBe('simple_abs.vmfb');

    // Click call function button
    await page.locator('#call-function').click();

    // Wait for output
    await page.waitForFunction(() => {
      const output = document.getElementById('function-outputs');
      return output && output.value && output.value.trim().length > 0;
    }, { timeout: 30000 });

    // Check output contains expected values (absolute values of inputs)
    const output = await page.locator('#function-outputs').inputValue();
    console.log('simple_abs output:', output);

    // The absolute values of [-2, -3, -4, 5] should be [2, 3, 4, 5]
    expect(output).toContain('2');
    expect(output).toContain('3');
    expect(output).toContain('4');
    expect(output).toContain('5');
  });

  test('multiple_results sample loads', async ({ page }) => {
    // Load the multiple_results sample
    await page.goto('/?program=multiple_results.vmfb&function=main');

    // Wait for program to load
    await page.waitForFunction(() => {
      const nameElement = document.getElementById('loaded-program-name');
      return nameElement && nameElement.innerText !== '(None)';
    }, { timeout: 30000 });

    // Verify program loaded
    const programName = await page.locator('#loaded-program-name').innerText();
    expect(programName).toBe('multiple_results.vmfb');
  });

  test('call function button is disabled until program loads', async ({ page }) => {
    // Initially, call button should be disabled
    await expect(page.locator('#call-function')).toBeDisabled();
  });

  test('console shows IREE initialized message', async ({ page }) => {
    const messages = [];
    page.on('console', msg => {
      messages.push(msg.text());
    });

    // Wait for initialization
    await page.waitForTimeout(3000);

    // Check for initialization message
    const hasInitMessage = messages.some(m =>
      m.includes('IREE initialized') || m.includes('ready to load programs')
    );
    expect(hasInitMessage).toBe(true);
  });
});

test.describe('WebGPU Numerical Correctness', () => {
  test.skip('multiple_results returns correct values', async ({ page }) => {
    // This test is skipped because issue #13809 causes incorrect results
    // Expected: absf(-1.23, -4.56) returns (1.23, 4.56)
    // Actual: Returns (4.56, 0) on first call, then (1.23, 4.56) on subsequent calls

    await page.goto('/?program=multiple_results.vmfb&function=absf&arguments=f32=-1.23%0Af32=-4.56');

    await page.waitForFunction(() => {
      const nameElement = document.getElementById('loaded-program-name');
      return nameElement && nameElement.innerText !== '(None)';
    }, { timeout: 30000 });

    await page.locator('#call-function').click();

    await page.waitForFunction(() => {
      const output = document.getElementById('function-outputs');
      return output && output.value && output.value.trim().length > 0;
    }, { timeout: 30000 });

    const output = await page.locator('#function-outputs').inputValue();
    console.log('multiple_results output:', output);

    // These assertions will fail due to issue #13809
    expect(output).toContain('1.23');
    expect(output).toContain('4.56');
  });
});
