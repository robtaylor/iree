# IREE WebGPU Playwright Tests

Automated browser tests for the IREE WebGPU sample using [Playwright](https://playwright.dev/).

## Prerequisites

1. Build the WebGPU sample (see `../sample_webgpu/README.md`)
2. Node.js 18+ installed

## Setup

```bash
cd experimental/web/playwright
npm install
npm run install-browsers
```

## Running Tests

```bash
# Run all tests (headless)
npm test

# Run tests with browser visible
npm run test:headed

# Debug mode (step through tests)
npm run test:debug
```

## Test Structure

- `tests/webgpu-sample.spec.js` - Tests for the WebGPU sample page
  - Page loading and initialization
  - WebGPU availability check
  - IREE runtime initialization
  - Sample program loading and execution
  - Numerical correctness tests (some skipped due to known issues)

## Known Issues

- **Issue #13809**: Memory/alignment errors cause incorrect numerical results
  - `multiple_results` sample returns wrong values on first call
  - Tests for numerical correctness are currently skipped

## CI Integration

The tests can be run in CI with:

```bash
npm ci
npx playwright install --with-deps chromium
npm test
```

The `playwright.config.js` automatically starts the sample server before running tests.

## WebGPU Browser Requirements

Tests run on Chromium with WebGPU flags:
- `--enable-unsafe-webgpu`
- `--enable-features=Vulkan`

For CI environments, ensure the system has Vulkan drivers installed.
