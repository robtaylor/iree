# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IREE (Intermediate Representation Execution Environment) is an MLIR-based end-to-end compiler and runtime that lowers ML models to a unified IR for deployment across datacenter to edge devices.

This is the **iree-webgpu** branch/fork which includes experimental WebGPU support.

## Build Commands

### CMake (Recommended)

```bash
# Clone and init submodules
git submodule update --init

# Configure (Linux - recommended development settings)
cmake -G Ninja -B ../iree-build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON

# Configure (macOS)
cmake -G Ninja -B ../iree-build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

# Build
cmake --build ../iree-build/

# Build test dependencies (required for full test suite)
cmake --build ../iree-build/ --target iree-test-deps
```

### Faster Builds with ccache

```bash
cmake ... -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

### Key CMake Options

- `-DIREE_BUILD_COMPILER=OFF` - Drastically simplifies build (runtime only)
- `-DIREE_BUILD_TESTS=OFF` - Skip test builds
- `-DIREE_TARGET_BACKEND_DEFAULTS=OFF` - Disable all target backends
- `-DIREE_TARGET_BACKEND_LLVM_CPU=ON` - Enable specific backend
- `-DIREE_HAL_DRIVER_DEFAULTS=OFF` - Disable all HAL drivers
- `-DIREE_HAL_DRIVER_VULKAN=ON` - Enable specific driver
- `-DIREE_BUILD_PYTHON_BINDINGS=ON` - Build Python bindings
- `-DIREE_TARGET_BACKEND_WEBGPU_SPIRV=ON` - Enable WebGPU backend (experimental)
- `-DIREE_EXTERNAL_HAL_DRIVERS=webgpu` - Enable WebGPU HAL driver (experimental)

### Bazel (Linux only, internal use)

```bash
python3 configure_bazel.py
bazel test -k //...
```

## Testing

```bash
# Run all tests
ctest --test-dir ../iree-build/

# Run specific compiler lit test
ctest -R iree/compiler/Dialect/VM/Conversion/MathToVM/test/arithmetic_ops.mlir.test

# Run specific runtime test
ctest -R iree/base/bitfield_test

# Run e2e check tests
ctest -R tests/e2e/stablehlo_ops/check_vmvx_local-task_floor.mlir

# Run tests with parallel execution
CTEST_PARALLEL_LEVEL=$(nproc) ctest --test-dir ../iree-build/

# Utility target that builds deps and runs tests
cmake --build ../iree-build --target iree-run-tests
```

## Architecture

### Compiler (`compiler/src/iree/compiler/`)

- **API/** - C and Python APIs
- **Bindings/** - ABI binding generation
- **Codegen/** - Device code generation for various targets
- **ConstEval/** - JIT evaluation for constant optimization
- **Dialect/**
  - **Flow/** - Tensor program modeling and workload partitioning
  - **HAL/** - Hardware Abstraction Layer (buffer/execution management)
  - **Stream/** - Device placement and async scheduling
  - **Util/** - Common types
  - **VM/** - Abstract Virtual Machine
- **InputConversion/** - Frontend dialect conversions
- **Pipelines/** - Pipeline definitions

Compiler plugins: `compiler/plugins/` (input dialects, target backends)

### Runtime (`runtime/src/iree/`)

- **base/** - Base utilities and types
- **hal/** - Hardware Abstraction Layer implementation
- **vm/** - Virtual Machine implementation
- **task/** - Task system
- **io/** - I/O utilities
- **runtime/** - High-level runtime API
- **tooling/** - Development tools support

### Key Tools (built in `tools/`)

- `iree-compile` - Main compiler
- `iree-run-module` - Module runner
- `iree-opt` - MLIR optimization tool
- `iree-run-mlir` - Compile and run in one step

### WebGPU Support (Experimental)

- Compiler target: `compiler/plugins/target/WebGPUSPIRV/`
- Runtime HAL driver: `experimental/webgpu/`
- Web samples: `experimental/web/`

## Code Style

- **Compiler code**: clang-format with LLVM style (see `.clang-format`)
- **Runtime code**: Google style
- Pre-commit hooks enabled (see `.pre-commit-config.yaml`)

## Python Bindings

```bash
# Install requirements
python -m pip install -r runtime/bindings/python/iree/runtime/build_requirements.txt

# Build with Python support
cmake ... -DIREE_BUILD_PYTHON_BINDINGS=ON -DPython3_EXECUTABLE="$(which python3)"

# Install as editable (Linux/macOS)
CMAKE_INSTALL_METHOD=ABS_SYMLINK python -m pip install -e ../iree-build/compiler
CMAKE_INSTALL_METHOD=ABS_SYMLINK python -m pip install -e ../iree-build/runtime

# Or use PYTHONPATH
source ../iree-build/.env && export PYTHONPATH
```

## Cross-Compilation (Emscripten/WebAssembly)

```bash
# Build host tools first
cmake -G Ninja -B ../iree-build-host/ -DCMAKE_INSTALL_PREFIX=../iree-build-host/install .
cmake --build ../iree-build-host/ --target install

# Configure for Emscripten
emcmake cmake -G Ninja -B ../iree-build-emscripten/ \
  -DIREE_HOST_BIN_DIR=$(realpath ../iree-build-host/install/bin) \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_COMPILER=OFF \
  .
```

## Communication

- GitHub Issues: https://github.com/iree-org/iree/issues
- Discord: https://discord.gg/wEWh6Z9nMU
