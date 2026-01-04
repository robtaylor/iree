# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)

FetchContent_Declare(
  tint
  GIT_REPOSITORY https://dawn.googlesource.com/tint
  GIT_TAG ${IREE_TINT_TAG}
)

set(TINT_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_DOCS  OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# Our usage at compile time primarily needs the SPIR-V reader and WGSL writer.
# Other readers and writers could be enabled, but we want to keep this small.

set(TINT_BUILD_SPV_READER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_WGSL_READER OFF CACHE BOOL "" FORCE)

set(TINT_BUILD_GLSL_WRITER OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_HLSL_WRITER OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_MSL_WRITER OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_SPV_WRITER OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_WGSL_WRITER ON CACHE BOOL "" FORCE)

# Tint expects SPIRV-Tools and SPIRV-Headers in a specific directory structure
# under TINT_THIRD_PARTY_DIR. Since IREE fetches these dependencies separately,
# we need to create the expected directory structure with symlinks before
# building Tint.
#
# First, populate tint without building it.
FetchContent_GetProperties(tint)
if(NOT tint_POPULATED)
  FetchContent_Populate(tint)
endif()

# Get the source directories for SPIRV-Tools and SPIRV-Headers
FetchContent_GetProperties(spirv-tools SOURCE_DIR SPIRV_TOOLS_SOURCE_DIR)
FetchContent_GetProperties(spirv-headers SOURCE_DIR SPIRV_HEADERS_SOURCE_DIR)

# Create the directory structure Tint expects:
# ${tint_SOURCE_DIR}/third_party/vulkan-deps/spirv-tools/src -> spirv-tools-src
# ${tint_SOURCE_DIR}/third_party/vulkan-deps/spirv-headers/src -> spirv-headers-src
set(TINT_VULKAN_DEPS_DIR "${tint_SOURCE_DIR}/third_party/vulkan-deps")
file(MAKE_DIRECTORY "${TINT_VULKAN_DEPS_DIR}/spirv-tools")
file(MAKE_DIRECTORY "${TINT_VULKAN_DEPS_DIR}/spirv-headers")

# Create symlinks (or copy on Windows)
if(NOT EXISTS "${TINT_VULKAN_DEPS_DIR}/spirv-tools/src")
  file(CREATE_LINK "${SPIRV_TOOLS_SOURCE_DIR}" "${TINT_VULKAN_DEPS_DIR}/spirv-tools/src" SYMBOLIC)
endif()
if(NOT EXISTS "${TINT_VULKAN_DEPS_DIR}/spirv-headers/src")
  file(CREATE_LINK "${SPIRV_HEADERS_SOURCE_DIR}" "${TINT_VULKAN_DEPS_DIR}/spirv-headers/src" SYMBOLIC)
endif()

# Suppress strict warnings that fail with newer compilers.
# Tint uses -Weverything -Werror which triggers these on macOS/newer clang.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    -Wno-poison-system-directories
    -Wno-switch-default
  )
endif()

# Now add Tint's subdirectory
add_subdirectory(${tint_SOURCE_DIR} ${tint_BINARY_DIR} EXCLUDE_FROM_ALL)
