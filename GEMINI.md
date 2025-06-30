# Gemini Project Context: UniversalCompute (ILGPU)

## Project Overview

This project is **UniversalCompute**, a high-performance computing framework built on the **ILGPU** foundation. Its goal is to provide a unified API for various hardware accelerators, including CPUs, GPUs, NPUs, and more, with a focus on native Ahead-of-Time (AOT) compilation with .NET 9.

## Key Information

- **Project Name:** UniversalCompute (based on ILGPU)
- **Language:** C#, F#
- **Framework:** .NET 9.0 (with preview features)
- **Main Solution File:** `Src/ILGPU.sln`
- **Other Solution Files:** 
    - `Samples/Examples.sln` (subset of samples)
    - `Samples/ILGPU.Samples.sln` (comprehensive samples)
    - `Tools/Tools.sln` (development tools)
- **Source Code Root:** `Src/`
- **Examples Root:** `Samples/`
- **Documentation:** `Docs/`, `wiki-docs/`
- **Core Project:** `Src/ILGPU/ILGPU.csproj`

## Core Project Details (`ILGPU.csproj`)

- **TargetFramework:** `net9.0`
- **LangVersion:** `preview`
- **EnablePreviewFeatures:** `true`
- **Nullable:** `enable`
- **AllowUnsafeBlocks:** `true`
- **AOT Compatible:** `true`
- **Trimmable:** `true`
- **PackageId:** `UniversalCompute`

### Key Dependencies:
- **Microsoft.Extensions.DependencyInjection**
- **Microsoft.Extensions.Hosting**
- **Microsoft.Extensions.Logging**
- **Microsoft.Extensions.Options**
- **T4.Build** (for code generation)

### Project References:
- **ILGPU.SourceGenerators**

## Core Technologies & Accelerators

- **General:** .NET 9, Native AOT, T4 Templates for code generation
- **CPU:** Multi-threading, Intel AMX, Velocity SIMD
- **GPU:** NVIDIA CUDA, OpenCL, DirectCompute
- **AI/NPU:** Apple Neural Engine, Intel NPU
- **Specialized:** Intel IPP, BLAS libraries

## Build & Test Commands

- **Build Project:** `dotnet build Src --configuration Release`
- **Run CPU Tests:** `dotnet test Src/ILGPU.Tests.CPU --configuration Release --framework=net9.0`
- **Publish AOT Example:** `dotnet publish Samples/01_GettingStarted --configuration Release --runtime win-x64 --self-contained /p:PublishAot=true --framework=net9.0`

## Project Structure

- `.`: Root directory with solution files, documentation, and build artifacts.
- `Src/`: Contains the main source code for the ILGPU library and related components.
- `Samples/`: Contains a large number of sample projects demonstrating how to use the library. These are organized into two solution files: `Examples.sln` (a smaller, curated set) and `ILGPU.Samples.sln` (the full set).
- `Docs/`: Contains in-depth documentation.
- `wiki-docs/`: Contains markdown files for the project wiki.
- `Tools/`: Contains utility tools for development, such as the `CudaVersionUpdateTool` and scripts for generating compatibility suppression files.
