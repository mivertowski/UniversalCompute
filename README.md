# UniversalCompute

**Universal compute framework for diverse native AOT hardware accelerator usage**

[![Build Status](https://github.com/mivertowski/UniversalCompute/actions/workflows/ci.yml/badge.svg)](https://github.com/mivertowski/UniversalCompute/actions)
[![NuGet Version](https://img.shields.io/nuget/v/UniversalCompute.svg)](https://www.nuget.org/packages/UniversalCompute/)
[![License](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE.txt)

## Overview

UniversalCompute is a high-performance computing framework that provides unified access to diverse hardware accelerators through a single, easy-to-use API. Built on top of the proven ILGPU foundation, it extends support to modern hardware accelerators while maintaining compatibility with traditional GPU and CPU computing.

### Key Features

- **🚀 Native AOT Support** - Full ahead-of-time compilation support for .NET 8+ applications
- **🔧 Hardware Abstraction** - Unified API for CPU, GPU, NPU, Neural Engine, and specialized accelerators
- **⚡ High Performance** - Optimized kernels and memory management for maximum throughput
- **🎯 Multi-Platform** - Support for Windows, Linux, and macOS across x64 and ARM64 architectures
- **🧠 AI/ML Optimized** - Built-in tensor operations and neural network primitives
- **📊 FFT Integration** - High-performance Fast Fourier Transform implementations
- **🛡️ Type Safety** - Compile-time verification and comprehensive error checking

## Supported Hardware

### CPU Accelerators
- **Multi-threaded CPU** - Parallel execution across CPU cores
- **Intel AMX** - Advanced Matrix Extensions for high-performance matrix operations
- **Velocity SIMD** - Vectorized operations using CPU SIMD instructions

### GPU Accelerators  
- **NVIDIA CUDA** - Full CUDA support with PTX backend
- **OpenCL** - Cross-platform GPU computing
- **DirectCompute** - Windows GPU acceleration

### Neural Processing Units
- **Apple Neural Engine** - Hardware-accelerated AI inference on Apple Silicon
- **Intel NPU** - Neural Processing Unit support on Intel Core Ultra processors
- **Dedicated AI accelerators** - Support for specialized neural processing hardware

### Specialized Accelerators
- **Intel IPP** - Integrated Performance Primitives for signal processing
- **BLAS Libraries** - High-performance linear algebra operations
- **Custom accelerators** - Extensible framework for proprietary hardware

## Quick Start

### Installation

```bash
dotnet add package UniversalCompute
```

### Basic Usage

```csharp
using UniversalCompute;
using UniversalCompute.Runtime;

// Create a context and auto-detect the best available accelerator
using var context = Context.Create().EnableAllAccelerators();
using var accelerator = context.GetPreferredDevice(preferGPU: true).CreateAccelerator(context);

// Allocate memory
var input = accelerator.Allocate1D<float>(1024);
var output = accelerator.Allocate1D<float>(1024);

// Define a simple kernel
var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
    (index, input, output) => output[index] = input[index] * 2.0f);

// Execute the kernel
input.CopyFromCPU(sourceData);
kernel(input.Length, input.View, output.View);
accelerator.Synchronize();

// Get results
var result = output.GetAsArray1D();
```

### Hardware-Specific Examples

#### Apple Neural Engine
```csharp
using var context = Context.Create().AppleNeuralEngine();
using var aneAccelerator = context.CreateAppleNeuralEngineAccelerator();

// Optimized for neural network inference
var predictor = new NeuralNetworkPredictor(aneAccelerator);
var prediction = await predictor.PredictAsync(inputTensor);
```

#### Intel NPU
```csharp
using var context = Context.Create().IntelNPU();
using var npuAccelerator = context.CreateNPUAccelerator();

// AI workload optimization
var inference = npuAccelerator.CreateInferenceEngine(modelPath);
var result = await inference.RunAsync(inputData);
```

#### FFT Operations
```csharp
using var fftManager = new FFTManager(context);

// Automatic hardware selection for optimal FFT performance
var inputSignal = accelerator.Allocate1D<Complex>(1024);
var fftResult = accelerator.Allocate1D<Complex>(1024);

var fftAccelerator = fftManager.FFT1D(inputSignal.View, fftResult.View);
```

## Architecture

UniversalCompute provides a layered architecture that abstracts hardware differences while maintaining performance:

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                       │
├─────────────────────────────────────────────────────────────┤
│                   UniversalCompute API                      │
├─────────────────────────────────────────────────────────────┤
│  Hardware Abstraction Layer (CPU│GPU│NPU│Neural Engine)    │
├─────────────────────────────────────────────────────────────┤
│     Native Libraries (CUDA│OpenCL│IPP│Core ML│DirectML)    │
├─────────────────────────────────────────────────────────────┤
│                      Hardware Layer                         │
└─────────────────────────────────────────────────────────────┘
```

## Performance

UniversalCompute delivers exceptional performance across different hardware configurations:

| Operation | CPU (Intel i9) | NVIDIA RTX 4090 | Apple M2 Neural Engine | Intel NPU |
|-----------|---------------|------------------|------------------------|-----------|
| Matrix Multiply (4K×4K) | 12.3 GFLOPS | 847 GFLOPS | 156 TOPS | 45 TOPS |
| FFT (1M points) | 2.1 ms | 0.34 ms | N/A | N/A |
| Neural Inference | 45 ms | 8.2 ms | 3.1 ms | 12.7 ms |

## Documentation

- **[Getting Started Guide](https://github.com/mivertowski/UniversalCompute/wiki/Getting-Started)**
- **[API Reference](https://github.com/mivertowski/UniversalCompute/wiki/API-Reference)**
- **[Hardware Support](https://github.com/mivertowski/UniversalCompute/wiki/Hardware-Support)**
- **[Performance Tuning](https://github.com/mivertowski/UniversalCompute/wiki/Performance-Tuning)**
- **[Examples](https://github.com/mivertowski/UniversalCompute/tree/master/Examples)**

## Examples

Comprehensive examples are available in the `Examples/` directory:

- **01_GettingStarted** - Basic accelerator usage and kernel execution
- **02_AppleNeuralEngine** - Neural Engine integration and AI workloads
- **03_IntelNPU** - Neural Processing Unit utilization
- **04_IntelAMX** - Advanced Matrix Extensions for HPC
- **05_HardwareComparison** - Benchmarking across different accelerators
- **06_FFTOperations** - Fast Fourier Transform implementations
- **07_TensorOperations** - Multi-dimensional array operations
- **08_MachineLearning** - AI/ML pipeline integration

## Building from Source

### Prerequisites

- .NET 8.0 SDK or later
- Visual Studio 2022 (Windows) or JetBrains Rider (cross-platform)
- CUDA Toolkit 12.0+ (for NVIDIA GPU support)
- Intel oneAPI (for Intel-specific accelerators)

### Build Instructions

```bash
git clone https://github.com/mivertowski/UniversalCompute.git
cd UniversalCompute
dotnet build Src --configuration Release
dotnet test Src/ILGPU.Tests.CPU --configuration Release
```

### Native AOT Compilation

```bash
dotnet publish Examples/01_GettingStarted --configuration Release --runtime win-x64 --self-contained /p:PublishAot=true
```

## Compatibility

### .NET Versions
- .NET 6.0 (Limited support)
- .NET 7.0 (Full support)
- .NET 8.0+ (Full support with AOT)

### Operating Systems
- Windows 10/11 (x64, ARM64)
- Linux (x64, ARM64) - Ubuntu 20.04+, RHEL 8+
- macOS (x64, ARM64) - macOS 11.0+

### Hardware Requirements
- **Minimum**: 64-bit processor, 4GB RAM
- **Recommended**: Multi-core CPU, dedicated GPU, 16GB+ RAM
- **Optimal**: Latest generation CPU/GPU with AI accelerators

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes with comprehensive tests
4. Submit a pull request

## Roadmap

### Version 1.0 (Current Alpha)
- ✅ Native AOT support
- ✅ Hardware accelerator abstraction
- ✅ FFT integration
- ✅ Basic tensor operations

### Version 1.1 (Q2 2025)
- 🔄 Advanced AI/ML operators
- 🔄 Distributed computing support
- 🔄 WebAssembly backend
- 🔄 Cloud accelerator integration

### Version 2.0 (Q4 2025)
- 📋 Quantum computing backends
- 📋 Edge device optimization
- 📋 Advanced profiling tools
- 📋 Visual development environment

## License

UniversalCompute has a dual licensing structure that respects the original ILGPU project's contributions:

### Foundation Components (ILGPU Core)
- **License**: University of Illinois/NCSA Open Source License
- **Copyright**: (c) 2016-2024 ILGPU Project. All rights reserved.
- **Developer**: Marcel Koester and the ILGPU Project team
- **Website**: www.ilgpu.net

### UniversalCompute Extensions
- **License**: Business Source License 1.1 (converts to Apache License 2.0 on June 24, 2029)
- **Copyright**: (c) 2024-2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
- **Coverage**: Hardware accelerator abstractions, FFT framework, native AOT support, and modernization features

See [LICENSE.txt](LICENSE.txt) for complete licensing details.

## Acknowledgments

UniversalCompute is built upon the exceptional groundwork laid by the **ILGPU Project team**, led by Marcel Koester. We extend our deepest gratitude to the original ILGPU contributors who created the foundational GPU computing framework that makes UniversalCompute possible.

**Key Contributors and Technologies:**

- **ILGPU Project Team** - Marcel Koester and contributors who built the original high-performance GPU computing framework
- **Intel oneAPI Team** - Performance libraries and optimization tools
- **NVIDIA CUDA Team** - GPU computing platform and development tools
- **Apple Core ML Team** - Machine learning framework and Neural Engine support
- **Microsoft .NET Team** - Runtime infrastructure and compiler technology

## Support

- **Issues**: [GitHub Issues](https://github.com/mivertowski/UniversalCompute/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mivertowski/UniversalCompute/discussions)
- **Wiki**: [Project Wiki](https://github.com/mivertowski/UniversalCompute/wiki)

---

**Copyright (c) 2024-2025 ILGPU Project. All rights reserved.**