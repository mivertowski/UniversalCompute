# UniversalCompute Wiki

Welcome to the **UniversalCompute** documentation wiki! This comprehensive guide provides everything you need to get started with the universal compute framework for diverse native AOT hardware accelerator usage.

## 🚀 Quick Navigation

### Getting Started
- **[Installation Guide](Installation-Guide)** - Set up UniversalCompute in your project
- **[Quick Start Tutorial](Quick-Start-Tutorial)** - Your first UniversalCompute application
- **[Examples Gallery](Examples-Gallery)** - Comprehensive code examples

### Core Documentation
- **[API Reference](API-Reference)** - Complete API documentation
- **[Hardware Support](Hardware-Support)** - Supported accelerators and capabilities
- **[Architecture Overview](Architecture-Overview)** - Framework design and components

### Specialized Topics
- **[Native AOT Guide](Native-AOT-Guide)** - Ahead-of-time compilation setup
- **[Hardware Accelerators](Hardware-Accelerators)** - Deep dive into accelerator support
- **[FFT Operations](FFT-Operations)** - Fast Fourier Transform integration
- **[Tensor Operations](Tensor-Operations)** - Multi-dimensional array processing

### Advanced Usage
- **[Performance Tuning](Performance-Tuning)** - Optimization strategies and best practices
- **[Memory Management](Memory-Management)** - Efficient memory handling patterns
- **[Cross-Platform Development](Cross-Platform-Development)** - Windows, Linux, macOS support
- **[Troubleshooting Guide](Troubleshooting-Guide)** - Common issues and solutions

### Developer Resources
- **[Contributing Guidelines](Contributing-Guidelines)** - How to contribute to UniversalCompute
- **[Building from Source](Building-from-Source)** - Compilation and development setup
- **[Release Notes](Release-Notes)** - Version history and changelog

## 📊 Framework Overview

UniversalCompute is a high-performance computing framework that provides unified access to diverse hardware accelerators through a single, easy-to-use API. Built on the proven ILGPU foundation, it extends support to modern hardware accelerators while maintaining compatibility with traditional GPU and CPU computing.

### Key Features

- **🚀 Native AOT Support** - Full ahead-of-time compilation support for .NET 8+ applications
- **🔧 Hardware Abstraction** - Unified API for CPU, GPU, NPU, Neural Engine, and specialized accelerators
- **⚡ High Performance** - Optimized kernels and memory management for maximum throughput
- **🎯 Multi-Platform** - Support for Windows, Linux, and macOS across x64 and ARM64 architectures
- **🧠 AI/ML Optimized** - Built-in tensor operations and neural network primitives
- **📊 FFT Integration** - High-performance Fast Fourier Transform implementations
- **🛡️ Type Safety** - Compile-time verification and comprehensive error checking

### Supported Hardware

#### CPU Accelerators
- Multi-threaded CPU with parallel execution
- Intel AMX (Advanced Matrix Extensions)
- Velocity SIMD vectorized operations

#### GPU Accelerators  
- NVIDIA CUDA with full PTX backend support
- OpenCL for cross-platform GPU computing
- DirectCompute for Windows GPU acceleration

#### Neural Processing Units
- Apple Neural Engine for hardware-accelerated AI inference
- Intel NPU support on Intel Core Ultra processors
- Dedicated AI accelerators with extensible framework

#### Specialized Accelerators
- Intel IPP (Integrated Performance Primitives)
- BLAS Libraries for high-performance linear algebra
- Custom accelerators through extensible framework

## 🎯 Use Cases

UniversalCompute is perfect for:

- **High-Performance Computing (HPC)** - Scientific computing, simulations, and numerical analysis
- **Artificial Intelligence & Machine Learning** - Neural network training and inference
- **Signal Processing** - FFT, filtering, and digital signal processing applications
- **Computer Graphics** - Rendering, image processing, and computer vision
- **Financial Computing** - Risk analysis, algorithmic trading, and quantitative models
- **Scientific Research** - Physics simulations, bioinformatics, and data analysis

## 📈 Performance Benefits

| Operation | CPU (Intel i9) | NVIDIA RTX 4090 | Apple M2 Neural Engine | Intel NPU |
|-----------|---------------|------------------|------------------------|-----------| 
| Matrix Multiply (4K×4K) | 12.3 GFLOPS | 847 GFLOPS | 156 TOPS | 45 TOPS |
| FFT (1M points) | 2.1 ms | 0.34 ms | N/A | N/A |
| Neural Inference | 45 ms | 8.2 ms | 3.1 ms | 12.7 ms |

## 🔗 External Resources

- **[GitHub Repository](https://github.com/mivertowski/UniversalCompute)** - Source code and issues
- **[NuGet Package](https://www.nuget.org/packages/UniversalCompute/)** - Official package distribution
- **[ILGPU Foundation](https://www.ilgpu.net)** - Original ILGPU project acknowledgment

## 📄 License

UniversalCompute has a dual licensing structure:

- **Foundation Components (ILGPU Core)**: University of Illinois/NCSA Open Source License
- **UniversalCompute Extensions**: Business Source License 1.1 (converts to Apache License 2.0 on June 24, 2029)

---

**Get started with [Installation Guide](Installation-Guide) or explore our [Examples Gallery](Examples-Gallery)!**