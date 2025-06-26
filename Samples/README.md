# UniversalCompute Samples

Welcome to the UniversalCompute samples repository! This directory contains a comprehensive collection of examples demonstrating various features and capabilities of the UniversalCompute framework.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Sample Categories](#sample-categories)
- [Hardware Accelerator Examples](#hardware-accelerator-examples)
- [Legacy ILGPU Samples](#legacy-ilgpu-samples)
- [Running Samples](#running-samples)
- [Contributing](#contributing)

## 🚀 Getting Started

The samples are organized into categories based on complexity and feature areas. If you're new to UniversalCompute, we recommend starting with:

1. **01_GettingStarted** - Basic accelerator setup and kernel execution
2. **SimpleKernel** - Fundamental kernel programming concepts
3. **DeviceInfo** - Hardware detection and capability discovery

## 📁 Sample Categories

### Modern UniversalCompute Examples

#### **01_GettingStarted**
- **BasicAccelerator.cs** - Introduction to accelerator creation and basic operations
- Demonstrates context creation, device selection, and simple kernel execution

#### **02_AppleNeuralEngine**
- **ConvolutionExample.cs** - Neural network operations on Apple Silicon
- Shows how to leverage the Apple Neural Engine for AI workloads

#### **03_IntelNPU**
- **ONNXInferenceExample.cs** - ONNX model inference on Intel NPU
- Demonstrates AI inference acceleration on Intel Neural Processing Units

#### **04_IntelAMX**
- **MatrixMultiplicationExample.cs** - Advanced Matrix Extensions usage
- High-performance matrix operations using Intel AMX instructions

#### **05_HardwareComparison**
- **AcceleratorSelectionExample.cs** - Compare performance across different accelerators
- Automatic hardware selection based on workload characteristics

#### **06_PerformanceBenchmarks**
- **BenchmarkingTutorial.cs** - Performance measurement and optimization
- Comprehensive benchmarking framework with real-time monitoring

#### **07_FFTOperations**
- **FFTFrameworkDemo.cs** - Fast Fourier Transform operations
- Shows integration with Intel IPP and CUDA cuFFT libraries

### Phase-Based Demonstrations

#### **Phase1-SimpleKernel**
- Foundation concepts and basic kernel development
- Entry point for understanding GPU programming fundamentals

#### **Phase8-UniversalKernels**
- Advanced universal kernel patterns
- Cross-platform kernel development with hardware abstraction

### Legacy ILGPU Samples

#### Basic Concepts
- **SimpleKernel** - Basic kernel execution patterns
- **SimpleAlloc** - Memory allocation fundamentals
- **SimpleViews** - Working with ArrayViews
- **SimpleStructures** - Using custom structures in kernels
- **SimpleMath** - Mathematical operations on GPU

#### Memory Management
- **SharedMemory** - Using shared memory for performance
- **DynamicSharedMemory** - Dynamic shared memory allocation
- **AdjustableSharedMemory** - Flexible shared memory configurations
- **MemoryBufferStrides** - Advanced memory access patterns
- **PinnedMemoryCopy** - Optimized host-device transfers

#### Advanced Features
- **ExplicitlyGroupedKernels** - Manual thread group configuration
- **ImplicitlyGroupedKernels** - Automatic grouping optimization
- **SpecializedKernel** - Kernel specialization techniques
- **GenericKernel** - Generic programming with kernels
- **CustomIntrinsics** - Custom hardware intrinsics

#### Synchronization and Atomics
- **SimpleAtomics** - Atomic operations basics
- **AdvancedAtomics** - Complex atomic patterns
- **WarpShuffle** - Warp-level operations
- **GroupOperations** - Thread group coordination

#### Algorithms Library
- **AlgorithmsReduce** - Parallel reduction operations
- **AlgorithmsScan** - Prefix sum operations
- **AlgorithmsRadixSort** - GPU-accelerated sorting
- **AlgorithmsHistogram** - Histogram computation
- **AlgorithmsOptimization** - Optimization algorithms
- **AlgorithmsCuBlas** - CUDA BLAS integration
- **AlgorithmsCuFFT** - CUDA FFT operations

#### Specialized Applications
- **Mandelbrot** - Interactive fractal visualization
- **MatrixMultiply** - Optimized matrix multiplication
- **BlazorSampleApp** - Web-based GPU computing with Blazor
- **MonitorProgress** - Real-time kernel progress monitoring
- **ProfilingMarkers** - Performance profiling integration

#### Platform-Specific
- **InlinePTXAssembly** - NVIDIA PTX inline assembly
- **LibDeviceKernel** - CUDA libdevice functions
- **AlgorithmsNvml** - NVIDIA Management Library integration
- **AlgorithmsNvJpeg** - GPU-accelerated JPEG processing

#### Language Features
- **SimpleFSharp** - F# language support
- **StaticAbstractInterfaceMembers** - C# 11 features
- **FixedSizeBuffers** - Fixed buffer support
- **InterleaveFields** - Field interleaving optimization

## 🖥️ Hardware Accelerator Examples

### CPU Accelerators
- Multi-threaded CPU execution
- Velocity SIMD optimizations
- Intel AMX matrix operations

### GPU Accelerators
- NVIDIA CUDA samples
- OpenCL cross-platform examples
- DirectCompute Windows samples

### Neural Processing Units
- Apple Neural Engine demos
- Intel NPU inference examples
- Custom AI accelerator integration

## 🏃 Running Samples

### Prerequisites
```bash
# Install .NET 9.0 with preview language features
dotnet --version
# Should show 9.0.x or higher

# Enable preview features for enhanced performance
export DOTNET_EnablePreviewFeatures=true

# Clone the repository
git clone https://github.com/mivertowski/UniversalCompute.git
cd UniversalCompute/Samples
```

### Building and Running
```bash
# Build all samples
dotnet build ILGPU.Samples.sln

# Run a specific sample
dotnet run --project SimpleKernel/SimpleKernel.csproj

# Run with specific configuration
dotnet run --project 01_GettingStarted/Examples.GettingStarted.csproj --configuration Release --framework=net9.0
```

### Hardware-Specific Samples
```bash
# Run CUDA-specific sample (requires NVIDIA GPU)
dotnet run --project AlgorithmsCuBlas/AlgorithmsCuBlas.csproj

# Run Apple Neural Engine sample (requires Apple Silicon)
dotnet run --project 02_AppleNeuralEngine/Examples.AppleNeuralEngine.csproj

# Run Intel NPU sample (requires Intel Core Ultra)
dotnet run --project 03_IntelNPU/Examples.IntelNPU.csproj
```

## 📝 Sample Documentation

Each sample includes:
- **README.md** - Detailed explanation of the sample
- **Inline Comments** - Code documentation explaining key concepts
- **Console Output** - Clear output showing results and performance metrics

## 🔧 Troubleshooting

### Common Issues

1. **Hardware Not Found**
   - Ensure required hardware is available (GPU, NPU, etc.)
   - Check driver installation and versions
   - Run DeviceInfo sample to diagnose

2. **Build Errors**
   - Verify .NET SDK version (.NET 9.0 with preview features required)
   - Enable preview features in your project files
   - Check NuGet package restoration
   - Ensure platform-specific dependencies are installed

3. **Runtime Errors**
   - Update graphics drivers to latest version
   - Check CUDA toolkit installation for NVIDIA samples
   - Verify OpenCL runtime for OpenCL samples

## 🤝 Contributing

We welcome contributions! To add a new sample:

1. Create a new folder with descriptive name
2. Add a .csproj file referencing UniversalCompute
3. Include a README.md explaining the sample
4. Add comprehensive inline documentation
5. Submit a pull request

### Sample Guidelines
- Keep samples focused on demonstrating specific features
- Include error handling and validation
- Provide performance measurements where relevant
- Support multiple hardware accelerators when possible
- Follow existing code style and organization

## 📚 Additional Resources

- [UniversalCompute Documentation](https://github.com/mivertowski/UniversalCompute/wiki)
- [API Reference](https://github.com/mivertowski/UniversalCompute/wiki/API-Reference)
- [Hardware Support Guide](https://github.com/mivertowski/UniversalCompute/wiki/Hardware-Accelerators)
- [Performance Optimization](https://github.com/mivertowski/UniversalCompute/wiki/Performance-Tuning)

## 📄 License

All samples are provided under the same license as UniversalCompute. See the root LICENSE.txt for details.

---

**Happy GPU Computing! 🚀**

For questions or support, please visit our [GitHub Discussions](https://github.com/mivertowski/UniversalCompute/discussions).