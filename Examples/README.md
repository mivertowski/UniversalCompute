# ILGPU Hardware Accelerator Examples

This directory contains comprehensive examples and tutorials for using ILGPU's hardware accelerator capabilities, including Apple Neural Engine, Intel NPU, and Intel AMX support.

## 📁 Directory Structure

```
Examples/
├── 01_GettingStarted/          # Basic setup and first kernels
├── 02_AppleNeuralEngine/       # ANE-specific examples
├── 03_IntelNPU/               # NPU machine learning examples
├── 04_IntelAMX/               # AMX matrix operations
├── 05_HardwareComparison/     # Cross-platform comparisons
├── 06_PerformanceBenchmarks/  # Benchmarking tutorials
├── 07_AdvancedOptimization/   # Advanced techniques
└── Common/                    # Shared utilities and helpers
```

## 🚀 Quick Start

1. **Prerequisites**: Ensure you have the appropriate hardware and drivers:
   - Apple Neural Engine: Apple Silicon Mac (M1/M2/M3/M4)
   - Intel NPU: Intel Core Ultra processors (Meteor Lake+)
   - Intel AMX: Intel Xeon Sapphire Rapids or newer

2. **Basic Example**: Start with `01_GettingStarted/BasicAccelerator.cs`

3. **Hardware Detection**: Run the hardware detection example to see what's available on your system

## 📋 Examples Overview

### 🏁 Getting Started
- **BasicAccelerator**: Simple accelerator detection and setup
- **FirstKernel**: Your first ILGPU kernel on hardware accelerators
- **MemoryManagement**: Understanding unified memory systems

### 🧠 Apple Neural Engine
- **ModelInference**: Running Core ML models on ANE
- **ConvolutionOperations**: Optimized convolution kernels
- **AttentionMechanisms**: Transformer attention on ANE
- **PowerEfficiency**: Monitoring power consumption

### 🔧 Intel NPU
- **ONNXInference**: Running ONNX models on NPU
- **OpenVINOIntegration**: Using OpenVINO with ILGPU
- **QuantizedModels**: INT8 and BF16 quantization
- **BatchProcessing**: Optimal batch sizes for NPU

### ⚡ Intel AMX
- **MatrixMultiplication**: High-performance GEMM operations
- **TileOperations**: Working with AMX tiles
- **DataTypes**: BF16, INT8, and FP32 operations
- **OptimalTiling**: Calculating optimal tile configurations

### 📊 Hardware Comparison
- **PerformanceComparison**: Benchmarking across accelerators
- **PowerEfficiency**: Comparing power consumption
- **AccuracyAnalysis**: Precision differences between backends
- **WorkloadSelection**: Choosing the right accelerator

### 🎯 Performance Optimization
- **MemoryOptimization**: Minimizing memory transfers
- **KernelFusion**: Combining operations for efficiency
- **AsyncExecution**: Overlapping computation and data transfer
- **ProfilingTools**: Understanding performance bottlenecks

## 🔧 Building and Running

```bash
# Build all examples
dotnet build Examples.sln

# Run a specific example
dotnet run --project 01_GettingStarted/BasicAccelerator

# Run with hardware detection
dotnet run --project Common/HardwareDetection
```

## 📖 Learning Path

### Beginner
1. Start with `01_GettingStarted/BasicAccelerator.cs`
2. Learn hardware detection with `Common/HardwareDetection.cs`
3. Try simple kernels in `01_GettingStarted/FirstKernel.cs`

### Intermediate
1. Explore hardware-specific examples (ANE, NPU, AMX)
2. Compare performance across accelerators
3. Learn memory optimization techniques

### Advanced
1. Implement custom optimization strategies
2. Build hybrid workload orchestration
3. Develop production-ready applications

## 🛠️ Prerequisites

### Software Requirements
- .NET 8.0 or later
- ILGPU 2.0.0-beta1 or later

### Hardware Requirements
- **Apple Neural Engine**: macOS 11.0+ on Apple Silicon
- **Intel NPU**: Windows 11 or Linux with OpenVINO runtime
- **Intel AMX**: Intel Xeon (Sapphire Rapids+) or Core (Alder Lake+)

### Driver Requirements
- **ANE**: Xcode Command Line Tools
- **NPU**: Intel GPU drivers and OpenVINO runtime
- **AMX**: Latest Intel CPU microcode

## 📚 Additional Resources

- [ILGPU Documentation](../Docs/)
- [Hardware Requirements](../HardwareRequirements.md)
- [Benchmark Results](../BenchmarkResults/)
- [API Reference](../Docs/99_Technical-Reference/)

## 🤝 Contributing

Found an issue or want to add an example? Please see our [contribution guidelines](../CONTRIBUTING.md).

## 📄 License

This project is licensed under the Business Source License 1.1 - see the [LICENSE](../LICENSE) file for details.