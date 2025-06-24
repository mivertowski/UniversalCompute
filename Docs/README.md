// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowsi/ILGPU/blob/main/LICENSE
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// NOTICE: This software is NOT licensed for commercial or production use.
// Change Date: 2029-06-24
// Change License: Apache License, Version 2.0

# ILGPU Tutorials

## Primers (How a GPU works)

This series introduces how a GPU works and what ILGPU does. If you have programmed with CUDA or OpenCL
before you can probably skip 01 and 02.

00 [Setting up ILGPU](01_Primers/01_Setting-Up-ILGPU.md) (ILGPU version 1.0.0)

01 [A GPU is not a CPU](01_Primers/01_Setting-Up-ILGPU.md) (ILGPU version 1.0.0)
> This page will provide a quick rundown the basics of how kernels (think GPU programs) run.

02 [Memory and bandwidth and threads. Oh my!](01_Primers/02_A-GPU-Is-Not-A-CPU.md)
> This will hopefully give you a better understanding of how memory works in hardware and the performance
> implications.

## Beginner (How ILGPU works)

This series is meant to be a brief overview of ILGPU and how to use it. It assumes you have at least a little knowledge
of how Cuda or OpenCL work.
If you need a primer look to something like [this for Cuda](https://developer.nvidia.com/about-cuda)
or [this for OpenCL](https://www.khronos.org/opencl/)

01 [Context and Accelerators](02_Beginner/01_Context-and-Accelerators.md)
> This tutorial covers creating the Context and Accelerator objects which setup ILGPU for use.
> It's mostly boiler plate and does no computation but it does print info about your GPU if you have one.
> There is some advice about ILGPU in here that makes it worth the quick read.
>
> See Also:
>
> [Device Info Sample](https://github.com/m4rs-mt/ILGPU/tree/master/Samples/DeviceInfo)

02 [MemoryBuffers and ArrayViews](02_Beginner/02_MemoryBuffers-and-ArrayViews.md)
> This tutorial covers the basics for Host / Device memory management.
>
> See Also:
>
> [Simple Allocation Sample](https://github.com/m4rs-mt/ILGPU/tree/master/Samples/SimpleAlloc)

03 [Kernels and Simple Programs](02_Beginner/03_Kernels-and-Simple-Programs.md)
> This is where it all comes together. This covers actual code, on the actual GPU (or the CPU if you are testing / dont
> have a GPU).
>
> See Also:
>
> [Simple Kernel Sample](https://github.com/m4rs-mt/ILGPU/tree/master/Samples/SimpleKernel)
>
> [Simple Math Sample](https://github.com/m4rs-mt/ILGPU/tree/master/Samples/SimpleMath)


04 [Structs and the N-body problem](02_Beginner/04_Structs.md)
> This tutorial actually does something! We use computing the N-body problem as a sample of how to better manage Host /
> Device memory.

## Beginner II (Something more interesting)

Well at least I think. This is where I will put ILGPUView bitmap shader things I (or other people if they want to)
eventually write. Below are the few I have planned / think would be easy.

1. Ray Tracing in One Weekend based raytracer
2. Cloud Simulation
3. 2D Physics Simulation
4. Other things I see on shadertoy

# Advanced Resources

## Samples

They cover a wide swath of uses for ILGPU including much of the more complex things that ILGPU is capable of.
[There are too many to list out so I will just link to the repository.](https://github.com/m4rs-mt/ILGPU/tree/master/Samples)

## Overview

[Memory Buffers & Views](03_Advanced/01_Memory-Buffers-and-Views.md)

[Kernels](03_Advanced/02_Kernels.md)

[Shared Memory](03_Advanced/03_Shared-Memory.md)

[Math Functions](03_Advanced/04_Math-Functions.md)

[Dynamically Specialized Kernels](03_Advanced/05_Dynamically-Specialized-Kernels.md)

[Debugging & Profiling](03_Advanced/06_Debugging-and-Profiling.md)

[Inside ILGPU](03_Advanced/07_Inside-ILGPU.md)

---

# 🚀 **ILGPU Evolution: 8-Phase Modernization Guide**

> **New in ILGPU 2.0+**: This section covers the revolutionary 8-phase evolution that transforms ILGPU from a traditional GPU library into the world's first Universal Compute Platform.

## Phase-by-Phase Evolution Guide

### **Phase 1-4: Foundation to Advanced GPU Programming**
- **[Phase 1: Foundation](05_Evolution/01_Foundation.md)** - Basic GPU programming with ILGPU
- **[Phase 2: Multi-Backend Architecture](05_Evolution/02_Multi-Backend.md)** - CPU, CUDA, OpenCL, Velocity support
- **[Phase 3: Advanced Memory Management](05_Evolution/03_Memory-Management.md)** - Hierarchical memory and optimization
- **[Phase 4: Advanced GPU Features](05_Evolution/04_GPU-Features.md)** - Warp-level programming and dynamic parallelism

### **Phase 5-8: Universal Computing Revolution**
- **[Phase 5: Cross-Platform Compatibility](05_Evolution/05_Cross-Platform.md)** - Universal deployment strategies
- **[Phase 6: Tensor Core Integration](05_Evolution/06_Tensor-Cores.md)** - AI/ML acceleration with .NET SIMD unification
- **[Phase 7: Emerging Platform Integration](05_Evolution/07_Emerging-Platforms.md)** - Apple Neural Engine, Intel NPU, quantum computing
- **[Phase 8: Universal Compute Platform](05_Evolution/08_Universal-Compute.md)** - Write-once, run-anywhere programming

## Universal Computing Features

### **🌟 Universal Kernels (Phase 8)**
```csharp
[UniversalKernel(SupportsMixedPrecision = true)]
[AppleOptimization(UseAMX = true, UseNeuralEngine = true)]
[IntelOptimization(UseAMX = true, UseNPU = true)]
[NvidiaOptimization(UseTensorCores = true)]
public static void UniversalMatrixMultiply(/* parameters */) 
{
    // Single kernel automatically optimizes for ALL hardware platforms
    // - Intel AMX/NPU: Matrix extensions + neural processing
    // - NVIDIA: Tensor Cores + cuBLAS optimizations  
    // - Apple: AMX + Neural Engine acceleration
    // - AMD: MFMA + ROCm BLAS optimizations
    // - CPU: SIMD vectorization (AVX-512, NEON, etc.)
}
```

### **🧠 Intelligent Memory Management (Phase 8)**
- **[Universal Memory Manager](06_Universal/01_Memory-Manager.md)** - Automatic optimal memory placement
- **[Adaptive Scheduling](06_Universal/02_Adaptive-Scheduling.md)** - Intelligent workload distribution
- **[Cross-Platform Coherency](06_Universal/03_Cross-Platform-Coherency.md)** - Seamless data movement

### **⚡ AI/ML Framework Integration (Phase 6-8)**
- **[ML.NET Acceleration](07_AI-ML/01_MLNet-Integration.md)** - Native ML.NET model acceleration
- **[ONNX Runtime Provider](07_AI-ML/02_ONNX-Integration.md)** - Industry-standard model execution
- **[Tensor Core Programming](07_AI-ML/03_Tensor-Core-Programming.md)** - Mixed precision AI acceleration
- **[Neural Engine Integration](07_AI-ML/04_Neural-Engine-Integration.md)** - Apple Silicon AI acceleration

## Specialized Computing

### **📊 Graph Analytics (CuGraph-Inspired)**
- **[Graph Processing Overview](08_Graph-Analytics/01_Overview.md)** - NVIDIA Rapids-inspired graph computing
- **[Centrality Algorithms](08_Graph-Analytics/02_Centrality.md)** - PageRank, Betweenness, Closeness
- **[Community Detection](08_Graph-Analytics/03_Community-Detection.md)** - Louvain, Label Propagation
- **[Traversal Algorithms](08_Graph-Analytics/04_Traversal.md)** - BFS, DFS, Shortest Paths

### **🔮 Emerging Technologies (Phase 7)**
- **[Quantum-Classical Hybrid](09_Emerging/01_Quantum-Classical.md)** - Hybrid quantum computing
- **[Neuromorphic Computing](09_Emerging/02_Neuromorphic.md)** - Brain-inspired computing
- **[Apple Silicon Advanced](09_Emerging/03_Apple-Silicon.md)** - Neural Engine and AMX programming
- **[Intel NPU Programming](09_Emerging/04_Intel-NPU.md)** - Dedicated AI acceleration

## Migration and Compatibility

### **🔄 Legacy to Universal Migration**
- **[Legacy Feature Mapping](10_Migration/01_Legacy-Mapping.md)** - Traditional ILGPU to Universal platform
- **[Performance Optimization Guide](10_Migration/02_Performance-Optimization.md)** - Maximizing universal performance
- **[Platform-Specific Tuning](10_Migration/03_Platform-Tuning.md)** - Hardware-specific optimizations
- **[Troubleshooting Guide](10_Migration/04_Troubleshooting.md)** - Common migration issues

### **📈 Performance and Benchmarking**
- **[Universal Benchmarking](11_Performance/01_Universal-Benchmarking.md)** - Cross-platform performance analysis
- **[Memory Optimization](11_Performance/02_Memory-Optimization.md)** - Advanced memory management techniques
- **[AI/ML Performance](11_Performance/03_AI-ML-Performance.md)** - Machine learning acceleration benchmarks
- **[Scalability Patterns](11_Performance/04_Scalability-Patterns.md)** - Multi-device and cluster scaling

---

## Legacy Documentation (ILGPU 1.x)

> **Note**: The following sections cover traditional ILGPU features. While still supported, consider migrating to Universal Computing features for optimal performance and future compatibility.

## Upgrade Guides

[Upgrade v0.1.X to v0.2.X](04_Upgrade-Guides/06_v0.1.X-to-v0.2.X.md)

[Upgrade v0.3.X to v0.5.X](04_Upgrade-Guides/05_v0.3.X-to-v0.5.X.md)

[Upgrade v0.6.X to v0.7.X](04_Upgrade-Guides/04_v0.6.X-to-v0.7.X.md)

[Upgrade v0.7.X to v0.8.X](04_Upgrade-Guides/03_v0.7.X-to-v0.8.X.md)

[Upgrade v0.8.0 to v0.8.1](04_Upgrade-Guides/02_v0.8.0-to-v0.8.1.md)

[Upgrade v0.8.X to v0.9.X](04_Upgrade-Guides/01_v0.8.X-to-v0.9.X.md)

**[Upgrade to Universal Compute Platform (v2.0+)](04_Upgrade-Guides/00_Legacy-to-Universal.md)** - Migration from traditional ILGPU to Universal Computing

---

## Quick Start for New Users

### **🎯 Recommended Learning Path**

1. **New to GPU Programming?** → Start with [Primers](01_Primers/) and [Beginner](02_Beginner/) sections
2. **Experienced with CUDA/OpenCL?** → Jump to [Phase 1: Foundation](05_Evolution/01_Foundation.md)
3. **Want Universal Computing?** → Begin with [Phase 8: Universal Compute Platform](05_Evolution/08_Universal-Compute.md)
4. **Need AI/ML Acceleration?** → Explore [Phase 6: Tensor Cores](05_Evolution/06_Tensor-Cores.md)
5. **Building for Multiple Platforms?** → Check [Phase 5: Cross-Platform](05_Evolution/05_Cross-Platform.md)

### **🚀 Universal Compute Quick Start**

```csharp
// Write-once, run-anywhere kernel that automatically optimizes for all hardware
[UniversalKernel]
static void ProcessData(ArrayView<float> data)
{
    var index = UniversalGrid.GlobalIndex.X;
    if (index < data.Length)
        data[index] = MathF.Sqrt(data[index] * 2.0f);
}

// Universal context automatically detects and optimizes for available hardware
using var context = Context.CreateDefault();
using var memoryManager = new UniversalMemoryManager(context);
using var scheduler = new AdaptiveScheduler(context.GetAvailableAccelerators());

// Automatically selects optimal device and memory placement
var result = await scheduler.ExecuteUniversalAsync(ProcessData, inputData);
```

**Experience the future of accelerated computing with ILGPU's Universal Compute Platform!** 🌟