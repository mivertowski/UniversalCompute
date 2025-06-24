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

# Universal Memory Manager

## Overview

The Universal Memory Manager provides intelligent, cross-platform memory allocation and management for ILGPU applications. This system automatically optimizes memory placement and access patterns across different accelerator types while maintaining a unified programming interface.

## Technical Background

### Memory Management Challenges

Modern heterogeneous computing systems present complex memory management requirements:

1. **Memory Hierarchy Diversity**: Different accelerators have varying memory architectures
2. **Performance Optimization**: Optimal memory placement varies by hardware and workload
3. **Cross-Platform Compatibility**: Applications must work efficiently across multiple backends
4. **Resource Management**: Memory allocation and deallocation must be coordinated across devices

### Traditional ILGPU Memory Management

Legacy ILGPU requires manual memory management per accelerator:

```csharp
// Traditional approach - manual per-accelerator management
using var context = Context.CreateDefault();
using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

// Manual allocation and explicit management
using var buffer = accelerator.Allocate1D<float>(1024);
buffer.CopyFromCPU(hostData);

// Explicit synchronization required
accelerator.Synchronize();
```

### Universal Memory Manager Solution

The Universal Memory Manager abstracts memory management complexity:

```csharp
// Universal approach - automatic optimization
using var context = Context.CreateDefault();
using var memoryManager = new UniversalMemoryManager(context);

// Intelligent allocation with optimization hints
using var buffer = memoryManager.AllocateUniversal<float>(1024, 
    MemoryPlacement.Auto, AccessPattern.Sequential);

// Automatic data transfer and synchronization
await buffer.CopyFromAsync(hostData);
```

## Core Architecture

### 1. Memory Placement Optimization

#### Placement Strategies

The system supports multiple memory placement strategies:

```csharp
public enum MemoryPlacement
{
    Auto,                    // System-selected optimal placement
    HostMemory,             // System RAM allocation
    DeviceMemory,           // Accelerator-specific memory
    UnifiedMemory,          // Shared CPU/GPU memory (where supported)
    PinnedMemory,           // Page-locked host memory
    CacheOptimized,         // Optimized for cache performance
    BandwidthOptimized      // Optimized for memory bandwidth
}
```

#### Access Pattern Analysis

Memory allocation considers expected access patterns:

```csharp
public enum AccessPattern
{
    Unknown,        // No specific pattern
    Sequential,     // Linear access (stride 1)
    Strided,        // Regular stride access
    Random,         // Unpredictable access
    Streaming,      // One-time sequential access
    ReadOnly,       // Read-only access
    WriteOnly,      // Write-only access
    ReadWrite       // Mixed read/write access
}
```

#### Implementation Example

```csharp
public class UniversalMemoryManager : IDisposable
{
    private readonly Context context;
    private readonly MemoryPlacementOptimizer optimizer;
    private readonly Dictionary<AcceleratorType, IAccelerator> accelerators;

    public UniversalMemoryManager(Context context)
    {
        this.context = context ?? throw new ArgumentNullException(nameof(context));
        this.optimizer = new MemoryPlacementOptimizer();
        this.accelerators = DiscoverAvailableAccelerators();
    }

    public IUniversalBuffer<T> AllocateUniversal<T>(
        long size,
        MemoryPlacement placement = MemoryPlacement.Auto,
        AccessPattern accessPattern = AccessPattern.Unknown) where T : unmanaged
    {
        if (size <= 0)
            throw new ArgumentException("Size must be positive", nameof(size));

        var optimalAccelerator = placement == MemoryPlacement.Auto
            ? optimizer.SelectOptimalAccelerator(size, accessPattern, accelerators.Values)
            : SelectAcceleratorByPlacement(placement);

        return new UniversalBuffer<T>(optimalAccelerator, size, placement, accessPattern);
    }

    private Dictionary<AcceleratorType, IAccelerator> DiscoverAvailableAccelerators()
    {
        var discovered = new Dictionary<AcceleratorType, IAccelerator>();

        // Discover CUDA accelerators
        if (context.GetCudaDevices().Any())
        {
            discovered[AcceleratorType.Cuda] = context.CreateCudaAccelerator(0);
        }

        // Discover OpenCL accelerators
        if (context.GetOpenCLDevices().Any())
        {
            discovered[AcceleratorType.OpenCL] = context.CreateOpenCLAccelerator(0);
        }

        // CPU accelerator always available
        discovered[AcceleratorType.CPU] = context.CreateCPUAccelerator(0);

        return discovered;
    }
}
```

### 2. Universal Buffer Interface

#### IUniversalBuffer Contract

```csharp
public interface IUniversalBuffer<T> : IDisposable where T : unmanaged
{
    // Basic properties
    long Length { get; }
    MemoryPlacement CurrentPlacement { get; }
    AccessPattern AccessPattern { get; }
    bool IsDisposed { get; }

    // Memory views
    ArrayView<T> GetView(IAccelerator accelerator);
    ArrayView1D<T, Stride1D.Dense> GetView1D();
    ArrayView2D<T, Stride2D.DenseX> GetView2D(long width, long height);

    // Data transfer operations
    void CopyFromCPU(ReadOnlySpan<T> data);
    Task CopyFromAsync(ReadOnlySpan<T> data);
    T[] GetAsArray1D();
    Task<T[]> GetAsArrayAsync();

    // Cross-platform operations
    Task MoveToAcceleratorAsync(IAccelerator targetAccelerator);
    void EnsureCoherency();
    Task EnsureCoherencyAsync();

    // Performance monitoring
    MemoryStatistics GetStatistics();
}
```

#### Implementation Example

```csharp
public class UniversalBuffer<T> : IUniversalBuffer<T> where T : unmanaged
{
    private readonly IAccelerator accelerator;
    private readonly MemoryBuffer1D<T, Stride1D.Dense> buffer;
    private readonly MemoryPlacement placement;
    private readonly AccessPattern accessPattern;
    private bool disposed = false;

    public UniversalBuffer(IAccelerator accelerator, long size, 
        MemoryPlacement placement, AccessPattern accessPattern)
    {
        this.accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        this.placement = placement;
        this.accessPattern = accessPattern;
        
        // Allocate buffer based on placement strategy
        this.buffer = AllocateBuffer(size);
    }

    public long Length => buffer.Length;
    public MemoryPlacement CurrentPlacement => placement;
    public AccessPattern AccessPattern => accessPattern;
    public bool IsDisposed => disposed;

    public ArrayView<T> GetView(IAccelerator targetAccelerator)
    {
        if (disposed)
            throw new ObjectDisposedException(nameof(UniversalBuffer<T>));
        
        // Return optimized view for target accelerator
        return buffer.View;
    }

    public void CopyFromCPU(ReadOnlySpan<T> data)
    {
        if (disposed)
            throw new ObjectDisposedException(nameof(UniversalBuffer<T>));
        if (data.Length > Length)
            throw new ArgumentException("Data size exceeds buffer capacity");

        buffer.CopyFromCPU(data);
    }

    public async Task CopyFromAsync(ReadOnlySpan<T> data)
    {
        // Asynchronous data transfer implementation
        await Task.Run(() => CopyFromCPU(data));
    }

    private MemoryBuffer1D<T, Stride1D.Dense> AllocateBuffer(long size)
    {
        return placement switch
        {
            MemoryPlacement.HostMemory => accelerator.Allocate1D<T>(size),
            MemoryPlacement.DeviceMemory => accelerator.Allocate1D<T>(size),
            MemoryPlacement.PinnedMemory => accelerator.Allocate1D<T>(size), // Page-locked allocation
            _ => accelerator.Allocate1D<T>(size)
        };
    }

    public void Dispose()
    {
        if (!disposed)
        {
            buffer?.Dispose();
            disposed = true;
        }
    }
}
```

### 3. Platform-Specific Optimizations

#### CUDA Memory Optimization

```csharp
public class CudaUniversalBuffer<T> : UniversalBuffer<T> where T : unmanaged
{
    private readonly CudaAccelerator cudaAccelerator;
    private bool useUnifiedMemory;
    private bool useManagedMemory;

    public CudaUniversalBuffer(CudaAccelerator accelerator, long size, 
        MemoryPlacement placement, AccessPattern accessPattern)
        : base(accelerator, size, placement, accessPattern)
    {
        this.cudaAccelerator = accelerator;
        this.useUnifiedMemory = ShouldUseUnifiedMemory(size, accessPattern);
        this.useManagedMemory = ShouldUseManagedMemory(size, accessPattern);
        
        OptimizeForCuda();
    }

    private void OptimizeForCuda()
    {
        if (useUnifiedMemory && cudaAccelerator.Device.SupportsUnifiedMemory)
        {
            // Enable unified memory optimizations
            ConfigureUnifiedMemory();
        }

        if (useManagedMemory)
        {
            // Configure managed memory settings
            ConfigureManagedMemory();
        }

        // Set memory advice for optimal performance
        SetMemoryAdvice();
    }

    private bool ShouldUseUnifiedMemory(long size, AccessPattern pattern)
    {
        // Unified memory beneficial for larger allocations with irregular access
        return size > 1024 * 1024 && pattern == AccessPattern.Random;
    }

    private bool ShouldUseManagedMemory(long size, AccessPattern pattern)
    {
        // Managed memory beneficial for read-heavy workloads
        return pattern == AccessPattern.ReadOnly || pattern == AccessPattern.Streaming;
    }
}
```

#### Apple Metal Optimization

```csharp
public class MetalUniversalBuffer<T> : UniversalBuffer<T> where T : unmanaged
{
    private readonly MetalAccelerator metalAccelerator;
    private readonly bool isAppleSilicon;

    public MetalUniversalBuffer(MetalAccelerator accelerator, long size,
        MemoryPlacement placement, AccessPattern accessPattern)
        : base(accelerator, size, placement, accessPattern)
    {
        this.metalAccelerator = accelerator;
        this.isAppleSilicon = DetectAppleSilicon();
        
        OptimizeForMetal();
    }

    private void OptimizeForMetal()
    {
        if (isAppleSilicon)
        {
            // Use unified memory architecture on Apple Silicon
            ConfigureUnifiedMemoryArchitecture();
        }
        else
        {
            // Use discrete memory model on Intel Macs
            ConfigureDiscreteMemoryModel();
        }
    }

    private bool DetectAppleSilicon()
    {
        // Implementation to detect Apple Silicon vs Intel architecture
        return Environment.OSVersion.Platform == PlatformID.Unix && 
               RuntimeInformation.ProcessArchitecture == Architecture.Arm64;
    }
}
```

## Performance Monitoring

### Memory Statistics

```csharp
public class MemoryStatistics
{
    public long TotalAllocatedBytes { get; set; }
    public long PeakAllocatedBytes { get; set; }
    public long ActiveAllocations { get; set; }
    public double MemoryEfficiency { get; set; }
    public double BandwidthUtilization { get; set; }
    public TimeSpan AverageAllocationTime { get; set; }
    public TimeSpan AverageTransferTime { get; set; }
    public Dictionary<MemoryPlacement, long> AllocationsByPlacement { get; set; }
    public Dictionary<AcceleratorType, long> AllocationsByAccelerator { get; set; }
}

// Usage example
public void MonitorMemoryUsage(UniversalMemoryManager memoryManager)
{
    var statistics = memoryManager.GetMemoryStatistics();
    
    Console.WriteLine($"Total Memory Allocated: {statistics.TotalAllocatedBytes / (1024 * 1024)} MB");
    Console.WriteLine($"Peak Memory Usage: {statistics.PeakAllocatedBytes / (1024 * 1024)} MB");
    Console.WriteLine($"Memory Efficiency: {statistics.MemoryEfficiency:P1}");
    Console.WriteLine($"Bandwidth Utilization: {statistics.BandwidthUtilization:P1}");
    
    foreach (var allocation in statistics.AllocationsByAccelerator)
    {
        Console.WriteLine($"{allocation.Key}: {allocation.Value / (1024 * 1024)} MB");
    }
}
```

### Performance Optimization

```csharp
public class MemoryOptimizer
{
    public static OptimizationRecommendations AnalyzePerformance(
        UniversalMemoryManager memoryManager,
        TimeSpan measurementPeriod)
    {
        var recommendations = new OptimizationRecommendations();
        var statistics = memoryManager.GetMemoryStatistics();

        // Analyze memory efficiency
        if (statistics.MemoryEfficiency < 0.7)
        {
            recommendations.Add(new Recommendation
            {
                Type = RecommendationType.MemoryPooling,
                Description = "Consider implementing memory pooling for frequent allocations",
                PotentialImprovement = "20-30% reduction in allocation overhead"
            });
        }

        // Analyze bandwidth utilization
        if (statistics.BandwidthUtilization < 0.5)
        {
            recommendations.Add(new Recommendation
            {
                Type = RecommendationType.AccessPatternOptimization,
                Description = "Optimize memory access patterns for better bandwidth utilization",
                PotentialImprovement = "15-25% improvement in memory throughput"
            });
        }

        return recommendations;
    }
}
```

## Usage Examples

### Basic Memory Allocation

```csharp
using System;
using ILGPU;
using ILGPU.Runtime;

class MemoryExample
{
    static void BasicAllocation()
    {
        using var context = Context.CreateDefault();
        using var memoryManager = new UniversalMemoryManager(context);

        // Allocate memory with optimization hints
        using var buffer = memoryManager.AllocateUniversal<float>(1024,
            MemoryPlacement.Auto,
            AccessPattern.Sequential);

        // Initialize data
        var inputData = new float[1024];
        for (int i = 0; i < inputData.Length; i++)
        {
            inputData[i] = i * 0.1f;
        }

        // Transfer data to device
        buffer.CopyFromCPU(inputData);

        // Ensure data is available on accelerator
        buffer.EnsureCoherency();

        Console.WriteLine($"Allocated {buffer.Length} elements with {buffer.CurrentPlacement} placement");
    }
}
```

### Advanced Memory Management

```csharp
class AdvancedMemoryExample
{
    static async Task AdvancedAllocation()
    {
        using var context = Context.CreateDefault();
        using var memoryManager = new UniversalMemoryManager(context);

        const int matrixSize = 512;
        const int elementCount = matrixSize * matrixSize;

        // Allocate matrix with specific optimization
        using var matrixA = memoryManager.AllocateUniversal<float>(elementCount,
            MemoryPlacement.DeviceMemory,
            AccessPattern.Sequential);

        using var matrixB = memoryManager.AllocateUniversal<float>(elementCount,
            MemoryPlacement.DeviceMemory,
            AccessPattern.Sequential);

        using var result = memoryManager.AllocateUniversal<float>(elementCount,
            MemoryPlacement.DeviceMemory,
            AccessPattern.WriteOnly);

        // Initialize matrices
        var dataA = new float[elementCount];
        var dataB = new float[elementCount];
        var random = new Random(42);

        for (int i = 0; i < elementCount; i++)
        {
            dataA[i] = (float)random.NextDouble();
            dataB[i] = (float)random.NextDouble();
        }

        // Asynchronous data transfers
        await Task.WhenAll(
            matrixA.CopyFromAsync(dataA),
            matrixB.CopyFromAsync(dataB)
        );

        // Matrices are now ready for computation
        Console.WriteLine($"Matrices allocated and initialized: {matrixSize}x{matrixSize}");

        // Monitor memory usage
        var stats = memoryManager.GetMemoryStatistics();
        Console.WriteLine($"Total allocated: {stats.TotalAllocatedBytes / (1024 * 1024)} MB");
    }
}
```

## Best Practices

### Memory Allocation

```csharp
// Recommended: Specify access patterns for optimization
using var buffer = memoryManager.AllocateUniversal<float>(
    size: dataSize,
    placement: MemoryPlacement.Auto,
    accessPattern: AccessPattern.Sequential  // Provides optimization hints
);

// Recommended: Use appropriate buffer lifetimes
using (var temporaryBuffer = memoryManager.AllocateUniversal<float>(tempSize))
{
    // Use buffer for short-lived operations
    ProcessTemporaryData(temporaryBuffer);
} // Automatic cleanup

// Avoid: Long-lived buffers without explicit management
var persistentBuffer = memoryManager.AllocateUniversal<float>(size);
// Remember to dispose explicitly when no longer needed
```

### Error Handling

```csharp
try
{
    using var buffer = memoryManager.AllocateUniversal<float>(largeSize);
    buffer.CopyFromCPU(hostData);
}
catch (OutOfMemoryException ex)
{
    // Handle memory allocation failures
    Console.WriteLine($"Memory allocation failed: {ex.Message}");
    // Implement fallback strategy or reduce allocation size
}
catch (InvalidOperationException ex)
{
    // Handle invalid operations
    Console.WriteLine($"Invalid operation: {ex.Message}");
}
```

### Performance Optimization

```csharp
// Monitor and optimize memory usage
public void OptimizeMemoryUsage(UniversalMemoryManager memoryManager)
{
    var stats = memoryManager.GetMemoryStatistics();
    
    if (stats.MemoryEfficiency < 0.8)
    {
        // Consider memory pooling or different allocation strategies
        memoryManager.EnableMemoryPooling(poolSizeHint: 64 * 1024 * 1024);
    }

    if (stats.BandwidthUtilization < 0.6)
    {
        // Analyze access patterns and consider batching operations
        Console.WriteLine("Consider optimizing memory access patterns");
    }
}
```

## Migration from Legacy ILGPU

### Traditional Memory Management

```csharp
// Legacy ILGPU approach
using var context = Context.CreateDefault();
using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);
using var buffer = accelerator.Allocate1D<float>(1024);
buffer.CopyFromCPU(hostData);
accelerator.Synchronize();
```

### Universal Memory Management

```csharp
// Universal Memory Manager approach
using var context = Context.CreateDefault();
using var memoryManager = new UniversalMemoryManager(context);
using var buffer = memoryManager.AllocateUniversal<float>(1024);
await buffer.CopyFromAsync(hostData);
// Automatic synchronization and optimization
```

### Migration Steps

1. **Replace direct accelerator allocations** with Universal Memory Manager calls
2. **Add access pattern hints** to improve optimization
3. **Use asynchronous operations** where possible for better performance
4. **Monitor memory usage** and adjust allocation strategies based on statistics

## Limitations and Considerations

### Current Limitations

1. **Memory Overhead**: Universal buffers may consume additional memory for cross-platform compatibility
2. **Initialization Cost**: First allocation may take longer due to accelerator discovery and optimization setup
3. **Platform Availability**: Some optimizations are only available on specific hardware platforms

### Performance Considerations

1. **Access Pattern Accuracy**: Incorrect access pattern hints may reduce optimization effectiveness
2. **Memory Placement**: Automatic placement decisions may not always be optimal for specific workloads
3. **Synchronization Overhead**: Cross-platform coherency operations may introduce latency

### Future Enhancements

1. **Machine Learning Optimization**: Automatic optimization based on historical usage patterns
2. **Extended Platform Support**: Additional optimization strategies for emerging accelerator types
3. **Improved Profiling**: Enhanced memory usage analysis and recommendation systems

---

The Universal Memory Manager provides a robust foundation for efficient memory management across diverse accelerator architectures while maintaining enterprise-grade reliability and performance characteristics.