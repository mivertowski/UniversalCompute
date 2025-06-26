# Quick Start Tutorial

Get up and running with UniversalCompute in 15 minutes! This tutorial will guide you through creating your first high-performance compute application.

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ [.NET 8.0+ SDK installed](https://dotnet.microsoft.com/download)
- ✅ [UniversalCompute package installed](Installation-Guide)
- ✅ A compatible IDE (Visual Studio, Rider, or VS Code)
- ✅ [Optional] CUDA-compatible GPU for maximum performance

## 🎯 What We'll Build

We'll create a comprehensive application that demonstrates:
1. **Dependency injection setup** for clean architecture
2. **Hardware accelerator detection** and automatic selection
3. **Modern kernel development** with attributes and source generation
4. **Unified memory management** with automatic optimization
5. **Performance monitoring** with real-time metrics
6. **Vector addition computation** across different accelerators
7. **Advanced features** like FFT operations and tensor math

## 🚀 Step 1: Create New Project

```bash
# Create a new console application
dotnet new console -n UniversalComputeDemo
cd UniversalComputeDemo

# Add UniversalCompute packages
dotnet add package UniversalCompute --version 1.0.0-alpha1
dotnet add package UniversalCompute.DependencyInjection --version 1.0.0-alpha1
dotnet add package Microsoft.Extensions.Hosting --version 8.0.0
dotnet add package Microsoft.Extensions.Logging.Console --version 8.0.0

# Enable source generators and unsafe code
```

Edit your `.csproj` file to enable source generation and unsafe code:
```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
    <Nullable>enable</Nullable>
    <EnableNETAnalyzers>true</EnableNETAnalyzers>
    <!-- Enable source generation for automatic kernel discovery -->
    <EmitCompilerGeneratedFiles>true</EmitCompilerGeneratedFiles>
    <CompilerGeneratedFilesOutputPath>Generated</CompilerGeneratedFilesOutputPath>
  </PropertyGroup>
  
  <ItemGroup>
    <PackageReference Include="UniversalCompute" Version="1.0.0-alpha1" />
    <PackageReference Include="UniversalCompute.DependencyInjection" Version="1.0.0-alpha1" />
    <PackageReference Include="Microsoft.Extensions.Hosting" Version="8.0.0" />
    <PackageReference Include="Microsoft.Extensions.Logging.Console" Version="8.0.0" />
  </ItemGroup>
</Project>
```

## 🎯 Step 2: Dependency Injection Setup

Create a new file `ComputeService.cs` to demonstrate modern service-based architecture:

```csharp
using Microsoft.Extensions.Logging;
using UniversalCompute;
using UniversalCompute.Memory.Unified;
using UniversalCompute.Runtime.Scheduling;
using System.Diagnostics;

namespace UniversalComputeDemo;

public class ComputeService
{
    private readonly ILogger<ComputeService> _logger;
    private readonly Context _context;
    private readonly UniversalMemoryManager _memoryManager;
    private readonly PerformanceMonitor _performanceMonitor;

    public ComputeService(
        ILogger<ComputeService> logger,
        Context context,
        UniversalMemoryManager memoryManager,
        PerformanceMonitor performanceMonitor)
    {
        _logger = logger;
        _context = context;
        _memoryManager = memoryManager;
        _performanceMonitor = performanceMonitor;
    }

    // Define vector addition kernel with modern attributes
    [KernelMethod]
    [OptimizedKernel(OptimizationLevel.Maximum)]
    static void VectorAddKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
    {
        result[index] = a[index] + b[index];
    }

    public async Task RunDemoAsync()
    {
        _logger.LogInformation("🚀 UniversalCompute Quick Start Demo");
        _logger.LogInformation("===================================");

        // Display available accelerators
        await DisplayHardwareInfoAsync();

        // Test specialized accelerators
        await TestSpecializedAcceleratorsAsync();

        // Run performance comparison with monitoring
        await RunPerformanceComparisonAsync();

        _logger.LogInformation("🎉 Demo completed successfully!");
    }

    private async Task DisplayHardwareInfoAsync()
    {
        _logger.LogInformation("📊 Available Hardware Accelerators:");
        _logger.LogInformation("-----------------------------------");

        foreach (var device in _context.Devices)
        {
            _logger.LogInformation("✅ {DeviceName}", device.Name);
            _logger.LogInformation("   Type: {AcceleratorType}", device.AcceleratorType);
            _logger.LogInformation("   Memory: {Memory:N0} MB", device.MemorySize / (1024 * 1024));
            _logger.LogInformation("   Compute Units: {ComputeUnits}", device.NumMultiProcessors);
            _logger.LogInformation("");
        }
    }

    // Additional methods will be defined below...
}
```

Now update `Program.cs` to use dependency injection:

```csharp
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using UniversalCompute;
using UniversalCompute.DependencyInjection;
using UniversalCompute.Memory.Unified;
using UniversalCompute.Runtime.Scheduling;
using UniversalComputeDemo;

// Create host with dependency injection
var builder = Host.CreateApplicationBuilder(args);

// Configure logging
builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.SetMinimumLevel(LogLevel.Information);

// Register UniversalCompute services
builder.Services.AddUniversalCompute(options =>
{
    options.EnableAllAccelerators();
    options.EnableUnifiedMemoryManagement();
    options.EnablePerformanceMonitoring();
    options.EnableSourceGeneration();
});

// Register application services
builder.Services.AddScoped<ComputeService>();

// Build and run
using var host = builder.Build();

// Get the compute service and run demo
var computeService = host.Services.GetRequiredService<ComputeService>();
await computeService.RunDemoAsync();

Console.WriteLine("Press any key to exit...");
Console.ReadKey();
```

## 🔧 Step 3: Specialized Accelerator Testing with Source Generation

Add these methods to your `ComputeService.cs` to test hardware-specific accelerators:

```csharp
private async Task TestSpecializedAcceleratorsAsync()
{
    _logger.LogInformation("🔍 Testing Specialized Accelerators:");
    _logger.LogInformation("-----------------------------------");

    // Test Intel AMX with performance monitoring
    await TestAcceleratorAsync("Intel AMX", () => _context.CreateAMXAccelerator());

    // Test Intel NPU
    await TestAcceleratorAsync("Intel NPU", () => _context.CreateNPUAccelerator());

    // Test Apple Neural Engine
    await TestAcceleratorAsync("Apple Neural Engine", () => _context.CreateANEAccelerator());

    _logger.LogInformation("");
}

private async Task TestAcceleratorAsync(string name, Func<Accelerator> acceleratorFactory)
{
    try
    {
        using var session = _performanceMonitor.BeginSession($"{name}_test");
        using var accelerator = acceleratorFactory();
        
        _logger.LogInformation("✅ {AcceleratorName}: Available", name);
        _logger.LogInformation("   Accelerator: {AcceleratorFullName}", accelerator.Name);
        _logger.LogInformation("   Max threads per group: {MaxThreads}", accelerator.MaxNumThreadsPerGroup);
        
        // Get memory statistics from unified memory manager
        var memoryStats = _memoryManager.GetMemoryStatistics(accelerator);
        _logger.LogInformation("   Available memory: {AvailableMemory:N0} MB", memoryStats.AvailableMemory / (1024 * 1024));
        _logger.LogInformation("   Used memory: {UsedMemory:N0} MB", memoryStats.UsedMemory / (1024 * 1024));
        
        session.RecordMetric("accelerator_detection", 1.0);
    }
    catch (Exception ex)
    {
        _logger.LogWarning("❌ {AcceleratorName}: Not available ({ErrorMessage})", 
            name, ex.Message.Split('.')[0]);
    }
}
```

## ⚡ Step 4: Performance Comparison with Unified Memory

Add the modern performance comparison method to your `ComputeService.cs`:

```csharp
private async Task RunPerformanceComparisonAsync()
{
    _logger.LogInformation("⚡ Performance Comparison: Vector Addition");
    _logger.LogInformation("=========================================");

    const int vectorSize = 10_000_000; // 10 million elements
    const int iterations = 5;

    using var globalSession = _performanceMonitor.BeginSession("vector_addition_benchmark");

    // Prepare test data using unified memory manager
    var placement = MemoryPlacement.PreferDevice;
    using var unifiedA = _memoryManager.AllocateUnified<float>(vectorSize, placement);
    using var unifiedB = _memoryManager.AllocateUnified<float>(vectorSize, placement);
    using var unifiedResult = _memoryManager.AllocateUnified<float>(vectorSize, placement);

    // Initialize with random data efficiently
    await InitializeTestDataAsync(unifiedA, unifiedB, vectorSize);

    _logger.LogInformation("Vector size: {VectorSize:N0} elements", vectorSize);
    _logger.LogInformation("Iterations: {Iterations}", iterations);
    _logger.LogInformation("Memory placement: {Placement}", placement);
    _logger.LogInformation("");

    // Test CPU performance first
    await TestCPUPerformanceAsync(unifiedA, unifiedB, unifiedResult, iterations);

    // Test each available accelerator with automatic kernel discovery
    foreach (var device in _context.Devices)
    {
        try
        {
            using var accelerator = device.CreateAccelerator(_context);
            await TestAcceleratorPerformanceAsync(accelerator, unifiedA, unifiedB, unifiedResult, iterations);
        }
        catch (Exception ex)
        {
            _logger.LogWarning("❌ {DeviceName}: Failed ({ErrorMessage})", 
                device.Name, ex.Message.Split('.')[0]);
        }
    }

    // Display aggregated performance metrics
    var metrics = globalSession.GetMetrics();
    _logger.LogInformation("📊 Performance Summary:");
    foreach (var metric in metrics)
    {
        _logger.LogInformation("   {MetricName}: {MetricValue:F2}", metric.Key, metric.Value);
    }
}

private async Task InitializeTestDataAsync(IUnifiedBuffer<float> bufferA, IUnifiedBuffer<float> bufferB, int size)
{
    using var session = _performanceMonitor.BeginSession("data_initialization");
    
    var random = new Random(42);
    var dataA = new float[size];
    var dataB = new float[size];
    
    // Vectorized initialization for better performance
    Parallel.For(0, size, i =>
    {
        dataA[i] = random.NextSingle() * 100;
        dataB[i] = random.NextSingle() * 100;
    });

    // Copy to unified memory with automatic placement optimization
    await bufferA.CopyFromAsync(dataA);
    await bufferB.CopyFromAsync(dataB);
    
    session.RecordMetric("data_initialization_time", session.ElapsedMilliseconds);
}
```

## 🧮 Step 5: CPU Baseline Performance with Monitoring

Add the enhanced CPU performance test to your `ComputeService.cs`:

```csharp
private async Task TestCPUPerformanceAsync(
    IUnifiedBuffer<float> bufferA, 
    IUnifiedBuffer<float> bufferB, 
    IUnifiedBuffer<float> bufferResult, 
    int iterations)
{
    _logger.LogInformation("🖥️  CPU Baseline (Vectorized):");
    
    using var session = _performanceMonitor.BeginSession("cpu_baseline");
    
    // Get data from unified buffers (automatic optimization based on current location)
    var dataA = await bufferA.ToArrayAsync();
    var dataB = await bufferB.ToArrayAsync();
    var result = new float[dataA.Length];
    
    var stopwatch = Stopwatch.StartNew();
    
    for (int iter = 0; iter < iterations; iter++)
    {
        // Use vectorized operations for better CPU performance
        Parallel.For(0, dataA.Length, i =>
        {
            result[i] = dataA[i] + dataB[i];
        });
    }
    
    stopwatch.Stop();
    
    // Copy result back to unified buffer
    await bufferResult.CopyFromAsync(result);
    
    var avgTime = stopwatch.ElapsedMilliseconds / (double)iterations;
    var throughput = (dataA.Length * sizeof(float) * 3) / (avgTime / 1000.0) / (1024 * 1024 * 1024); // GB/s
    
    _logger.LogInformation("   Average time: {AvgTime:F2} ms", avgTime);
    _logger.LogInformation("   Throughput: {Throughput:F2} GB/s", throughput);
    _logger.LogInformation("   Threads used: {ThreadCount}", Environment.ProcessorCount);
    
    // Record performance metrics
    session.RecordMetric("cpu_time_ms", avgTime);
    session.RecordMetric("cpu_throughput_gbps", throughput);
    session.RecordMetric("cpu_threads", Environment.ProcessorCount);
    
    // Verify correctness
    var isCorrect = await VerifyResultsAsync(dataA, dataB, result);
    _logger.LogInformation("   Correctness: {Correctness}", isCorrect ? "✅ Verified" : "❌ Failed");
    _logger.LogInformation("");
}

private async Task<bool> VerifyResultsAsync(float[] a, float[] b, float[] result)
{
    return await Task.Run(() =>
    {
        for (int i = 0; i < Math.Min(100, result.Length); i++)
        {
            var expected = a[i] + b[i];
            if (Math.Abs(result[i] - expected) > 1e-5f)
            {
                return false;
            }
        }
        return true;
    });
}
```

## 🚀 Step 6: Accelerator Performance with Source Generation

Add the modern accelerator performance test to your `ComputeService.cs`. Note that the `[KernelMethod]` attribute enables automatic kernel discovery and compilation:

```csharp
private async Task TestAcceleratorPerformanceAsync(
    Accelerator accelerator,
    IUnifiedBuffer<float> bufferA,
    IUnifiedBuffer<float> bufferB,
    IUnifiedBuffer<float> bufferResult,
    int iterations)
{
    _logger.LogInformation("🔥 {AcceleratorName} ({AcceleratorType}):", 
        accelerator.Name, accelerator.AcceleratorType);
    
    try
    {
        using var session = _performanceMonitor.BeginSession($"{accelerator.AcceleratorType}_performance");
        
        // Source generator automatically creates optimized kernel launcher
        // The [KernelMethod] and [OptimizedKernel] attributes enable this
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);

        // Get accelerator-specific views from unified memory
        var viewA = bufferA.GetAcceleratorView(accelerator);
        var viewB = bufferB.GetAcceleratorView(accelerator);
        var viewResult = bufferResult.GetAcceleratorView(accelerator);

        // Warm-up run for accurate performance measurement
        kernel(viewA.Length, viewA, viewB, viewResult);
        accelerator.Synchronize();

        // Create adaptive scheduler for optimal performance
        using var scheduler = new AdaptiveScheduler(accelerator);
        var schedulingContext = scheduler.CreateContext("vector_addition");

        // Timed runs with adaptive scheduling
        var stopwatch = Stopwatch.StartNew();
        
        for (int iter = 0; iter < iterations; iter++)
        {
            // Use adaptive scheduling for optimal performance
            var task = schedulingContext.ScheduleKernel(
                () => kernel(viewA.Length, viewA, viewB, viewResult),
                TaskPriority.High);
            
            await task.CompletionAsync;
        }
        
        stopwatch.Stop();

        // Calculate performance metrics
        var avgTime = stopwatch.ElapsedMilliseconds / (double)iterations;
        var dataSize = viewA.Length * sizeof(float) * 3; // 3 arrays: A, B, result
        var throughput = dataSize / (avgTime / 1000.0) / (1024 * 1024 * 1024); // GB/s
        
        // Get scheduling statistics
        var schedulingStats = schedulingContext.GetStatistics();
        
        _logger.LogInformation("   Average time: {AvgTime:F2} ms", avgTime);
        _logger.LogInformation("   Throughput: {Throughput:F2} GB/s", throughput);
        _logger.LogInformation("   Occupancy: {Occupancy:P1}", schedulingStats.AverageOccupancy);
        _logger.LogInformation("   Kernel launches: {KernelLaunches}", schedulingStats.KernelLaunches);
        
        // Record detailed performance metrics
        session.RecordMetric($"{accelerator.AcceleratorType}_time_ms", avgTime);
        session.RecordMetric($"{accelerator.AcceleratorType}_throughput_gbps", throughput);
        session.RecordMetric($"{accelerator.AcceleratorType}_occupancy", schedulingStats.AverageOccupancy);
        
        // Verify correctness using unified memory
        var isCorrect = await VerifyAcceleratorResultsAsync(bufferA, bufferB, bufferResult);
        _logger.LogInformation("   Correctness: {Correctness}", isCorrect ? "✅ Verified" : "❌ Failed");
        
        // Display memory statistics
        var memoryStats = _memoryManager.GetMemoryStatistics(accelerator);
        _logger.LogInformation("   Memory efficiency: {MemoryEfficiency:P1}", memoryStats.MemoryEfficiency);
        
        _logger.LogInformation("");
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "❌ Error testing {AcceleratorName}: {ErrorMessage}", 
            accelerator.Name, ex.Message);
        _logger.LogInformation("");
    }
}

private async Task<bool> VerifyAcceleratorResultsAsync(
    IUnifiedBuffer<float> bufferA,
    IUnifiedBuffer<float> bufferB,
    IUnifiedBuffer<float> bufferResult)
{
    // Efficiently verify results using unified memory
    var dataA = await bufferA.ToArrayAsync();
    var dataB = await bufferB.ToArrayAsync();
    var result = await bufferResult.ToArrayAsync();
    
    return await Task.Run(() =>
    {
        for (int i = 0; i < Math.Min(100, result.Length); i++)
        {
            var expected = dataA[i] + dataB[i];
            if (Math.Abs(result[i] - expected) > 1e-5f)
            {
                return false;
            }
        }
        return true;
    });
}
```

## 🎯 Step 7: Build and Run with Source Generation

Build the application to trigger source generation:

```bash
# Build to generate source files
dotnet build --configuration Release

# Check generated files (optional)
ls Generated/

# Run the application
dotnet run --configuration Release
```

You should see enhanced output similar to:

```
info: UniversalComputeDemo.ComputeService[0]
      🚀 UniversalCompute Quick Start Demo
info: UniversalComputeDemo.ComputeService[0]
      ===================================
info: UniversalComputeDemo.ComputeService[0]
      📊 Available Hardware Accelerators:
info: UniversalComputeDemo.ComputeService[0]
      -----------------------------------
info: UniversalComputeDemo.ComputeService[0]
      ✅ Intel(R) Core(TM) i9-12900K CPU @ 3.20GHz
info: UniversalComputeDemo.ComputeService[0]
         Type: CPU
info: UniversalComputeDemo.ComputeService[0]
         Memory: 32,768 MB
info: UniversalComputeDemo.ComputeService[0]
         Compute Units: 16
info: UniversalComputeDemo.ComputeService[0]
      ✅ NVIDIA GeForce RTX 4090
info: UniversalComputeDemo.ComputeService[0]
         Type: Cuda
info: UniversalComputeDemo.ComputeService[0]
         Memory: 24,564 MB
info: UniversalComputeDemo.ComputeService[0]
         Compute Units: 128

info: UniversalComputeDemo.ComputeService[0]
      🔍 Testing Specialized Accelerators:
info: UniversalComputeDemo.ComputeService[0]
      -----------------------------------
info: UniversalComputeDemo.ComputeService[0]
      ✅ Intel AMX: Available
info: UniversalComputeDemo.ComputeService[0]
         Accelerator: Intel AMX CPU Accelerator
info: UniversalComputeDemo.ComputeService[0]
         Max threads per group: 1024
info: UniversalComputeDemo.ComputeService[0]
         Available memory: 31,744 MB
info: UniversalComputeDemo.ComputeService[0]
         Used memory: 1,024 MB
warn: UniversalComputeDemo.ComputeService[0]
      ❌ Intel NPU: Not available (Hardware not detected)
warn: UniversalComputeDemo.ComputeService[0]
      ❌ Apple Neural Engine: Not available (Platform not supported)

info: UniversalComputeDemo.ComputeService[0]
      ⚡ Performance Comparison: Vector Addition
info: UniversalComputeDemo.ComputeService[0]
      =========================================
info: UniversalComputeDemo.ComputeService[0]
      Vector size: 10,000,000 elements
info: UniversalComputeDemo.ComputeService[0]
      Iterations: 5
info: UniversalComputeDemo.ComputeService[0]
      Memory placement: PreferDevice

info: UniversalComputeDemo.ComputeService[0]
      🖥️  CPU Baseline (Vectorized):
info: UniversalComputeDemo.ComputeService[0]
         Average time: 12.34 ms
info: UniversalComputeDemo.ComputeService[0]
         Throughput: 9.72 GB/s
info: UniversalComputeDemo.ComputeService[0]
         Threads used: 16
info: UniversalComputeDemo.ComputeService[0]
         Correctness: ✅ Verified

info: UniversalComputeDemo.ComputeService[0]
      🔥 Intel AMX CPU Accelerator (CPU):
info: UniversalComputeDemo.ComputeService[0]
         Average time: 3.45 ms
info: UniversalComputeDemo.ComputeService[0]
         Throughput: 34.78 GB/s
info: UniversalComputeDemo.ComputeService[0]
         Occupancy: 95.2%
info: UniversalComputeDemo.ComputeService[0]
         Kernel launches: 5
info: UniversalComputeDemo.ComputeService[0]
         Correctness: ✅ Verified
info: UniversalComputeDemo.ComputeService[0]
         Memory efficiency: 87.3%

info: UniversalComputeDemo.ComputeService[0]
      🔥 NVIDIA GeForce RTX 4090 (Cuda):
info: UniversalComputeDemo.ComputeService[0]
         Average time: 0.89 ms
info: UniversalComputeDemo.ComputeService[0]
         Throughput: 134.83 GB/s
info: UniversalComputeDemo.ComputeService[0]
         Occupancy: 98.7%
info: UniversalComputeDemo.ComputeService[0]
         Kernel launches: 5
info: UniversalComputeDemo.ComputeService[0]
         Correctness: ✅ Verified
info: UniversalComputeDemo.ComputeService[0]
         Memory efficiency: 94.1%

info: UniversalComputeDemo.ComputeService[0]
      📊 Performance Summary:
info: UniversalComputeDemo.ComputeService[0]
         data_initialization_time: 45.67
info: UniversalComputeDemo.ComputeService[0]
         cpu_time_ms: 12.34
info: UniversalComputeDemo.ComputeService[0]
         cpu_throughput_gbps: 9.72
info: UniversalComputeDemo.ComputeService[0]
         CPU_time_ms: 3.45
info: UniversalComputeDemo.ComputeService[0]
         Cuda_time_ms: 0.89
info: UniversalComputeDemo.ComputeService[0]
         Cuda_throughput_gbps: 134.83

info: UniversalComputeDemo.ComputeService[0]
      🎉 Demo completed successfully!
Press any key to exit...
```

## 🎓 Step 8: Understanding the New Features

### What Just Happened?

1. **Dependency Injection**: Clean architecture with service-based design and scoped lifetimes
2. **Source Generation**: Automatic kernel discovery and compilation at build time
3. **Unified Memory Management**: Cross-accelerator memory sharing with automatic optimization
4. **Performance Monitoring**: Real-time metrics collection and performance analysis
5. **Kernel Attributes**: Modern attribute-based kernel development with optimization hints
6. **Adaptive Scheduling**: Intelligent task scheduling for optimal hardware utilization

### Key Concepts Demonstrated

#### Modern Architecture
- **Service-based Design**: Dependency injection with `IServiceProvider` integration
- **Structured Logging**: Comprehensive logging with `ILogger<T>` for better debugging
- **Performance Monitoring**: Built-in metrics collection and analysis
- **Memory Management**: Unified memory system with automatic placement optimization

#### Source Generation Features
- **`[KernelMethod]`**: Marks methods for automatic kernel compilation
- **`[OptimizedKernel]`**: Provides optimization hints to the compiler
- **Automatic Discovery**: Kernels are found and compiled at build time
- **Compile-time Validation**: Early error detection and optimization

#### Advanced Performance Features
- **Adaptive Scheduling**: Dynamic workload distribution across hardware
- **Memory Efficiency Tracking**: Real-time memory usage and optimization metrics
- **Occupancy Monitoring**: Hardware utilization statistics
- **Cross-accelerator Memory**: Shared memory pools for efficient data transfer

## 🚀 Step 9: Advanced Features

Now that you have the core functionality working, let's explore advanced features. Add these methods to your `ComputeService.cs`:

### Native AOT Compilation

First, let's prepare for Native AOT by updating your `.csproj`:

```xml
<PropertyGroup>
  <!-- Add these for Native AOT support -->
  <PublishAot>true</PublishAot>
  <InvariantGlobalization>true</InvariantGlobalization>
  <TrimMode>full</TrimMode>
</PropertyGroup>
```

### Advanced FFT Operations with Source Generation

```csharp
using UniversalCompute.FFT;
using System.Numerics;

// Add this kernel for custom FFT processing
[KernelMethod]
[OptimizedKernel(OptimizationLevel.Maximum, TargetAccelerator = AcceleratorType.All)]
static void PostProcessFFTKernel(Index1D index, ArrayView<Complex> fftData, ArrayView<float> magnitudes)
{
    magnitudes[index] = (float)fftData[index].Magnitude;
}

public async Task DemoAdvancedFFTOperationsAsync()
{
    _logger.LogInformation("📊 Advanced FFT Operations Demo:");
    _logger.LogInformation("--------------------------------");
    
    using var session = _performanceMonitor.BeginSession("advanced_fft_demo");
    using var fftManager = new FFTManager(_context);
    
    // Get the best accelerator for FFT operations
    var bestAccelerator = _context.GetBestAccelerator(AcceleratorType.Cuda) ?? 
                         _context.GetBestAccelerator(AcceleratorType.CPU);
    
    using var accelerator = bestAccelerator.CreateAccelerator(_context);
    
    // Create a complex test signal (multiple frequencies)
    const int N = 4096;
    var signal = new Complex[N];
    var frequencies = new[] { 5.0, 15.0, 25.0 }; // Multiple frequency components
    
    for (int i = 0; i < N; i++)
    {
        double sample = 0;
        foreach (var freq in frequencies)
        {
            sample += Math.Sin(2 * Math.PI * freq * i / N);
        }
        signal[i] = new Complex(sample, 0);
    }
    
    // Use unified memory for cross-accelerator efficiency
    using var signalBuffer = _memoryManager.AllocateUnified<Complex>(N, MemoryPlacement.PreferDevice);
    using var fftBuffer = _memoryManager.AllocateUnified<Complex>(N, MemoryPlacement.PreferDevice);
    using var magnitudeBuffer = _memoryManager.AllocateUnified<float>(N, MemoryPlacement.PreferDevice);
    
    await signalBuffer.CopyFromAsync(signal);
    
    // Perform FFT with performance monitoring
    var fftStopwatch = Stopwatch.StartNew();
    await fftManager.FFT1DAsync(
        signalBuffer.GetAcceleratorView(accelerator), 
        fftBuffer.GetAcceleratorView(accelerator), 
        forward: true);
    fftStopwatch.Stop();
    
    // Process FFT results with custom kernel
    var postProcessKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Complex>, ArrayView<float>>(PostProcessFFTKernel);
    postProcessKernel(N, fftBuffer.GetAcceleratorView(accelerator), magnitudeBuffer.GetAcceleratorView(accelerator));
    accelerator.Synchronize();
    
    // Find peak frequencies
    var magnitudes = await magnitudeBuffer.ToArrayAsync();
    var peaks = FindPeaks(magnitudes, N);
    
    _logger.LogInformation("FFT processing time: {FFTTime:F2} ms", fftStopwatch.ElapsedMilliseconds);
    _logger.LogInformation("Peak frequencies detected:");
    foreach (var (frequency, magnitude) in peaks)
    {
        _logger.LogInformation("  {Frequency:F1} Hz: {Magnitude:F2}", frequency, magnitude);
    }
    
    session.RecordMetric("fft_processing_time_ms", fftStopwatch.ElapsedMilliseconds);
    session.RecordMetric("fft_size", N);
    session.RecordMetric("peaks_detected", peaks.Count);
    
    _logger.LogInformation("");
}

private List<(double frequency, double magnitude)> FindPeaks(float[] magnitudes, int sampleRate)
{
    var peaks = new List<(double, double)>();
    
    for (int i = 1; i < magnitudes.Length / 2 - 1; i++)
    {
        if (magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1] && magnitudes[i] > 10.0)
        {
            var frequency = (double)i * sampleRate / magnitudes.Length;
            peaks.Add((frequency, magnitudes[i]));
        }
    }
    
    return peaks.OrderByDescending(p => p.Item2).Take(5).ToList();
}
```

### Unified Tensor Operations with Automatic Optimization

```csharp
using UniversalCompute.Numerics;

// Advanced matrix multiplication kernel with tiling optimization
[KernelMethod]
[OptimizedKernel(OptimizationLevel.Maximum, EnableTiling = true, TileSize = 16)]
static void TiledMatrixMultiplyKernel(
    Index2D index, 
    ArrayView2D<float, Stride2D.DenseX> matrixA,
    ArrayView2D<float, Stride2D.DenseX> matrixB,
    ArrayView2D<float, Stride2D.DenseX> result)
{
    var row = index.X;
    var col = index.Y;
    
    if (row >= result.Height || col >= result.Width)
        return;
    
    float sum = 0.0f;
    for (int k = 0; k < matrixA.Width; k++)
    {
        sum += matrixA[row, k] * matrixB[k, col];
    }
    
    result[row, col] = sum;
}

public async Task DemoAdvancedTensorOperationsAsync()
{
    _logger.LogInformation("🧮 Advanced Tensor Operations Demo:");
    _logger.LogInformation("----------------------------------");
    
    using var session = _performanceMonitor.BeginSession("advanced_tensor_demo");
    
    // Create tensors with automatic optimization
    const int matrixSize = 1024;
    var shape = TensorShape.Create2D(matrixSize, matrixSize);
    
    // Use ITensorFactory for automatic accelerator selection
    var tensorFactory = _context.Services.GetRequiredService<ITensorFactory>();
    
    using var tensorA = tensorFactory.CreateTensor<float>(shape, MemoryPlacement.PreferDevice);
    using var tensorB = tensorFactory.CreateTensor<float>(shape, MemoryPlacement.PreferDevice);
    using var result = tensorFactory.CreateTensor<float>(shape, MemoryPlacement.PreferDevice);
    
    // Initialize with test data using parallel processing
    await InitializeTensorDataAsync(tensorA, tensorB, matrixSize);
    
    // Get the best accelerator for tensor operations
    var accelerator = tensorA.GetOptimalAccelerator();
    
    // Perform optimized matrix multiplication
    var matMulStopwatch = Stopwatch.StartNew();
    
    // Use the tiled kernel for better cache efficiency
    var tiledKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(TiledMatrixMultiplyKernel);
    
    tiledKernel(
        new Index2D(matrixSize, matrixSize),
        tensorA.As2DView(),
        tensorB.As2DView(),
        result.As2DView());
    
    accelerator.Synchronize();
    matMulStopwatch.Stop();
    
    // Calculate performance metrics
    var operations = (long)matrixSize * matrixSize * matrixSize * 2; // multiply + add
    var gflops = operations / (matMulStopwatch.ElapsedMilliseconds / 1000.0) / 1e9;
    
    // Get tensor statistics
    var resultStats = await result.GetStatisticsAsync();
    
    _logger.LogInformation("Matrix multiplication ({Size}x{Size}):", matrixSize, matrixSize);
    _logger.LogInformation("  Time: {Time:F2} ms", matMulStopwatch.ElapsedMilliseconds);
    _logger.LogInformation("  Performance: {GFLOPS:F2} GFLOPS", gflops);
    _logger.LogInformation("  Accelerator: {Accelerator}", accelerator.Name);
    _logger.LogInformation("  Result mean: {Mean:F6}", resultStats.Mean);
    _logger.LogInformation("  Result std: {StdDev:F6}", resultStats.StandardDeviation);
    
    // Test tensor fusion operations
    await DemoTensorFusionAsync(tensorA, tensorB, result);
    
    session.RecordMetric("matmul_time_ms", matMulStopwatch.ElapsedMilliseconds);
    session.RecordMetric("matmul_gflops", gflops);
    session.RecordMetric("matrix_size", matrixSize);
    
    _logger.LogInformation("");
}

private async Task InitializeTensorDataAsync(IUnifiedTensor<float> tensorA, IUnifiedTensor<float> tensorB, int size)
{
    var random = new Random(42);
    var dataA = new float[size, size];
    var dataB = new float[size, size];
    
    await Task.Run(() =>
    {
        Parallel.For(0, size, i =>
        {
            for (int j = 0; j < size; j++)
            {
                dataA[i, j] = random.NextSingle() * 2.0f - 1.0f; // Range [-1, 1]
                dataB[i, j] = random.NextSingle() * 2.0f - 1.0f;
            }
        });
    });
    
    await tensorA.CopyFromAsync(dataA);
    await tensorB.CopyFromAsync(dataB);
}

private async Task DemoTensorFusionAsync(IUnifiedTensor<float> tensorA, IUnifiedTensor<float> tensorB, IUnifiedTensor<float> result)
{
    _logger.LogInformation("🔗 Tensor Fusion Operations:");
    
    // Demonstrate kernel fusion for better performance
    using var fusedSession = _performanceMonitor.BeginSession("tensor_fusion");
    
    // Create a fused operation: result = (A + B) * 0.5
    var fusionStopwatch = Stopwatch.StartNew();
    
    // This would be automatically fused by the compiler
    var intermediate = await tensorA.AddAsync(tensorB);
    var final = await intermediate.MultiplyAsync(0.5f);
    
    fusionStopwatch.Stop();
    
    var fusionStats = await final.GetStatisticsAsync();
    
    _logger.LogInformation("  Fused operation time: {Time:F2} ms", fusionStopwatch.ElapsedMilliseconds);
    _logger.LogInformation("  Fused result mean: {Mean:F6}", fusionStats.Mean);
    
    fusedSession.RecordMetric("fusion_time_ms", fusionStopwatch.ElapsedMilliseconds);
}
```

### Update your main RunDemoAsync method

Add calls to the new advanced features:

```csharp
public async Task RunDemoAsync()
{
    _logger.LogInformation("🚀 UniversalCompute Quick Start Demo");
    _logger.LogInformation("===================================");

    // Original functionality
    await DisplayHardwareInfoAsync();
    await TestSpecializedAcceleratorsAsync();
    await RunPerformanceComparisonAsync();

    // New advanced features
    await DemoAdvancedFFTOperationsAsync();
    await DemoAdvancedTensorOperationsAsync();

    _logger.LogInformation("🎉 Demo completed successfully!");
}
```

## 🎯 Step 10: Native AOT Deployment

Deploy your application as a self-contained native binary:

```bash
# Publish for Native AOT (Windows)
dotnet publish -r win-x64 -c Release

# Publish for Native AOT (Linux)
dotnet publish -r linux-x64 -c Release

# Publish for Native AOT (macOS)
dotnet publish -r osx-x64 -c Release
```

The resulting binary will be:
- **Self-contained**: No .NET runtime required
- **Fast startup**: Near-instant application launch
- **Small footprint**: Optimized binary size
- **High performance**: Native code execution

## 📚 What's Next?

Congratulations! You've successfully created a comprehensive UniversalCompute application with all the latest features. Here are your next steps:

### 🔧 Development & Architecture
- **[Dependency Injection Guide](Dependency-Injection-Guide)** - Advanced service configuration and lifetime management
- **[Source Generation Deep Dive](Source-Generation-Deep-Dive)** - Custom kernel attributes and compile-time optimization
- **[Performance Monitoring](Performance-Monitoring)** - Advanced metrics collection and analysis
- **[Memory Management](Memory-Management)** - Unified memory patterns and optimization strategies

### 🚀 Advanced Features
- **[Unified Memory System](Unified-Memory-System)** - Cross-accelerator memory sharing and automatic placement
- **[Adaptive Scheduling](Adaptive-Scheduling)** - Dynamic workload distribution and optimization
- **[Kernel Fusion](Kernel-Fusion)** - Automatic operation fusion for better performance
- **[FFT Operations](FFT-Operations)** - Advanced signal processing and frequency analysis

### 🎯 Specialized Hardware
- **[Hardware Accelerators Guide](Hardware-Accelerators)** - Intel AMX, NPU, Apple Neural Engine integration
- **[CUDA Optimization](CUDA-Optimization)** - NVIDIA-specific performance tuning
- **[OpenCL Development](OpenCL-Development)** - Cross-platform GPU programming
- **[Velocity Backend](Velocity-Backend)** - CPU vectorization and SIMD optimization

### 🏗️ Production & Deployment
- **[Native AOT Guide](Native-AOT-Guide)** - Complete AOT compilation and deployment strategies
- **[Cross-Platform Development](Cross-Platform-Development)** - Multi-platform development and testing
- **[Performance Tuning](Performance-Tuning)** - Production-grade optimization techniques
- **[Monitoring & Diagnostics](Monitoring-Diagnostics)** - Production monitoring and troubleshooting

### 📖 Reference & Examples
- **[API Reference](API-Reference)** - Complete API documentation with examples
- **[Examples Gallery](Examples-Gallery)** - Real-world use cases and implementation patterns
- **[Best Practices](Best-Practices)** - Recommended patterns and architecture guidelines
- **[Migration Guide](Migration-Guide)** - Upgrading from ILGPU or other compute frameworks

### 🤝 Community & Contribution
- **[Building from Source](Building-from-Source)** - Development environment setup and contribution guidelines
- **[Community Forum](Community-Forum)** - Get help and share knowledge
- **[Contributing](Contributing)** - How to contribute code, documentation, and examples

## 🎯 Key Takeaways

### ✅ Modern Development Features
- **Dependency Injection**: Clean, testable, service-based architecture
- **Source Generation**: Compile-time kernel discovery and optimization
- **Structured Logging**: Comprehensive diagnostic information
- **Performance Monitoring**: Real-time metrics and analysis

### ✅ Advanced Compute Capabilities
- **Unified Memory Management**: Seamless cross-accelerator memory sharing
- **Adaptive Scheduling**: Intelligent workload distribution
- **Kernel Attributes**: Modern, declarative kernel development
- **Automatic Optimization**: Compiler-driven performance improvements

### ✅ Production-Ready Features
- **Native AOT Support**: Self-contained, high-performance deployments
- **Cross-Platform**: Consistent behavior across Windows, Linux, and macOS
- **Hardware Abstraction**: Write once, run on any accelerator
- **Type Safety**: Compile-time verification and error checking

### ✅ Enterprise Integration
- **Service Integration**: Full .NET ecosystem compatibility
- **Monitoring & Diagnostics**: Production-grade observability
- **Scalable Architecture**: From prototypes to production workloads
- **Easy Migration**: Smooth transition from existing compute frameworks

---

## 🚀 Ready to Build Production Applications!

You now have all the tools to create high-performance, production-ready compute applications with UniversalCompute. The combination of modern .NET development practices with cutting-edge compute capabilities gives you unprecedented power and flexibility.

**Start building:** [Examples Gallery](Examples-Gallery) | [Best Practices](Best-Practices) | [API Reference](API-Reference)

**Get help:** [Community Forum](Community-Forum) | [Documentation](Documentation) | [GitHub Issues](https://github.com/universalcompute/issues)