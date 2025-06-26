# Quick Start Tutorial

Get up and running with UniversalCompute in 15 minutes! This tutorial will guide you through creating your first high-performance compute application.

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ [.NET 8.0+ SDK installed](https://dotnet.microsoft.com/download)
- ✅ [UniversalCompute package installed](Installation-Guide)
- ✅ A compatible IDE (Visual Studio, Rider, or VS Code)

## 🎯 What We'll Build

We'll create a simple application that:
1. Detects available hardware accelerators
2. Performs vector addition using GPU/CPU acceleration
3. Compares performance across different accelerators
4. Demonstrates memory management best practices

## 🚀 Step 1: Create New Project

```bash
# Create a new console application
dotnet new console -n UniversalComputeDemo
cd UniversalComputeDemo

# Add UniversalCompute package
dotnet add package UniversalCompute --version 1.0.0-alpha1

# Enable unsafe code (required for high-performance operations)
```

Edit your `.csproj` file to enable unsafe code:
```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
    <Nullable>enable</Nullable>
  </PropertyGroup>
  
  <ItemGroup>
    <PackageReference Include="UniversalCompute" Version="1.0.0-alpha1" />
  </ItemGroup>
</Project>
```

## 🎯 Step 2: Hardware Detection

Replace the contents of `Program.cs`:

```csharp
using UniversalCompute;
using UniversalCompute.Runtime;
using UniversalCompute.Runtime.CPU;
using System.Diagnostics;

Console.WriteLine("🚀 UniversalCompute Quick Start Demo");
Console.WriteLine("===================================\n");

// Create context with all available accelerators
using var context = Context.Create().EnableAllAccelerators();

// Display available accelerators
Console.WriteLine("📊 Available Hardware Accelerators:");
Console.WriteLine("-----------------------------------");

foreach (var device in context.Devices)
{
    Console.WriteLine($"✅ {device.Name}");
    Console.WriteLine($"   Type: {device.AcceleratorType}");
    Console.WriteLine($"   Memory: {device.MemorySize / (1024 * 1024):N0} MB");
    Console.WriteLine($"   Compute Units: {device.NumMultiProcessors}");
    Console.WriteLine();
}

// Test specialized accelerators
await TestSpecializedAccelerators(context);

// Run performance comparison
await RunPerformanceComparison(context);

Console.WriteLine("🎉 Demo completed successfully!");
Console.WriteLine("Press any key to exit...");
Console.ReadKey();

// Helper methods will be defined below...
```

## 🔧 Step 3: Specialized Accelerator Testing

Add these methods to test hardware-specific accelerators:

```csharp
static async Task TestSpecializedAccelerators(Context context)
{
    Console.WriteLine("🔍 Testing Specialized Accelerators:");
    Console.WriteLine("-----------------------------------");

    // Test Intel AMX
    try
    {
        using var amxAccelerator = context.CreateAMXAccelerator();
        Console.WriteLine("✅ Intel AMX: Available");
        Console.WriteLine($"   Accelerator: {amxAccelerator.Name}");
        Console.WriteLine($"   Max threads per group: {amxAccelerator.MaxNumThreadsPerGroup}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"❌ Intel AMX: Not available ({ex.Message.Split('.')[0]})");
    }

    // Test Intel NPU
    try
    {
        using var npuAccelerator = context.CreateNPUAccelerator();
        Console.WriteLine("✅ Intel NPU: Available");
        Console.WriteLine($"   Accelerator: {npuAccelerator.Name}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"❌ Intel NPU: Not available ({ex.Message.Split('.')[0]})");
    }

    // Test Apple Neural Engine
    try
    {
        using var aneAccelerator = context.CreateANEAccelerator();
        Console.WriteLine("✅ Apple Neural Engine: Available");
        Console.WriteLine($"   Accelerator: {aneAccelerator.Name}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"❌ Apple Neural Engine: Not available ({ex.Message.Split('.')[0]})");
    }

    Console.WriteLine();
}
```

## ⚡ Step 4: Performance Comparison

Add the performance comparison method:

```csharp
static async Task RunPerformanceComparison(Context context)
{
    Console.WriteLine("⚡ Performance Comparison: Vector Addition");
    Console.WriteLine("=========================================");

    const int vectorSize = 10_000_000; // 10 million elements
    const int iterations = 5;

    // Prepare test data
    var hostA = new float[vectorSize];
    var hostB = new float[vectorSize];
    var hostResult = new float[vectorSize];

    // Initialize with random data
    var random = new Random(42);
    for (int i = 0; i < vectorSize; i++)
    {
        hostA[i] = random.NextSingle() * 100;
        hostB[i] = random.NextSingle() * 100;
    }

    Console.WriteLine($"Vector size: {vectorSize:N0} elements");
    Console.WriteLine($"Iterations: {iterations}");
    Console.WriteLine();

    // Test CPU performance first
    await TestCPUPerformance(hostA, hostB, hostResult, iterations);

    // Test each available accelerator
    foreach (var device in context.Devices)
    {
        try
        {
            using var accelerator = device.CreateAccelerator(context);
            await TestAcceleratorPerformance(accelerator, hostA, hostB, iterations);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ {device.Name}: Failed ({ex.Message.Split('.')[0]})");
        }
    }
}
```

## 🧮 Step 5: CPU Baseline Performance

Add the CPU performance test:

```csharp
static async Task TestCPUPerformance(float[] a, float[] b, float[] result, int iterations)
{
    Console.WriteLine("🖥️  CPU Baseline (Single-threaded):");
    
    var stopwatch = Stopwatch.StartNew();
    
    for (int iter = 0; iter < iterations; iter++)
    {
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = a[i] + b[i];
        }
    }
    
    stopwatch.Stop();
    
    var avgTime = stopwatch.ElapsedMilliseconds / (double)iterations;
    var throughput = (a.Length * sizeof(float) * 3) / (avgTime / 1000.0) / (1024 * 1024 * 1024); // GB/s
    
    Console.WriteLine($"   Average time: {avgTime:F2} ms");
    Console.WriteLine($"   Throughput: {throughput:F2} GB/s");
    Console.WriteLine();
}
```

## 🚀 Step 6: Accelerator Performance Testing

Add the accelerator performance test with our vector addition kernel:

```csharp
// Define the vector addition kernel
static void VectorAddKernel(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
{
    result[index] = a[index] + b[index];
}

static async Task TestAcceleratorPerformance(Accelerator accelerator, float[] hostA, float[] hostB, int iterations)
{
    Console.WriteLine($"🔥 {accelerator.Name} ({accelerator.AcceleratorType}):");
    
    try
    {
        // Allocate memory on accelerator
        using var bufferA = accelerator.Allocate1D<float>(hostA.Length);
        using var bufferB = accelerator.Allocate1D<float>(hostB.Length);
        using var bufferResult = accelerator.Allocate1D<float>(hostA.Length);

        // Load kernel
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);

        // Copy data to accelerator (one-time setup)
        bufferA.CopyFromCPU(hostA);
        bufferB.CopyFromCPU(hostB);

        // Warm-up run
        kernel(bufferA.Length, bufferA.View, bufferB.View, bufferResult.View);
        accelerator.Synchronize();

        // Timed runs
        var stopwatch = Stopwatch.StartNew();
        
        for (int iter = 0; iter < iterations; iter++)
        {
            kernel(bufferA.Length, bufferA.View, bufferB.View, bufferResult.View);
            accelerator.Synchronize(); // Wait for completion
        }
        
        stopwatch.Stop();

        // Calculate performance metrics
        var avgTime = stopwatch.ElapsedMilliseconds / (double)iterations;
        var throughput = (hostA.Length * sizeof(float) * 3) / (avgTime / 1000.0) / (1024 * 1024 * 1024); // GB/s
        var speedup = await CalculateSpeedup(avgTime, hostA, hostB);

        Console.WriteLine($"   Average time: {avgTime:F2} ms");
        Console.WriteLine($"   Throughput: {throughput:F2} GB/s");
        Console.WriteLine($"   Speedup: {speedup:F1}x vs CPU");
        
        // Verify correctness (check first few elements)
        var result = bufferResult.GetAsArray1D();
        var isCorrect = true;
        for (int i = 0; i < Math.Min(100, result.Length); i++)
        {
            var expected = hostA[i] + hostB[i];
            if (Math.Abs(result[i] - expected) > 1e-5f)
            {
                isCorrect = false;
                break;
            }
        }
        
        Console.WriteLine($"   Correctness: {(isCorrect ? "✅ Verified" : "❌ Failed")}");
        Console.WriteLine();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"   ❌ Error: {ex.Message}");
        Console.WriteLine();
    }
}

static async Task<double> CalculateSpeedup(double acceleratorTime, float[] a, float[] b)
{
    // Quick CPU baseline for comparison
    var stopwatch = Stopwatch.StartNew();
    var result = new float[a.Length];
    
    for (int i = 0; i < a.Length; i++)
    {
        result[i] = a[i] + b[i];
    }
    
    stopwatch.Stop();
    
    return stopwatch.ElapsedMilliseconds / acceleratorTime;
}
```

## 🎯 Step 7: Run the Application

Now run your application:

```bash
dotnet run --configuration Release
```

You should see output similar to:

```
🚀 UniversalCompute Quick Start Demo
===================================

📊 Available Hardware Accelerators:
-----------------------------------
✅ Intel(R) Core(TM) i9-12900K CPU @ 3.20GHz
   Type: CPU
   Memory: 32,768 MB
   Compute Units: 16

✅ NVIDIA GeForce RTX 4090
   Type: Cuda
   Memory: 24,564 MB
   Compute Units: 128

🔍 Testing Specialized Accelerators:
-----------------------------------
✅ Intel AMX: Available
   Accelerator: Intel AMX CPU Accelerator
   Max threads per group: 1024
❌ Intel NPU: Not available (Hardware not detected)
❌ Apple Neural Engine: Not available (Platform not supported)

⚡ Performance Comparison: Vector Addition
=========================================
Vector size: 10,000,000 elements
Iterations: 5

🖥️  CPU Baseline (Single-threaded):
   Average time: 45.20 ms
   Throughput: 2.65 GB/s

🔥 Intel AMX CPU Accelerator (CPU):
   Average time: 8.15 ms
   Throughput: 14.72 GB/s
   Speedup: 5.5x vs CPU
   Correctness: ✅ Verified

🔥 NVIDIA GeForce RTX 4090 (Cuda):
   Average time: 1.23 ms
   Throughput: 97.56 GB/s
   Speedup: 36.7x vs CPU
   Correctness: ✅ Verified

🎉 Demo completed successfully!
Press any key to exit...
```

## 🎓 Step 8: Understanding the Results

### What Just Happened?

1. **Hardware Detection**: UniversalCompute automatically detected all available accelerators on your system
2. **Unified API**: The same kernel code ran on different hardware without modification
3. **Performance Optimization**: Each accelerator was automatically optimized for the specific hardware
4. **Memory Management**: Efficient memory allocation and transfer handled automatically

### Key Concepts Demonstrated

- **Context Creation**: Central orchestrator for all UniversalCompute operations
- **Accelerator Abstraction**: Unified interface for different hardware types
- **Kernel Definition**: Regular C# methods compiled for accelerator execution
- **Memory Buffers**: Type-safe, high-performance memory management
- **Synchronization**: Proper coordination between CPU and accelerator

## 🚀 Step 9: Next Steps - Advanced Features

Now that you have the basics working, try these advanced features:

### Add FFT Operations

```csharp
using UniversalCompute.FFT;
using System.Numerics;

// Add this method to your program
static async Task DemoFFTOperations(Context context)
{
    Console.WriteLine("📊 FFT Operations Demo:");
    Console.WriteLine("----------------------");
    
    using var fftManager = new FFTManager(context);
    using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);
    
    // Create a test signal (sine wave)
    const int N = 1024;
    var signal = new Complex[N];
    for (int i = 0; i < N; i++)
    {
        signal[i] = new Complex(Math.Sin(2 * Math.PI * 5 * i / N), 0); // 5 Hz
    }
    
    // Allocate GPU memory
    using var inputBuffer = accelerator.Allocate1D<Complex>(N);
    using var outputBuffer = accelerator.Allocate1D<Complex>(N);
    
    // Perform FFT
    inputBuffer.CopyFromCPU(signal);
    fftManager.FFT1D(inputBuffer.View, outputBuffer.View, forward: true);
    
    var fftResult = outputBuffer.GetAsArray1D();
    
    // Find peak frequency
    var maxMagnitude = 0.0;
    var peakFrequency = 0;
    for (int i = 0; i < N / 2; i++)
    {
        var magnitude = fftResult[i].Magnitude;
        if (magnitude > maxMagnitude)
        {
            maxMagnitude = magnitude;
            peakFrequency = i;
        }
    }
    
    Console.WriteLine($"Peak frequency detected: {peakFrequency} Hz (expected: 5 Hz)");
    Console.WriteLine($"Peak magnitude: {maxMagnitude:F2}");
    Console.WriteLine();
}
```

### Add Tensor Operations

```csharp
using UniversalCompute.Core;

// Add this method for tensor operations
static async Task DemoTensorOperations(Context context)
{
    Console.WriteLine("🧮 Tensor Operations Demo:");
    Console.WriteLine("-------------------------");
    
    using var accelerator = context.GetPreferredDevice().CreateAccelerator(context);
    
    // Create 2D tensors for matrix multiplication
    var shape = TensorShape.Create2D(512, 512);
    using var tensorA = new UnifiedTensor<float>(shape, accelerator);
    using var tensorB = new UnifiedTensor<float>(shape, accelerator);
    
    // Initialize with test data
    var dataA = new float[512, 512];
    var dataB = new float[512, 512];
    var random = new Random(42);
    
    for (int i = 0; i < 512; i++)
    {
        for (int j = 0; j < 512; j++)
        {
            dataA[i, j] = random.NextSingle();
            dataB[i, j] = random.NextSingle();
        }
    }
    
    // Copy data to GPU
    tensorA.CopyFromCPU(dataA);
    tensorB.CopyFromCPU(dataB);
    
    // Perform matrix multiplication
    var stopwatch = Stopwatch.StartNew();
    using var result = tensorA.MatrixMultiply(tensorB);
    stopwatch.Stop();
    
    Console.WriteLine($"Matrix multiplication (512x512): {stopwatch.ElapsedMilliseconds} ms");
    
    // Calculate statistics
    var sum = result.Sum();
    var mean = result.Mean();
    
    Console.WriteLine($"Result sum: {sum:F2}");
    Console.WriteLine($"Result mean: {mean:F6}");
    Console.WriteLine();
}
```

Call these new methods from your main method:

```csharp
// Add after the performance comparison
await DemoFFTOperations(context);
await DemoTensorOperations(context);
```

## 📚 What's Next?

Congratulations! You've successfully created your first UniversalCompute application. Here are some next steps:

### Explore More Features
- **[Hardware Accelerators Guide](Hardware-Accelerators)** - Deep dive into specialized hardware
- **[FFT Operations](FFT-Operations)** - Advanced signal processing
- **[Memory Management](Memory-Management)** - Optimize memory usage
- **[Performance Tuning](Performance-Tuning)** - Squeeze out maximum performance

### Learn Advanced Concepts
- **[Native AOT Guide](Native-AOT-Guide)** - Compile to native binaries
- **[Cross-Platform Development](Cross-Platform-Development)** - Deploy everywhere
- **[API Reference](API-Reference)** - Complete documentation

### Browse Examples
- **[Examples Gallery](Examples-Gallery)** - Real-world use cases
- **[Building from Source](Building-from-Source)** - Contribute to development

## 🎯 Key Takeaways

✅ **Universal API**: Write once, run on any accelerator  
✅ **High Performance**: Automatic optimization for each hardware type  
✅ **Type Safety**: Compile-time verification and error checking  
✅ **Memory Efficiency**: Automatic memory management and optimization  
✅ **Easy Integration**: Simple NuGet package installation  
✅ **Cross-Platform**: Windows, Linux, macOS support  

---

**🚀 Ready to build amazing high-performance applications with UniversalCompute!**

**Continue learning:** [Hardware Accelerators](Hardware-Accelerators) | [Performance Tuning](Performance-Tuning) | [Examples Gallery](Examples-Gallery)