// Copyright (c) 2025 Michael Ivertowski, Ernst & Young Ltd. Switzerland
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/mivertowski/UniversalCompute/blob/master/LICENSE.txt
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

# Profiling and Debugging

## Overview

Effective profiling and debugging are essential for developing high-performance ILGPU applications. This guide covers comprehensive debugging strategies, performance profiling techniques, and tools for identifying and resolving performance bottlenecks across different accelerator platforms.

## Technical Background

### Performance Analysis Challenges

ILGPU applications present unique debugging and profiling challenges:

- **Asynchronous execution**: GPU kernels execute asynchronously with CPU code
- **Limited debugging capabilities**: Traditional debuggers have limited GPU support
- **Memory access patterns**: Non-obvious memory performance characteristics
- **Platform differences**: Different behavior across CPU, CUDA, and OpenCL backends
- **Timing accuracy**: Proper measurement of GPU execution times

### ILGPU Debugging and Profiling Strategy

ILGPU provides multiple debugging and profiling approaches:

1. **CPU debugging**: Use CPU accelerator for full debugging capability
2. **Performance counters**: Built-in timing and memory usage monitoring
3. **Platform-specific tools**: Integration with CUDA profilers and debugging tools
4. **Validation techniques**: Comparative testing across different backends

## Debugging Techniques

### CPU Accelerator Debugging

```csharp
using ILGPU;
using ILGPU.Runtime;
using System.Diagnostics;

public class DebugExample
{
    public static void DebugWithCPUAccelerator()
    {
        // Create context with debug mode enabled
        using var context = Context.Create(builder => 
            builder.CPU().EnableAssertions().OptimizationLevel(OptimizationLevel.Debug));
        
        using var accelerator = context.CreateCPUAccelerator(0);
        
        const int dataSize = 1000;
        using var inputBuffer = accelerator.Allocate1D<float>(dataSize);
        using var outputBuffer = accelerator.Allocate1D<float>(dataSize);
        
        // Initialize test data
        var inputData = new float[dataSize];
        for (int i = 0; i < dataSize; i++)
        {
            inputData[i] = i * 0.1f;
        }
        inputBuffer.CopyFromCPU(inputData);
        
        // Load and execute kernel
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(DebugKernel);
        
        kernel(dataSize, inputBuffer.View, outputBuffer.View);
        accelerator.Synchronize();
        
        // Verify results
        var output = outputBuffer.GetAsArray1D();
        for (int i = 0; i < Math.Min(10, dataSize); i++)
        {
            Console.WriteLine($"output[{i}] = {output[i]:F3} (expected: {inputData[i] * 2:F3})");
        }
    }
    
    static void DebugKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
    {
        var i = (int)index;
        if (i < input.Length)
        {
            // Add debug assertions (only work on CPU accelerator)
            Debug.Assert(i >= 0, "Index should be non-negative");
            Debug.Assert(input[i] >= 0, $"Input value at {i} should be non-negative: {input[i]}");
            
            // Computation with debug output
            var result = input[i] * 2.0f;
            
            // Debug output (only visible on CPU accelerator)
            if (i < 5)
            {
                Console.WriteLine($"Debug: input[{i}] = {input[i]:F3}, output = {result:F3}");
            }
            
            output[i] = result;
        }
    }
}
```

### Cross-Platform Validation

```csharp
public class CrossPlatformValidator
{
    public static void ValidateAcrossPlatforms<T>(
        Context context,
        Action<Index1D, ArrayView<T>, ArrayView<T>> kernel,
        T[] inputData) where T : unmanaged, IEquatable<T>
    {
        var results = new Dictionary<AcceleratorType, T[]>();
        
        // Test on all available accelerators
        foreach (var device in context)
        {
            try
            {
                using var accelerator = device.CreateAccelerator(context);
                using var inputBuffer = accelerator.Allocate1D(inputData);
                using var outputBuffer = accelerator.Allocate1D<T>(inputData.Length);
                
                var loadedKernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<T>, ArrayView<T>>(kernel);
                
                loadedKernel(inputData.Length, inputBuffer.View, outputBuffer.View);
                accelerator.Synchronize();
                
                results[device.AcceleratorType] = outputBuffer.GetAsArray1D();
                
                Console.WriteLine($"✓ {device.AcceleratorType} execution completed");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ {device.AcceleratorType} execution failed: {ex.Message}");
            }
        }
        
        // Compare results across platforms
        CompareResults(results);
    }
    
    private static void CompareResults<T>(Dictionary<AcceleratorType, T[]> results) 
        where T : unmanaged, IEquatable<T>
    {
        if (results.Count < 2)
        {
            Console.WriteLine("Need at least 2 platforms to compare results");
            return;
        }
        
        var platforms = results.Keys.ToArray();
        var reference = results[platforms[0]];
        
        for (int p = 1; p < platforms.Length; p++)
        {
            var current = results[platforms[p]];
            var differences = 0;
            
            for (int i = 0; i < Math.Min(reference.Length, current.Length); i++)
            {
                if (!reference[i].Equals(current[i]))
                {
                    differences++;
                    if (differences <= 5) // Show first 5 differences
                    {
                        Console.WriteLine($"Difference at index {i}: {platforms[0]}={reference[i]}, {platforms[p]}={current[i]}");
                    }
                }
            }
            
            if (differences == 0)
            {
                Console.WriteLine($"✓ {platforms[0]} and {platforms[p]} results match perfectly");
            }
            else
            {
                Console.WriteLine($"✗ {differences} differences found between {platforms[0]} and {platforms[p]}");
            }
        }
    }
}
```

### Error Handling and Diagnostics

```csharp
public class ErrorHandling
{
    public static void SafeKernelExecution()
    {
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);
        
        try
        {
            // Attempt kernel compilation and execution
            var result = ExecuteKernelSafely(accelerator);
            Console.WriteLine($"Kernel executed successfully: {result}");
        }
        catch (AcceleratorException ex)
        {
            Console.WriteLine($"Accelerator error: {ex.Message}");
            
            // Try fallback to CPU
            using var cpuAccelerator = context.CreateCPUAccelerator(0);
            var fallbackResult = ExecuteKernelSafely(cpuAccelerator);
            Console.WriteLine($"Fallback CPU execution: {fallbackResult}");
        }
        catch (CompilationException ex)
        {
            Console.WriteLine($"Kernel compilation failed: {ex.Message}");
            Console.WriteLine($"Compilation output: {ex.CompilerOutput}");
        }
        catch (OutOfMemoryException ex)
        {
            Console.WriteLine($"Out of memory: {ex.Message}");
            // Implement memory reduction strategy
            ExecuteWithReducedMemory(accelerator);
        }
    }
    
    private static float ExecuteKernelSafely(Accelerator accelerator)
    {
        const int dataSize = 1000;
        
        // Check memory availability
        if (accelerator.MemorySize < dataSize * sizeof(float) * 2)
        {
            throw new OutOfMemoryException("Insufficient device memory");
        }
        
        using var inputBuffer = accelerator.Allocate1D<float>(dataSize);
        using var outputBuffer = accelerator.Allocate1D<float>(dataSize);
        
        // Initialize data
        var inputData = Enumerable.Range(0, dataSize).Select(i => (float)i).ToArray();
        inputBuffer.CopyFromCPU(inputData);
        
        // Compile and execute kernel
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(SafeKernel);
        
        kernel(dataSize, inputBuffer.View, outputBuffer.View);
        accelerator.Synchronize();
        
        // Validate results
        var output = outputBuffer.GetAsArray1D();
        return output.Sum();
    }
    
    static void SafeKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
    {
        var i = (int)index;
        
        // Bounds checking
        if (i >= input.Length || i >= output.Length)
            return;
        
        // Safe computation
        var value = input[i];
        
        // Check for invalid values
        if (float.IsNaN(value) || float.IsInfinity(value))
        {
            output[i] = 0.0f;
            return;
        }
        
        output[i] = value * 2.0f;
    }
    
    private static void ExecuteWithReducedMemory(Accelerator accelerator)
    {
        // Implement chunked processing for large datasets
        Console.WriteLine("Executing with reduced memory footprint...");
        // Implementation would process data in smaller chunks
    }
}
```

## Performance Profiling

### Built-in Performance Monitoring

```csharp
public class PerformanceProfiler
{
    private readonly Accelerator accelerator;
    private readonly Dictionary<string, PerformanceMetrics> metrics;
    
    public PerformanceProfiler(Accelerator accelerator)
    {
        this.accelerator = accelerator;
        this.metrics = new Dictionary<string, PerformanceMetrics>();
    }
    
    public T ProfileKernel<T>(string name, Func<T> kernelExecution)
    {
        var stopwatch = Stopwatch.StartNew();
        var memoryBefore = GC.GetTotalMemory(false);
        
        // Execute kernel
        var result = kernelExecution();
        
        // Ensure GPU work is complete
        accelerator.Synchronize();
        
        stopwatch.Stop();
        var memoryAfter = GC.GetTotalMemory(false);
        
        // Record metrics
        var metric = new PerformanceMetrics
        {
            ExecutionTime = stopwatch.Elapsed,
            MemoryUsage = memoryAfter - memoryBefore,
            Timestamp = DateTime.UtcNow
        };
        
        metrics[name] = metric;
        
        Console.WriteLine($"[{name}] Time: {metric.ExecutionTime.TotalMilliseconds:F2}ms, Memory: {metric.MemoryUsage / 1024:F1}KB");
        
        return result;
    }
    
    public void ProfileMemoryBandwidth(string name, int dataSize, int elementSize)
    {
        var totalBytes = (long)dataSize * elementSize * 2; // Read + Write
        
        var metric = metrics[name];
        var bandwidth = totalBytes / metric.ExecutionTime.TotalSeconds / (1024 * 1024 * 1024); // GB/s
        
        Console.WriteLine($"[{name}] Bandwidth: {bandwidth:F2} GB/s");
        
        // Compare to theoretical peak
        var theoreticalPeak = GetTheoreticalBandwidth();
        var efficiency = bandwidth / theoreticalPeak * 100;
        
        Console.WriteLine($"[{name}] Memory efficiency: {efficiency:F1}%");
    }
    
    private double GetTheoreticalBandwidth()
    {
        // Simplified - would need actual hardware specifications
        return accelerator.AcceleratorType switch
        {
            AcceleratorType.CPU => 50.0, // GB/s
            AcceleratorType.Cuda => 900.0, // GB/s for high-end GPU
            AcceleratorType.OpenCL => 500.0, // GB/s
            _ => 100.0
        };
    }
    
    public void GeneratePerformanceReport()
    {
        Console.WriteLine("\n=== Performance Report ===");
        Console.WriteLine($"Accelerator: {accelerator.Name}");
        Console.WriteLine($"Memory Size: {accelerator.MemorySize / (1024 * 1024):F0} MB");
        Console.WriteLine($"Max Threads per Group: {accelerator.MaxNumThreadsPerGroup}");
        
        foreach (var kvp in metrics.OrderBy(m => m.Value.ExecutionTime))
        {
            var name = kvp.Key;
            var metric = kvp.Value;
            
            Console.WriteLine($"\n{name}:");
            Console.WriteLine($"  Execution Time: {metric.ExecutionTime.TotalMilliseconds:F2} ms");
            Console.WriteLine($"  Memory Usage: {metric.MemoryUsage / 1024:F1} KB");
            Console.WriteLine($"  Timestamp: {metric.Timestamp:HH:mm:ss.fff}");
        }
    }
}

public class PerformanceMetrics
{
    public TimeSpan ExecutionTime { get; set; }
    public long MemoryUsage { get; set; }
    public DateTime Timestamp { get; set; }
}
```

### Detailed Memory Profiling

```csharp
public class MemoryProfiler
{
    private readonly Accelerator accelerator;
    private readonly List<MemoryAllocation> allocations;
    
    public MemoryProfiler(Accelerator accelerator)
    {
        this.accelerator = accelerator;
        this.allocations = new List<MemoryAllocation>();
    }
    
    public MemoryBuffer1D<T, Stride1D.Dense> TrackedAllocate1D<T>(int size, string name = null) 
        where T : unmanaged
    {
        var buffer = accelerator.Allocate1D<T>(size);
        var allocation = new MemoryAllocation
        {
            Name = name ?? $"Buffer_{allocations.Count}",
            Size = size * Marshal.SizeOf<T>(),
            Type = typeof(T).Name,
            Timestamp = DateTime.UtcNow,
            Buffer = buffer
        };
        
        allocations.Add(allocation);
        
        Console.WriteLine($"Allocated: {allocation.Name} ({allocation.Size / 1024:F1} KB)");
        
        return buffer;
    }
    
    public void TrackDisposal(MemoryBuffer buffer)
    {
        var allocation = allocations.FirstOrDefault(a => a.Buffer == buffer);
        if (allocation != null)
        {
            allocation.DisposedAt = DateTime.UtcNow;
            var lifetime = allocation.DisposedAt.Value - allocation.Timestamp;
            
            Console.WriteLine($"Disposed: {allocation.Name} (lifetime: {lifetime.TotalMilliseconds:F1}ms)");
        }
    }
    
    public void AnalyzeMemoryUsage()
    {
        Console.WriteLine("\n=== Memory Usage Analysis ===");
        
        var totalAllocated = allocations.Sum(a => a.Size);
        var currentlyAllocated = allocations.Where(a => !a.DisposedAt.HasValue).Sum(a => a.Size);
        var peakUsage = GetPeakMemoryUsage();
        
        Console.WriteLine($"Total allocated: {totalAllocated / (1024 * 1024):F1} MB");
        Console.WriteLine($"Currently allocated: {currentlyAllocated / (1024 * 1024):F1} MB");
        Console.WriteLine($"Peak usage: {peakUsage / (1024 * 1024):F1} MB");
        Console.WriteLine($"Device memory: {accelerator.MemorySize / (1024 * 1024):F1} MB");
        
        var memoryUtilization = (double)peakUsage / accelerator.MemorySize * 100;
        Console.WriteLine($"Peak utilization: {memoryUtilization:F1}%");
        
        // Identify memory leaks
        var leaks = allocations.Where(a => !a.DisposedAt.HasValue).ToList();
        if (leaks.Any())
        {
            Console.WriteLine($"\nPotential memory leaks ({leaks.Count} allocations):");
            foreach (var leak in leaks.Take(5))
            {
                Console.WriteLine($"  {leak.Name}: {leak.Size / 1024:F1} KB (age: {(DateTime.UtcNow - leak.Timestamp).TotalSeconds:F1}s)");
            }
        }
        
        // Allocation patterns
        AnalyzeAllocationPatterns();
    }
    
    private long GetPeakMemoryUsage()
    {
        var timeline = new List<(DateTime Time, long Delta)>();
        
        foreach (var allocation in allocations)
        {
            timeline.Add((allocation.Timestamp, allocation.Size));
            if (allocation.DisposedAt.HasValue)
            {
                timeline.Add((allocation.DisposedAt.Value, -allocation.Size));
            }
        }
        
        timeline.Sort((a, b) => a.Time.CompareTo(b.Time));
        
        long currentUsage = 0;
        long peakUsage = 0;
        
        foreach (var (time, delta) in timeline)
        {
            currentUsage += delta;
            peakUsage = Math.Max(peakUsage, currentUsage);
        }
        
        return peakUsage;
    }
    
    private void AnalyzeAllocationPatterns()
    {
        Console.WriteLine("\nAllocation Patterns:");
        
        var byType = allocations.GroupBy(a => a.Type)
                               .OrderByDescending(g => g.Sum(a => a.Size))
                               .Take(5);
        
        foreach (var group in byType)
        {
            var totalSize = group.Sum(a => a.Size);
            var count = group.Count();
            var avgSize = totalSize / count;
            
            Console.WriteLine($"  {group.Key}: {count} allocations, {totalSize / 1024:F1} KB total, {avgSize / 1024:F1} KB avg");
        }
        
        // Lifetime analysis
        var withLifetime = allocations.Where(a => a.DisposedAt.HasValue).ToList();
        if (withLifetime.Any())
        {
            var avgLifetime = withLifetime.Average(a => (a.DisposedAt.Value - a.Timestamp).TotalMilliseconds);
            Console.WriteLine($"\nAverage allocation lifetime: {avgLifetime:F1} ms");
        }
    }
}

public class MemoryAllocation
{
    public string Name { get; set; }
    public long Size { get; set; }
    public string Type { get; set; }
    public DateTime Timestamp { get; set; }
    public DateTime? DisposedAt { get; set; }
    public MemoryBuffer Buffer { get; set; }
}
```

### Kernel Performance Analysis

```csharp
public class KernelProfiler
{
    public static KernelPerformanceData AnalyzeKernel<T>(
        Accelerator accelerator,
        Action<Index1D, ArrayView<T>> kernel,
        int dataSize,
        string kernelName = null) where T : unmanaged
    {
        kernelName ??= kernel.Method.Name;
        
        // Test different block sizes
        var blockSizes = new[] { 32, 64, 128, 256, 512 };
        var results = new List<BlockSizeResult>();
        
        using var inputBuffer = accelerator.Allocate1D<T>(dataSize);
        
        foreach (var blockSize in blockSizes)
        {
            if (blockSize > accelerator.MaxNumThreadsPerGroup)
                continue;
            
            var numBlocks = (dataSize + blockSize - 1) / blockSize;
            var config = new KernelConfig(numBlocks, blockSize);
            
            try
            {
                var loadedKernel = accelerator.LoadStreamKernel<Index1D, ArrayView<T>>(kernel);
                
                // Warm-up
                loadedKernel(config, inputBuffer.View);
                accelerator.Synchronize();
                
                // Benchmark
                var iterations = 10;
                var stopwatch = Stopwatch.StartNew();
                
                for (int i = 0; i < iterations; i++)
                {
                    loadedKernel(config, inputBuffer.View);
                }
                
                accelerator.Synchronize();
                stopwatch.Stop();
                
                var avgTime = stopwatch.Elapsed.TotalMilliseconds / iterations;
                var occupancy = EstimateOccupancy(accelerator, config);
                
                results.Add(new BlockSizeResult
                {
                    BlockSize = blockSize,
                    NumBlocks = numBlocks,
                    AverageTime = avgTime,
                    EstimatedOccupancy = occupancy
                });
                
                Console.WriteLine($"Block size {blockSize}: {avgTime:F2}ms, occupancy: {occupancy:P1}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Block size {blockSize} failed: {ex.Message}");
            }
        }
        
        // Find optimal configuration
        var optimalResult = results.OrderBy(r => r.AverageTime).FirstOrDefault();
        
        return new KernelPerformanceData
        {
            KernelName = kernelName,
            DataSize = dataSize,
            OptimalBlockSize = optimalResult?.BlockSize ?? 0,
            BestTime = optimalResult?.AverageTime ?? 0,
            BestOccupancy = optimalResult?.EstimatedOccupancy ?? 0,
            AllResults = results
        };
    }
    
    private static double EstimateOccupancy(Accelerator accelerator, KernelConfig config)
    {
        var threadsPerBlock = config.GroupSize.Size;
        var maxThreadsPerSM = accelerator.MaxNumThreadsPerMultiprocessor;
        var blocksPerSM = maxThreadsPerSM / threadsPerBlock;
        var activeThreads = Math.Min(blocksPerSM * threadsPerBlock, maxThreadsPerSM);
        
        return (double)activeThreads / maxThreadsPerSM;
    }
}

public class BlockSizeResult
{
    public int BlockSize { get; set; }
    public int NumBlocks { get; set; }
    public double AverageTime { get; set; }
    public double EstimatedOccupancy { get; set; }
}

public class KernelPerformanceData
{
    public string KernelName { get; set; }
    public int DataSize { get; set; }
    public int OptimalBlockSize { get; set; }
    public double BestTime { get; set; }
    public double BestOccupancy { get; set; }
    public List<BlockSizeResult> AllResults { get; set; }
}
```

## Platform-Specific Profiling

### CUDA Profiling Integration

```csharp
public class CudaProfiler
{
    public static void ProfileWithNvprof(string executablePath, string[] args)
    {
        // Generate nvprof command
        var nvprofArgs = new List<string>
        {
            "--print-gpu-trace",
            "--log-file", "ilgpu_profile.log",
            "--csv",
            executablePath
        };
        nvprofArgs.AddRange(args);
        
        var processInfo = new ProcessStartInfo
        {
            FileName = "nvprof",
            Arguments = string.Join(" ", nvprofArgs),
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        
        try
        {
            using var process = Process.Start(processInfo);
            process.WaitForExit();
            
            if (process.ExitCode == 0)
            {
                Console.WriteLine("CUDA profiling completed successfully");
                Console.WriteLine("Check ilgpu_profile.log for detailed results");
            }
            else
            {
                Console.WriteLine($"nvprof failed with exit code: {process.ExitCode}");
                Console.WriteLine(process.StandardError.ReadToEnd());
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to launch nvprof: {ex.Message}");
        }
    }
    
    public static void EnableCudaEvents(Accelerator accelerator)
    {
        if (accelerator.AcceleratorType != AcceleratorType.Cuda)
            return;
        
        // CUDA event-based timing (requires CUDA backend)
        // This would require ILGPU CUDA-specific extensions
        Console.WriteLine("CUDA events enabled for precise timing");
    }
}
```

## Usage Examples

```csharp
public class ProfilingExample
{
    public static void RunComprehensiveProfiling()
    {
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);
        
        Console.WriteLine("ILGPU Comprehensive Profiling Example");
        Console.WriteLine($"Using: {accelerator.Name}");
        
        // Initialize profilers
        var perfProfiler = new PerformanceProfiler(accelerator);
        var memProfiler = new MemoryProfiler(accelerator);
        
        const int dataSize = 1024 * 1024;
        
        // Profile different kernels
        ProfileVectorAddition(perfProfiler, memProfiler, dataSize);
        ProfileMatrixMultiplication(perfProfiler, memProfiler, 512);
        ProfileReduction(perfProfiler, memProfiler, dataSize);
        
        // Generate comprehensive reports
        perfProfiler.GeneratePerformanceReport();
        memProfiler.AnalyzeMemoryUsage();
        
        // Cross-platform validation
        Console.WriteLine("\nCross-platform validation:");
        CrossPlatformValidator.ValidateAcrossPlatforms(
            context, VectorAddKernel, GenerateTestData(1000));
    }
    
    private static void ProfileVectorAddition(
        PerformanceProfiler perfProfiler,
        MemoryProfiler memProfiler,
        int dataSize)
    {
        perfProfiler.ProfileKernel("VectorAddition", () =>
        {
            using var inputA = memProfiler.TrackedAllocate1D<float>(dataSize, "InputA");
            using var inputB = memProfiler.TrackedAllocate1D<float>(dataSize, "InputB");
            using var output = memProfiler.TrackedAllocate1D<float>(dataSize, "Output");
            
            var kernel = perfProfiler.accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VectorAddKernel);
            
            kernel(dataSize, inputA.View, inputB.View, output.View);
            
            memProfiler.TrackDisposal(inputA);
            memProfiler.TrackDisposal(inputB);
            memProfiler.TrackDisposal(output);
            
            return true;
        });
        
        perfProfiler.ProfileMemoryBandwidth("VectorAddition", dataSize, sizeof(float));
    }
    
    static void VectorAddKernel(
        Index1D index,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> result)
    {
        if (index < result.Length)
        {
            result[index] = a[index] + b[index];
        }
    }
    
    private static void ProfileMatrixMultiplication(
        PerformanceProfiler perfProfiler,
        MemoryProfiler memProfiler,
        int size)
    {
        perfProfiler.ProfileKernel("MatrixMultiplication", () =>
        {
            using var matrixA = memProfiler.TrackedAllocate1D<float>(size * size, "MatrixA");
            using var matrixB = memProfiler.TrackedAllocate1D<float>(size * size, "MatrixB");
            using var result = memProfiler.TrackedAllocate1D<float>(size * size, "MatrixResult");
            
            var kernel = perfProfiler.accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(MatMulKernel);
            
            kernel((size, size), matrixA.View, matrixB.View, result.View, size);
            
            memProfiler.TrackDisposal(matrixA);
            memProfiler.TrackDisposal(matrixB);
            memProfiler.TrackDisposal(result);
            
            return true;
        });
    }
    
    static void MatMulKernel(
        Index2D index,
        ArrayView<float> matrixA,
        ArrayView<float> matrixB,
        ArrayView<float> result,
        int size)
    {
        var row = index.Y;
        var col = index.X;
        
        if (row < size && col < size)
        {
            float sum = 0.0f;
            for (int k = 0; k < size; k++)
            {
                sum += matrixA[row * size + k] * matrixB[k * size + col];
            }
            result[row * size + col] = sum;
        }
    }
    
    private static void ProfileReduction(
        PerformanceProfiler perfProfiler,
        MemoryProfiler memProfiler,
        int dataSize)
    {
        // Analyze kernel performance characteristics
        var kernelData = KernelProfiler.AnalyzeKernel(
            perfProfiler.accelerator, ReductionKernel, dataSize, "Reduction");
        
        Console.WriteLine($"Optimal block size for reduction: {kernelData.OptimalBlockSize}");
        Console.WriteLine($"Best performance: {kernelData.BestTime:F2}ms");
    }
    
    static void ReductionKernel(Index1D index, ArrayView<float> data)
    {
        // Simplified reduction kernel
        if (index < data.Length)
        {
            data[index] = data[index] + 1.0f;
        }
    }
    
    private static float[] GenerateTestData(int size)
    {
        var random = new Random(42);
        return Enumerable.Range(0, size).Select(_ => (float)random.NextDouble()).ToArray();
    }
}
```

## Best Practices

1. **Always Profile**: Measure performance before and after optimizations
2. **Use CPU Debugging**: Leverage CPU accelerator for comprehensive debugging
3. **Cross-Platform Validation**: Test algorithms across different backends
4. **Memory Tracking**: Monitor memory usage patterns and identify leaks
5. **Kernel Analysis**: Optimize block sizes and occupancy systematically
6. **Platform-Specific Tools**: Use native profiling tools when available

## Common Issues and Solutions

1. **Timing Inaccuracies**: Always synchronize accelerator before timing measurements
2. **Memory Leaks**: Use proper disposal patterns and track allocations
3. **Platform Differences**: Validate results across different accelerator types
4. **Performance Regression**: Establish baseline measurements and track changes
5. **Debug vs Release**: Be aware of performance differences between build configurations

---

Effective profiling and debugging enable systematic performance optimization and reliable ILGPU application development across diverse hardware platforms.