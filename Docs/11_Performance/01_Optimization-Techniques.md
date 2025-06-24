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

# Performance Optimization Techniques

## Overview

This guide covers comprehensive performance optimization strategies for ILGPU applications, including memory optimization, algorithmic improvements, and platform-specific tuning. These techniques enable maximum performance extraction from diverse hardware accelerators.

## Technical Background

### Performance Bottlenecks

Common performance limitations in GPU computing:

- **Memory bandwidth**: Often the primary limiting factor for many algorithms
- **Occupancy**: Insufficient parallel threads to hide memory latency
- **Divergence**: Control flow differences between threads in the same warp
- **Synchronization overhead**: Frequent barriers and atomic operations
- **Memory coalescing**: Non-optimal memory access patterns

### ILGPU Performance Optimization Strategy

ILGPU provides multiple optimization approaches:

1. **Memory access optimization**: Coalesced memory patterns and caching strategies
2. **Occupancy optimization**: Thread block sizing and register usage management
3. **Algorithmic optimization**: Algorithm selection based on data characteristics
4. **Platform-specific optimization**: Hardware-specific code generation and tuning

## Memory Optimization

### Memory Coalescing

```csharp
using ILGPU;
using ILGPU.Runtime;

public static class MemoryOptimization
{
    // Optimized: Coalesced memory access
    static void CoalescedKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> input,
        ArrayView2D<float, Stride2D.DenseX> output)
    {
        var x = index.X;
        var y = index.Y;
        
        if (x < input.Width && y < input.Height)
        {
            // Consecutive threads access consecutive memory locations
            output[y, x] = input[y, x] * 2.0f;
        }
    }
    
    // Unoptimized: Non-coalesced memory access
    static void NonCoalescedKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseY> input,
        ArrayView2D<float, Stride2D.DenseY> output)
    {
        var x = index.X;
        var y = index.Y;
        
        if (x < input.Width && y < input.Height)
        {
            // Consecutive threads access strided memory locations
            output[x, y] = input[x, y] * 2.0f;
        }
    }
    
    // Optimized: Blocked memory access for cache efficiency
    static void BlockedMemoryKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        int blockSize)
    {
        var globalId = (int)index;
        var numBlocks = (input.Length + blockSize - 1) / blockSize;
        
        for (int block = 0; block < numBlocks; block++)
        {
            var blockStart = block * blockSize;
            var blockEnd = Math.Min(blockStart + blockSize, input.Length);
            
            // Process block with good spatial locality
            for (int i = blockStart + (globalId % blockSize); i < blockEnd; i += blockSize)
            {
                output[i] = input[i] * 2.0f;
            }
        }
    }
}
```

### Shared Memory Optimization

```csharp
public static class SharedMemoryOptimization
{
    // Matrix transpose with shared memory optimization
    static void OptimizedTransposeKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> input,
        ArrayView2D<float, Stride2D.DenseX> output)
    {
        const int tileSize = 32;
        var sharedMemory = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
            new Index2D(tileSize, tileSize + 1)); // +1 to avoid bank conflicts
        
        var blockX = Group.IdxX;
        var blockY = Group.IdxY;
        var threadX = Group.DimX;
        var threadY = Group.DimY;
        
        var globalX = blockX * tileSize + threadX;
        var globalY = blockY * tileSize + threadY;
        
        // Load tile into shared memory
        if (globalX < input.Width && globalY < input.Height)
        {
            sharedMemory[threadY, threadX] = input[globalY, globalX];
        }
        
        Group.Barrier();
        
        // Write transposed tile to output
        var outputX = blockY * tileSize + threadX;
        var outputY = blockX * tileSize + threadY;
        
        if (outputX < output.Width && outputY < output.Height)
        {
            output[outputY, outputX] = sharedMemory[threadX, threadY];
        }
    }
    
    // Reduction with shared memory
    static void SharedMemoryReductionKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        var sharedData = SharedMemory.Allocate1D<float>(Group.DimX);
        var tid = Group.IdxX;
        var globalId = Grid.GlobalIndex.X;
        
        // Load data into shared memory
        sharedData[tid] = globalId < input.Length ? input[globalId] : 0.0f;
        Group.Barrier();
        
        // Reduction in shared memory
        for (int stride = Group.DimX / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                sharedData[tid] += sharedData[tid + stride];
            }
            Group.Barrier();
        }
        
        // Write result
        if (tid == 0)
        {
            output[Group.IdxY] = sharedData[0];
        }
    }
}
```

## Occupancy Optimization

### Thread Block Sizing

```csharp
public class OccupancyOptimizer
{
    public static KernelConfig OptimizeBlockSize<T>(
        Accelerator accelerator,
        Action<Index1D, T> kernel,
        int dataSize) where T : struct
    {
        var bestConfig = new KernelConfig(1, 1);
        var bestOccupancy = 0.0;
        
        // Test different block sizes
        var blockSizes = new[] { 32, 64, 128, 256, 512, 1024 };
        
        foreach (var blockSize in blockSizes)
        {
            if (blockSize > accelerator.MaxNumThreadsPerGroup)
                continue;
            
            var numBlocks = (dataSize + blockSize - 1) / blockSize;
            var config = new KernelConfig(numBlocks, blockSize);
            
            // Estimate occupancy
            var occupancy = EstimateOccupancy(accelerator, config);
            
            if (occupancy > bestOccupancy)
            {
                bestOccupancy = occupancy;
                bestConfig = config;
            }
        }
        
        return bestConfig;
    }
    
    private static double EstimateOccupancy(Accelerator accelerator, KernelConfig config)
    {
        // Simplified occupancy calculation
        var threadsPerBlock = config.GroupSize.Size;
        var blocksPerSM = accelerator.MaxNumThreadsPerMultiprocessor / threadsPerBlock;
        var activeBlocks = Math.Min(blocksPerSM, config.GridSize.Size);
        
        return (double)(activeBlocks * threadsPerBlock) / accelerator.MaxNumThreadsPerMultiprocessor;
    }
    
    // Dynamic block size selection based on problem size
    public static KernelConfig GetAdaptiveConfig(int problemSize, int maxThreadsPerGroup)
    {
        // Use heuristics based on problem characteristics
        int blockSize;
        
        if (problemSize < 1024)
        {
            blockSize = Math.Min(problemSize, 64); // Small problems
        }
        else if (problemSize < 65536)
        {
            blockSize = 128; // Medium problems
        }
        else
        {
            blockSize = 256; // Large problems
        }
        
        blockSize = Math.Min(blockSize, maxThreadsPerGroup);
        var numBlocks = (problemSize + blockSize - 1) / blockSize;
        
        return new KernelConfig(numBlocks, blockSize);
    }
}
```

### Register Usage Optimization

```csharp
public static class RegisterOptimization
{
    // Optimized kernel with minimal register usage
    static void LowRegisterKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        var i = (int)index;
        if (i < input.Length)
        {
            // Minimize register pressure by avoiding temporary variables
            output[i] = MathF.Sin(input[i]) + MathF.Cos(input[i]);
        }
    }
    
    // High register usage kernel (avoid this pattern)
    static void HighRegisterKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        var i = (int)index;
        if (i < input.Length)
        {
            // Multiple temporary variables increase register pressure
            var temp1 = input[i];
            var temp2 = MathF.Sin(temp1);
            var temp3 = MathF.Cos(temp1);
            var temp4 = temp2 + temp3;
            var temp5 = temp4 * 2.0f;
            var temp6 = temp5 - 1.0f;
            output[i] = temp6;
        }
    }
    
    // Loop unrolling for better register utilization
    static void UnrolledLoopKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        int stride)
    {
        var baseIndex = (int)index * 4; // Process 4 elements per thread
        
        // Manual loop unrolling
        if (baseIndex < input.Length)
            output[baseIndex] = input[baseIndex] * 2.0f;
        
        if (baseIndex + 1 < input.Length)
            output[baseIndex + 1] = input[baseIndex + 1] * 2.0f;
        
        if (baseIndex + 2 < input.Length)
            output[baseIndex + 2] = input[baseIndex + 2] * 2.0f;
        
        if (baseIndex + 3 < input.Length)
            output[baseIndex + 3] = input[baseIndex + 3] * 2.0f;
    }
}
```

## Algorithmic Optimization

### Parallel Reduction Patterns

```csharp
public static class ReductionOptimization
{
    // Optimized parallel reduction
    static void OptimizedReductionKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        var sharedData = SharedMemory.Allocate1D<float>(Group.DimX);
        var tid = Group.IdxX;
        var globalId = Grid.GlobalIndex.X;
        
        // Load and reduce multiple elements per thread
        var sum = 0.0f;
        var stride = Grid.DimX;
        
        for (int i = globalId; i < input.Length; i += stride)
        {
            sum += input[i];
        }
        
        sharedData[tid] = sum;
        Group.Barrier();
        
        // Unrolled reduction in shared memory
        if (Group.DimX >= 1024)
        {
            if (tid < 512) sharedData[tid] += sharedData[tid + 512];
            Group.Barrier();
        }
        
        if (Group.DimX >= 512)
        {
            if (tid < 256) sharedData[tid] += sharedData[tid + 256];
            Group.Barrier();
        }
        
        if (Group.DimX >= 256)
        {
            if (tid < 128) sharedData[tid] += sharedData[tid + 128];
            Group.Barrier();
        }
        
        if (Group.DimX >= 128)
        {
            if (tid < 64) sharedData[tid] += sharedData[tid + 64];
            Group.Barrier();
        }
        
        // Final warp reduction (no synchronization needed)
        if (tid < 32)
        {
            if (Group.DimX >= 64) sharedData[tid] += sharedData[tid + 32];
            if (Group.DimX >= 32) sharedData[tid] += sharedData[tid + 16];
            if (Group.DimX >= 16) sharedData[tid] += sharedData[tid + 8];
            if (Group.DimX >= 8) sharedData[tid] += sharedData[tid + 4];
            if (Group.DimX >= 4) sharedData[tid] += sharedData[tid + 2];
            if (Group.DimX >= 2) sharedData[tid] += sharedData[tid + 1];
        }
        
        if (tid == 0)
        {
            output[Group.IdxY] = sharedData[0];
        }
    }
    
    // Two-phase reduction for very large arrays
    public static float ReduceLargeArray(
        Accelerator accelerator,
        MemoryBuffer1D<float, Stride1D.Dense> input)
    {
        var size = input.Length;
        var blockSize = 256;
        var numBlocks = (int)((size + blockSize - 1) / blockSize);
        
        using var tempBuffer = accelerator.Allocate1D<float>(numBlocks);
        
        // Phase 1: Reduce within blocks
        var kernel1 = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(OptimizedReductionKernel);
        
        kernel1(new KernelConfig(numBlocks, blockSize), input.View, tempBuffer.View);
        accelerator.Synchronize();
        
        // Phase 2: Reduce block results
        if (numBlocks > 1)
        {
            return ReduceLargeArray(accelerator, tempBuffer);
        }
        else
        {
            return tempBuffer.GetAsArray1D()[0];
        }
    }
}
```

### Matrix Multiplication Optimization

```csharp
public static class MatrixOptimization
{
    // Tiled matrix multiplication
    static void TiledMatMulKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> matrixA,
        ArrayView2D<float, Stride2D.DenseX> matrixB,
        ArrayView2D<float, Stride2D.DenseX> result)
    {
        const int tileSize = 16;
        var sharedA = SharedMemory.Allocate2D<float, Stride2D.DenseX>(new Index2D(tileSize, tileSize));
        var sharedB = SharedMemory.Allocate2D<float, Stride2D.DenseX>(new Index2D(tileSize, tileSize));
        
        var row = index.Y;
        var col = index.X;
        var localRow = Group.IdxY;
        var localCol = Group.IdxX;
        
        float sum = 0.0f;
        
        // Process tiles
        var numTiles = (matrixA.Width + tileSize - 1) / tileSize;
        
        for (int tile = 0; tile < numTiles; tile++)
        {
            // Load tiles into shared memory
            var aCol = tile * tileSize + localCol;
            var bRow = tile * tileSize + localRow;
            
            sharedA[localRow, localCol] = (row < matrixA.Height && aCol < matrixA.Width) ?
                matrixA[row, aCol] : 0.0f;
            
            sharedB[localRow, localCol] = (bRow < matrixB.Height && col < matrixB.Width) ?
                matrixB[bRow, col] : 0.0f;
            
            Group.Barrier();
            
            // Compute partial result
            for (int k = 0; k < tileSize; k++)
            {
                sum += sharedA[localRow, k] * sharedB[k, localCol];
            }
            
            Group.Barrier();
        }
        
        // Store result
        if (row < result.Height && col < result.Width)
        {
            result[row, col] = sum;
        }
    }
    
    // Vectorized matrix operations
    static void VectorizedMatMulKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> matrixA,
        ArrayView2D<float, Stride2D.DenseX> matrixB,
        ArrayView2D<float, Stride2D.DenseX> result)
    {
        var row = index.Y;
        var col = index.X * 4; // Process 4 columns per thread
        
        if (row >= result.Height || col >= result.Width)
            return;
        
        // Vectorized computation
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        
        for (int k = 0; k < matrixA.Width; k++)
        {
            var aVal = matrixA[row, k];
            
            if (col < matrixB.Width) sum0 += aVal * matrixB[k, col];
            if (col + 1 < matrixB.Width) sum1 += aVal * matrixB[k, col + 1];
            if (col + 2 < matrixB.Width) sum2 += aVal * matrixB[k, col + 2];
            if (col + 3 < matrixB.Width) sum3 += aVal * matrixB[k, col + 3];
        }
        
        // Store results
        if (col < result.Width) result[row, col] = sum0;
        if (col + 1 < result.Width) result[row, col + 1] = sum1;
        if (col + 2 < result.Width) result[row, col + 2] = sum2;
        if (col + 3 < result.Width) result[row, col + 3] = sum3;
    }
}
```

## Platform-Specific Optimization

### CUDA-Specific Optimizations

```csharp
public static class CudaOptimizations
{
    // Warp-level primitives
    static void WarpOptimizedKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        var warpSize = 32;
        var laneId = Group.IdxX % warpSize;
        var warpId = Group.IdxX / warpSize;
        
        if (index < input.Length)
        {
            var value = input[index];
            
            // Warp-level reduction using shuffle operations
            for (int offset = warpSize / 2; offset > 0; offset /= 2)
            {
                value += Warp.Shuffle(value, laneId + offset);
            }
            
            // First thread in warp writes result
            if (laneId == 0)
            {
                output[warpId] = value;
            }
        }
    }
    
    // Cooperative groups optimization
    static void CooperativeGroupKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        // Use cooperative groups for flexible synchronization
        var tile = CooperativeGroup.ThisThreadBlock();
        var warp = CooperativeGroup.CoalescedThreads();
        
        if (index < input.Length)
        {
            var value = input[index];
            
            // Warp-level operation
            var warpSum = warp.Reduce(value, CooperativeGroup.ReduceOp.Add);
            
            if (warp.ThreadRank() == 0)
            {
                output[index / 32] = warpSum;
            }
        }
    }
}
```

### CPU-Specific Optimizations

```csharp
public static class CPUOptimizations
{
    // SIMD-friendly algorithms
    static void SIMDOptimizedKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
    {
        var i = (int)index * 4; // Process 4 elements for SIMD
        
        if (i + 3 < input.Length)
        {
            // This pattern encourages SIMD vectorization
            output[i] = input[i] * 2.0f;
            output[i + 1] = input[i + 1] * 2.0f;
            output[i + 2] = input[i + 2] * 2.0f;
            output[i + 3] = input[i + 3] * 2.0f;
        }
        else
        {
            // Handle remaining elements
            for (int j = i; j < input.Length; j++)
            {
                output[j] = input[j] * 2.0f;
            }
        }
    }
    
    // Cache-optimized algorithms
    static void CacheOptimizedKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> input,
        ArrayView2D<float, Stride2D.DenseX> output)
    {
        const int blockSize = 64; // Cache line size consideration
        
        var blockX = index.X * blockSize;
        var blockY = index.Y * blockSize;
        
        // Process in cache-friendly blocks
        for (int y = blockY; y < Math.Min(blockY + blockSize, input.Height); y++)
        {
            for (int x = blockX; x < Math.Min(blockX + blockSize, input.Width); x++)
            {
                output[y, x] = input[y, x] * 2.0f;
            }
        }
    }
}
```

## Performance Measurement

### Benchmarking Framework

```csharp
public class PerformanceBenchmark
{
    public static BenchmarkResult BenchmarkKernel<T>(
        Accelerator accelerator,
        Action<Index1D, ArrayView<T>> kernel,
        int dataSize,
        int iterations = 100) where T : unmanaged
    {
        // Allocate test data
        using var inputBuffer = accelerator.Allocate1D<T>(dataSize);
        using var outputBuffer = accelerator.Allocate1D<T>(dataSize);
        
        // Load kernel
        var loadedKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>>(kernel);
        
        // Warm-up
        loadedKernel(dataSize, inputBuffer.View);
        accelerator.Synchronize();
        
        // Benchmark
        var stopwatch = Stopwatch.StartNew();
        
        for (int i = 0; i < iterations; i++)
        {
            loadedKernel(dataSize, inputBuffer.View);
        }
        
        accelerator.Synchronize();
        stopwatch.Stop();
        
        var avgTime = stopwatch.Elapsed.TotalMilliseconds / iterations;
        var throughput = dataSize / (avgTime / 1000.0); // Elements per second
        var bandwidth = throughput * Marshal.SizeOf<T>() / (1024 * 1024 * 1024); // GB/s
        
        return new BenchmarkResult
        {
            AverageTime = avgTime,
            Throughput = throughput,
            Bandwidth = bandwidth,
            DataSize = dataSize,
            Iterations = iterations
        };
    }
    
    public static void CompareBenchmarks(
        string name1, BenchmarkResult result1,
        string name2, BenchmarkResult result2)
    {
        Console.WriteLine($"Performance Comparison:");
        Console.WriteLine($"{name1}: {result1.AverageTime:F2}ms, {result1.Bandwidth:F2} GB/s");
        Console.WriteLine($"{name2}: {result2.AverageTime:F2}ms, {result2.Bandwidth:F2} GB/s");
        
        var speedup = result1.AverageTime / result2.AverageTime;
        Console.WriteLine($"Speedup: {speedup:F2}x");
    }
}

public class BenchmarkResult
{
    public double AverageTime { get; set; }
    public double Throughput { get; set; }
    public double Bandwidth { get; set; }
    public int DataSize { get; set; }
    public int Iterations { get; set; }
}
```

## Usage Examples

```csharp
public class OptimizationExample
{
    public static void RunOptimizationComparison()
    {
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);
        
        const int dataSize = 1024 * 1024;
        
        Console.WriteLine("ILGPU Performance Optimization Comparison");
        
        // Compare different reduction implementations
        var basicResult = PerformanceBenchmark.BenchmarkKernel(
            accelerator, BasicReductionKernel, dataSize);
        
        var optimizedResult = PerformanceBenchmark.BenchmarkKernel(
            accelerator, ReductionOptimization.OptimizedReductionKernel, dataSize);
        
        PerformanceBenchmark.CompareBenchmarks(
            "Basic Reduction", basicResult,
            "Optimized Reduction", optimizedResult);
        
        // Test different block sizes
        TestBlockSizeOptimization(accelerator, dataSize);
        
        // Memory coalescing comparison
        TestMemoryCoalescing(accelerator, 1024, 1024);
    }
    
    static void BasicReductionKernel(Index1D index, ArrayView<float> data)
    {
        // Simple, unoptimized reduction
        var sum = 0.0f;
        for (int i = 0; i < data.Length; i++)
        {
            sum += data[i];
        }
        data[0] = sum; // Store result
    }
    
    private static void TestBlockSizeOptimization(Accelerator accelerator, int dataSize)
    {
        Console.WriteLine("\nBlock Size Optimization:");
        
        var blockSizes = new[] { 32, 64, 128, 256, 512 };
        
        foreach (var blockSize in blockSizes)
        {
            var numBlocks = (dataSize + blockSize - 1) / blockSize;
            var config = new KernelConfig(numBlocks, blockSize);
            
            using var buffer = accelerator.Allocate1D<float>(dataSize);
            var kernel = accelerator.LoadStreamKernel<Index1D, ArrayView<float>>(SimpleKernel);
            
            var stopwatch = Stopwatch.StartNew();
            kernel(config, buffer.View);
            accelerator.Synchronize();
            stopwatch.Stop();
            
            Console.WriteLine($"Block size {blockSize}: {stopwatch.Elapsed.TotalMilliseconds:F2}ms");
        }
    }
    
    static void SimpleKernel(Index1D index, ArrayView<float> data)
    {
        if (index < data.Length)
        {
            data[index] = data[index] * 2.0f;
        }
    }
    
    private static void TestMemoryCoalescing(Accelerator accelerator, int width, int height)
    {
        Console.WriteLine("\nMemory Coalescing Comparison:");
        
        using var input = accelerator.Allocate2D<float, Stride2D.DenseX>(new Index2D(width, height));
        using var output = accelerator.Allocate2D<float, Stride2D.DenseX>(new Index2D(width, height));
        
        // Test coalesced access
        var coalescedKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(
            MemoryOptimization.CoalescedKernel);
        
        var stopwatch = Stopwatch.StartNew();
        coalescedKernel((width, height), input.View, output.View);
        accelerator.Synchronize();
        stopwatch.Stop();
        
        Console.WriteLine($"Coalesced access: {stopwatch.Elapsed.TotalMilliseconds:F2}ms");
        
        // Note: Non-coalesced test would require different memory layout
    }
}
```

## Best Practices

1. **Memory Access Patterns**: Always prioritize coalesced memory access
2. **Occupancy**: Balance thread count with register/shared memory usage
3. **Algorithm Selection**: Choose algorithms that minimize divergence
4. **Platform Awareness**: Use platform-specific optimizations when available
5. **Profiling**: Measure performance regularly and identify bottlenecks
6. **Iterative Optimization**: Apply optimizations incrementally and measure impact

## Common Pitfalls

1. **Premature Optimization**: Profile first, then optimize
2. **Over-synchronization**: Minimize barriers and atomic operations
3. **Register Pressure**: Avoid excessive temporary variables
4. **Memory Bandwidth**: Consider memory bandwidth limits in algorithm design
5. **Platform Assumptions**: Test on target hardware configurations

---

Performance optimization in ILGPU requires understanding both the algorithms and the underlying hardware characteristics. Systematic application of these techniques can yield significant performance improvements across diverse accelerator platforms.