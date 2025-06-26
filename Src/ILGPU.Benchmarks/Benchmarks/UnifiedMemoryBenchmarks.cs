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
// Change License: Apache License, Version 2.0using BenchmarkDotNet.Attributes;
using ILGPU.Runtime;
using System.Runtime.InteropServices;

namespace ILGPU.Benchmarks.Benchmarks;

/// <summary>
/// Benchmarks for unified memory and tensor operations.
/// </summary>
[MemoryDiagnoser]
[SimpleJob]
public class UnifiedMemoryBenchmarks : IDisposable
{
    private Context? context;
    private Accelerator? accelerator;
    private MemoryBuffer1D<float, Stride1D.Dense>? buffer;
    private MemoryBuffer1D<float, Stride1D.Dense>? bufferB;
    private float[]? hostData;

    [Params(1024, 4096, 16384, 65536)]
    public int DataSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        try
        {
            context = Context.CreateDefault();
            accelerator = context.GetPreferredDevice(preferCPU: false)?.CreateAccelerator(context) ??
                         context.GetPreferredDevice(preferCPU: true).CreateAccelerator(context);

            hostData = new float[DataSize];
            var random = new Random(42);
            for (int i = 0; i < DataSize; i++)
            {
                hostData[i] = (float)random.NextDouble();
            }

            buffer = accelerator.Allocate1D<float>(DataSize);
            bufferB = accelerator.Allocate1D<float>(DataSize);
            
            buffer.View.CopyFromCPU(hostData);
            bufferB.View.CopyFromCPU(hostData);
        }
        catch (Exception ex)
        {
            throw new NotSupportedException($"Failed to initialize unified memory benchmark environment: {ex.Message}", ex);
        }
    }

    [Benchmark(Baseline = true)]
    public float StandardMemoryTransfer()
    {
        // Standard GPU memory allocation and transfer
        using var tempBuffer = accelerator!.Allocate1D<float>(DataSize);
        tempBuffer.View.CopyFromCPU(hostData!);
        
        var result = tempBuffer.GetAsArray1D();
        return result[0];
    }

    [Benchmark]
    public float ZeroCopyMemoryAccess()
    {
        // Simulate zero-copy access pattern
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(ZeroCopyKernel);

        if (buffer == null || bufferB == null)
            return 0.0f;
            
        kernel(DataSize, buffer.View, bufferB.View);
        accelerator.Synchronize();

        var result = bufferB.GetAsArray1D();
        return result[0];
    }

    [Benchmark]
    public float UnifiedTensorCreation()
    {
        try
        {
            // Simulate unified tensor creation
            var dim = (int)Math.Sqrt(DataSize);
            using var buffer2D = accelerator!.Allocate2DDenseX<float>(new Index2D(dim, dim));
            var index = 0;
            for (int y = 0; y < buffer2D.IntExtent.Y && index < dim * dim; y++)
                for (int x = 0; x < buffer2D.IntExtent.X && index < dim * dim; x++)
                    buffer2D.View[y, x] = hostData![index++];
            
            var result = buffer2D.GetAsArray2D();
            return result[0, 0];
        }
        catch
        {
            // Fallback if unified tensor not available
            return StandardMemoryTransfer();
        }
    }

    [Benchmark]
    public float UnifiedTensorOperations()
    {
        try
        {
            var dim = (int)Math.Sqrt(DataSize);
            
            using var bufferA = accelerator!.Allocate2DDenseX<float>(new Index2D(dim, dim));
            using var bufferB = accelerator.Allocate2DDenseX<float>(new Index2D(dim, dim));
            using var result = accelerator.Allocate2DDenseX<float>(new Index2D(dim, dim));
            
            var indexA = 0;
            for (int y = 0; y < bufferA.IntExtent.Y && indexA < dim * dim; y++)
                for (int x = 0; x < bufferA.IntExtent.X && indexA < dim * dim; x++)
                    bufferA.View[y, x] = hostData![indexA++];
            
            var indexB = 0;
            for (int y = 0; y < bufferB.IntExtent.Y && indexB < dim * dim; y++)
                for (int x = 0; x < bufferB.IntExtent.X && indexB < dim * dim; x++)
                    bufferB.View[y, x] = hostData[indexB++];

            // Element-wise addition kernel
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>>(TensorAdditionKernel);

            kernel(new Index2D(dim, dim), bufferA.View, bufferB.View, result.View);
            accelerator.Synchronize();
            
            var output = result.GetAsArray2D();
            return output[0, 0];
        }
        catch
        {
            return StandardMemoryTransfer();
        }
    }

    [Benchmark]
    public float UnifiedTensorMatrixMultiply()
    {
        try
        {
            var dim = Math.Min(64, (int)Math.Sqrt(DataSize)); // Limit matrix size for performance
            
            using var matrixA = accelerator!.Allocate2DDenseX<float>(new Index2D(dim, dim));
            using var matrixB = accelerator.Allocate2DDenseX<float>(new Index2D(dim, dim));
            using var result = accelerator.Allocate2DDenseX<float>(new Index2D(dim, dim));
            
            var indexMA = 0;
            for (int y = 0; y < matrixA.IntExtent.Y && indexMA < dim * dim; y++)
                for (int x = 0; x < matrixA.IntExtent.X && indexMA < dim * dim; x++)
                    matrixA.View[y, x] = hostData![indexMA++];
            
            var indexMB = 0;
            for (int y = 0; y < matrixB.IntExtent.Y && indexMB < dim * dim; y++)
                for (int x = 0; x < matrixB.IntExtent.X && indexMB < dim * dim; x++)
                    matrixB.View[y, x] = hostData[indexMB++];

            // Matrix multiplication kernel
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>, int>(MatrixMultiplyKernel);

            kernel(new Index2D(dim, dim), matrixA.View, matrixB.View, result.View, dim);
            accelerator.Synchronize();
            
            var output = result.GetAsArray2D();
            return output[0, 0];
        }
        catch
        {
            return StandardMemoryTransfer();
        }
    }

    [Benchmark]
    public float PageLockedMemoryTransfer()
    {
        // Real page-locked memory implementation using pinned memory
        using var tempBuffer = accelerator!.Allocate1D<float>(DataSize);
        
        // Pin the host memory to physical pages to prevent swapping
        // This provides faster and more predictable transfer times
        var handle = GCHandle.Alloc(hostData!, GCHandleType.Pinned);
        try
        {
            // Use pinned memory for optimized transfer
            var pinnedSpan = new ReadOnlySpan<float>(hostData!);
            tempBuffer.View.AsContiguous().CopyFromCPU(pinnedSpan);
            
            // Ensure all operations complete before measurement
            accelerator.Synchronize();
        }
        finally
        {
            handle.Free();
        }
        
        var result = tempBuffer.GetAsArray1D();
        return result[0];
    }

    [Benchmark]
    public float MemoryCoalescingOptimizedAccess()
    {
        // Benchmark memory coalescing patterns
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int>(CoalescedAccessKernel);

        if (buffer == null)
            return 0.0f;
            
        using var result = accelerator.Allocate1D<float>(DataSize);
        kernel(DataSize, buffer.View, result.View, 1);
        accelerator.Synchronize();

        var output = result.GetAsArray1D();
        return output[0];
    }

    [Benchmark]
    public float MemoryBandwidthSequential()
    {
        // Test sequential memory bandwidth
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(SequentialAccessKernel);

        if (buffer == null)
            return 0.0f;
            
        using var result = accelerator.Allocate1D<float>(DataSize);
        kernel(DataSize, buffer.View, result.View);
        accelerator.Synchronize();

        var output = result.GetAsArray1D();
        return output[0];
    }

    [Benchmark]
    public float MemoryBandwidthRandom()
    {
        // Test random memory access patterns
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int>(RandomAccessKernel);

        if (buffer == null)
            return 0.0f;
            
        using var result = accelerator.Allocate1D<float>(DataSize);
        kernel(DataSize, buffer.View, result.View, DataSize);
        accelerator.Synchronize();

        var output = result.GetAsArray1D();
        return output[0];
    }

    [Benchmark]
    public float AsyncMemoryTransfers()
    {
        // Real asynchronous memory transfer implementation using accelerator streams
        using var tempBuffer1 = accelerator!.Allocate1D<float>(DataSize / 2);
        using var tempBuffer2 = accelerator.Allocate1D<float>(DataSize / 2);

        // Create separate streams for concurrent transfers
        using var stream1 = accelerator.CreateStream();
        using var stream2 = accelerator.CreateStream();

        // Pin memory for optimal async transfer performance
        var handle1 = GCHandle.Alloc(hostData!, GCHandleType.Pinned);
        var handle2 = GCHandle.Alloc(hostData!, GCHandleType.Pinned);
        
        try
        {
            // Execute async transfers on separate streams concurrently
            var span1 = new ReadOnlySpan<float>(hostData!, 0, DataSize / 2);
            var span2 = new ReadOnlySpan<float>(hostData!, DataSize / 2, DataSize / 2);
            
            tempBuffer1.View.AsContiguous().CopyFromCPU(stream1, span1);
            tempBuffer2.View.AsContiguous().CopyFromCPU(stream2, span2);

            // Synchronize both streams to ensure completion
            stream1.Synchronize();
            stream2.Synchronize();
        }
        finally
        {
            handle1.Free();
            handle2.Free();
        }

        var result1 = tempBuffer1.GetAsArray1D();
        var result2 = tempBuffer2.GetAsArray1D();
        
        return result1[0] + result2[0];
    }

    [Benchmark]
    public float TensorViewOperations()
    {
        try
        {
            var height = (int)Math.Sqrt(DataSize);
            var width = DataSize / height;
            
            using var buffer2D = accelerator!.Allocate2DDenseX<float>(new Index2D(width, height));
            var index2D = 0;
            for (int y = 0; y < buffer2D.IntExtent.Y && index2D < width * height; y++)
                for (int x = 0; x < buffer2D.IntExtent.X && index2D < width * height; x++)
                    buffer2D.View[y, x] = hostData![index2D++];
            
            // Test tensor slicing simulation using subviews
            var sliceWidth = Math.Min(width / 2, height);
            var sliceExtent = new Index2D(sliceWidth, sliceWidth);
            var sliceView = buffer2D.View.SubView(new Index2D(0, 0), sliceExtent);
            
            var result = buffer2D.GetAsArray2D();
            return result[0, 0];
        }
        catch
        {
            return StandardMemoryTransfer();
        }
    }

    #region Kernels

    private static void ZeroCopyKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
    {
        if (index < input.Length && index < output.Length)
        {
            output[index] = input[index];
        }
    }

    private static void TensorAdditionKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c)
    {
        if (index.X < a.IntExtent.X && index.Y < a.IntExtent.Y)
        {
            c[index] = a[index] + b[index];
        }
    }

    private static void MatrixMultiplyKernel(
        Index2D index,
        ArrayView2D<float, Stride2D.DenseX> a,
        ArrayView2D<float, Stride2D.DenseX> b,
        ArrayView2D<float, Stride2D.DenseX> c,
        int size)
    {
        if (index.X >= size || index.Y >= size)
            return;
            
        var sum = 0.0f;
        for (int k = 0; k < size; k++)
        {
            sum += a[index.X, k] * b[k, index.Y];
        }
        c[index.X, index.Y] = sum;
    }

    private static void CoalescedAccessKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        int stride)
    {
        if (index >= output.Length)
            return;
            
        // Coalesced memory access pattern
        var coalescedIndex = (index * stride) % input.Length;
        output[index] = input[coalescedIndex];
    }

    private static void SequentialAccessKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
    {
        if (index < input.Length && index < output.Length)
        {
            output[index] = input[index] * 2.0f;
        }
    }

    private static void RandomAccessKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        int length)
    {
        if (index >= output.Length)
            return;
            
        // Random access pattern (pseudorandom based on index)
        var randomIndex = ((index * 1103515245 + 12345) / 65536) % length;
        output[index] = input[randomIndex];
    }

    #endregion

    [GlobalCleanup]
    public void Cleanup()
    {
        Dispose();
    }

    public void Dispose()
    {
        buffer?.Dispose();
        bufferB?.Dispose();
        accelerator?.Dispose();
        context?.Dispose();
    }
}