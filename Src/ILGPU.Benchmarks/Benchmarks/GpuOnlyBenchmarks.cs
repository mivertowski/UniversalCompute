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
/// Benchmarks for GPU-only operations without CPU fallback.
/// </summary>
[MemoryDiagnoser]
[SimpleJob]
public class GpuOnlyBenchmarks : IDisposable
{
    private Context? context;
    private Accelerator? accelerator;
    private MemoryBuffer1D<float, Stride1D.Dense>? inputBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense>? outputBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense>? workBuffer;

    [Params(1024, 4096, 16384, 65536, 262144)]
    public int DataSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        try
        {
            context = Context.CreateDefault();
            var device = context.GetPreferredDevice(preferCPU: false);
            
            if (device == null || device.AcceleratorType == AcceleratorType.CPU)
                throw new NotSupportedException("GPU-only benchmarks require a non-CPU accelerator");
                
            accelerator = device.CreateAccelerator(context);

            // Allocate GPU memory
            inputBuffer = accelerator.Allocate1D<float>(DataSize);
            outputBuffer = accelerator.Allocate1D<float>(DataSize);
            workBuffer = accelerator.Allocate1D<float>(DataSize);

            // Initialize with test data
            InitializeTestData();
        }
        catch (Exception ex)
        {
            throw new NotSupportedException($"Failed to initialize GPU-only environment: {ex.Message}", ex);
        }
    }

    private void InitializeTestData()
    {
        var testData = new float[DataSize];
        var random = new Random(42);
        
        for (int i = 0; i < DataSize; i++)
        {
            testData[i] = random.NextSingle() * 100.0f;
        }
        
        inputBuffer?.View.CopyFromCPU(testData);
    }

    [Benchmark(Baseline = true)]
    public void VectorAddition()
    {
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(VectorAddKernel);

        if (inputBuffer == null || outputBuffer == null)
            return;
            
        kernel(DataSize, inputBuffer.View, outputBuffer.View);
        accelerator!.Synchronize();
    }

    [Benchmark]
    public void VectorMultiplication()
    {
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(VectorMulKernel);

        if (inputBuffer == null || outputBuffer == null)
            return;
            
        kernel(DataSize, inputBuffer.View, outputBuffer.View);
        accelerator!.Synchronize();
    }

    [Benchmark]
    public void ParallelReduction()
    {
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(ReductionKernel);

        if (inputBuffer == null || workBuffer == null)
            return;
            
        kernel(DataSize, inputBuffer.View, workBuffer.View);
        accelerator!.Synchronize();
    }

    [Benchmark]
    public void GpuMemoryBandwidth()
    {
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
            MemoryBandwidthKernel);

        if (inputBuffer == null || outputBuffer == null || workBuffer == null)
            return;
            
        kernel(DataSize, inputBuffer.View, outputBuffer.View, workBuffer.View);
        accelerator!.Synchronize();
    }

    [Benchmark]
    public void ComplexComputation()
    {
        var kernel = accelerator!.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(ComplexComputeKernel);

        if (inputBuffer == null || outputBuffer == null)
            return;
            
        kernel(DataSize, inputBuffer.View, outputBuffer.View);
        accelerator!.Synchronize();
    }

    #region GPU Kernels

    private static void VectorAddKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
    {
        if (index >= input.Length)
            return;
            
        output[index] = input[index] + 1.0f;
    }

    private static void VectorMulKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
    {
        if (index >= input.Length)
            return;
            
        output[index] = input[index] * 2.0f;
    }

    private static void ReductionKernel(Index1D index, ArrayView<float> input, ArrayView<float> work)
    {
        if (index >= input.Length)
            return;
            
        // Comprehensive parallel reduction using GPU memory hierarchy
        var groupSize = Group.DimX;
        var localId = Group.IdxX;
        var groupId = Group.IdxY;
        
        // Real shared memory simulation through group-local operations
        // Each thread loads data and performs local reduction
        float localSum = 0.0f;
        
        // Each thread processes multiple elements for better efficiency
        const int elementsPerThread = 4;
        for (int i = 0; i < elementsPerThread; i++)
        {
            var globalIdx = index + i * (groupSize * Group.DimY);
            if (globalIdx < input.Length)
            {
                localSum += input[globalIdx];
            }
        }
        
        // Butterfly reduction pattern within warp/group
        // Simulate shared memory by using global memory with group coordination
        var tempIdx = groupId * groupSize + localId;
        if (tempIdx < work.Length)
        {
            work[tempIdx] = localSum;
        }
        
        // Group-level barrier simulation
        Group.Barrier();
        
        // Tree reduction within group
        for (int stride = groupSize / 2; stride > 0; stride /= 2)
        {
            if (localId < stride && tempIdx < work.Length && (tempIdx + stride) < work.Length)
            {
                work[tempIdx] += work[tempIdx + stride];
            }
            Group.Barrier();
        }
        
        // Final result stored by first thread of each group
        if (localId == 0 && tempIdx < work.Length)
        {
            // Apply additional operations to simulate complex reduction
            var result = work[tempIdx];
            
            // Variance computation as part of reduction
            float variance = 0.0f;
            for (int i = 0; i < elementsPerThread && (index + i) < input.Length; i++)
            {
                var diff = input[index + i] - (result / elementsPerThread);
                variance += diff * diff;
            }
            
            work[tempIdx] = result + variance * 0.01f; // Include variance in final result
        }
    }

    private static void MemoryBandwidthKernel(
        Index1D index, 
        ArrayView<float> input, 
        ArrayView<float> output, 
        ArrayView<float> work)
    {
        if (index >= input.Length)
            return;
            
        // Memory-intensive operations to test bandwidth
        var temp = input[index];
        work[index] = temp * 0.5f;
        output[index] = work[index] + temp;
    }

    private static void ComplexComputeKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
    {
        if (index >= input.Length)
            return;
            
        var value = input[index];
        
        // Complex mathematical operations
        var result = 0.0f;
        for (int i = 0; i < 10; i++)
        {
            result += SinApprox(value + i) * CosApprox(value * i);
            result += SqrtApprox(IntrinsicMath.Abs(value + result));
        }
        
        output[index] = result;
    }

    #endregion

    #region Math Approximations

    private static float SinApprox(float x)
    {
        // Fast sine approximation using Taylor series
        x = x - ((int)(x / (2.0f * 3.14159f))) * (2.0f * 3.14159f);
        if (x > 3.14159f) x -= 2.0f * 3.14159f;
        if (x < -3.14159f) x += 2.0f * 3.14159f;
        
        var x2 = x * x;
        return x * (1.0f - x2 / 6.0f + x2 * x2 / 120.0f);
    }

    private static float CosApprox(float x)
    {
        // Fast cosine approximation
        x = x - ((int)(x / (2.0f * 3.14159f))) * (2.0f * 3.14159f);
        if (x > 3.14159f) x -= 2.0f * 3.14159f;
        if (x < -3.14159f) x += 2.0f * 3.14159f;
        
        var x2 = x * x;
        return 1.0f - x2 / 2.0f + x2 * x2 / 24.0f;
    }

    private static float SqrtApprox(float x)
    {
        // Fast square root approximation using Newton's method
        if (x <= 0.0f) return 0.0f;
        
        var guess = x;
        for (int i = 0; i < 3; i++) // Limited iterations for speed
        {
            guess = 0.5f * (guess + x / guess);
        }
        return guess;
    }

    #endregion

    [GlobalCleanup]
    public void Cleanup()
    {
        Dispose();
    }

    public void Dispose()
    {
        inputBuffer?.Dispose();
        outputBuffer?.Dispose();
        workBuffer?.Dispose();
        accelerator?.Dispose();
        context?.Dispose();
    }
}