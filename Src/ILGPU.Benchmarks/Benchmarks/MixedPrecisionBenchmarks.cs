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

using BenchmarkDotNet.Attributes;
using ILGPU.Numerics;
using ILGPU.Runtime;
using ILGPU.TensorCores;

namespace ILGPU.Benchmarks.Benchmarks;

/// <summary>
/// Benchmarks for mixed precision operations (FP16, BF16, TF32, INT8).
/// </summary>
[MemoryDiagnoser]
[SimpleJob]
public class MixedPrecisionBenchmarks : IDisposable
{
    private Context? context;
    private Accelerator? accelerator;

    [Params(128, 256, 512)]
    public int MatrixSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        try
        {
            // Try to use existing context or create new one
            context = SharedBenchmarkContext.GetOrCreateContext();
            var device = context.GetPreferredDevice(preferCPU: false); // GPU preferred, CPU fallback
            accelerator = device?.CreateAccelerator(context);
        }
        catch
        {
            try
            {
                context = SharedBenchmarkContext.GetOrCreateContext();
                var device = context.GetPreferredDevice(preferCPU: true);
                accelerator = device?.CreateAccelerator(context);
            }
            catch
            {
                // Last resort - skip this benchmark
                context = null;
                accelerator = null;
            }
        }
    }

    [Benchmark]
    public float FP16ToFP32Conversion()
    {
        if (accelerator == null) return 0f;
        
        var size = MatrixSize * MatrixSize;
        using var fp16Buffer = accelerator.Allocate1D<Half>(size);
        using var fp32Buffer = accelerator.Allocate1D<float>(size);

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<Half>, ArrayView<float>>(
            FP16ToFP32Kernel);

        kernel(size, fp16Buffer.View, fp32Buffer.View);
        accelerator.Synchronize();
        
        // Return a sample value
        var result = fp32Buffer.GetAsArray1D();
        return result.Length > 0 ? result[0] : 0f;
    }

    [Benchmark]
    public float BF16Operations()
    {
        var size = MatrixSize * MatrixSize;
        var bf16Data = new BFloat16[size];
        var fp32Result = new float[size];

        var random = new Random(42);
        for (int i = 0; i < size; i++)
        {
            bf16Data[i] = BFloat16.FromFloat(random.NextSingle());
        }

        for (int i = 0; i < size; i++)
        {
            fp32Result[i] = bf16Data[i].ToFloat() * 2.0f;
        }
        
        return fp32Result.Length > 0 ? fp32Result[0] : 0f;
    }

    [Benchmark]
    public float MixedPrecisionGEMM()
    {
        if (accelerator == null) return 0f;
        
        try
        {
            using var matrixA = accelerator.Allocate1D<Half>(MatrixSize * MatrixSize);
            using var matrixB = accelerator.Allocate1D<Half>(MatrixSize * MatrixSize);
            using var result = accelerator.Allocate1D<float>(MatrixSize * MatrixSize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<Half>, ArrayView<Half>, ArrayView<float>, int>(
                MixedPrecisionGEMMKernel);

            kernel(new Index2D(MatrixSize, MatrixSize),
                matrixA.View, matrixB.View, result.View, MatrixSize);
            accelerator.Synchronize();
            
            var output = result.GetAsArray1D();
            return output.Length > 0 ? output[0] : 0f;
        }
        catch
        {
            // Fallback operation
            return 0f;
        }
    }

    [Benchmark]
    public float QuantizedOperations()
    {
        var size = MatrixSize * MatrixSize;
        var int8Data = new sbyte[size];
        var fp32Result = new float[size];

        var random = new Random(42);
        for (int i = 0; i < size; i++)
        {
            int8Data[i] = (sbyte)(random.Next(-128, 128));
        }

        // Simulate quantized to float conversion
        const float scale = 1.0f / 127.0f;
        for (int i = 0; i < size; i++)
        {
            fp32Result[i] = int8Data[i] * scale;
        }
        
        return fp32Result.Length > 0 ? fp32Result[0] : 0f;
    }

    #region Kernels

    private static void FP16ToFP32Kernel(
        Index1D index,
        ArrayView<Half> input,
        ArrayView<float> output)
    {
        output[index] = (float)input[index];
    }

    private static void MixedPrecisionGEMMKernel(
        Index2D index,
        ArrayView<Half> matrixA,
        ArrayView<Half> matrixB,
        ArrayView<float> result,
        int size)
    {
        if (index.X >= size || index.Y >= size)
            return;

        float sum = 0.0f;
        for (int k = 0; k < size; k++)
        {
            var a = (float)matrixA[index.X * size + k];
            var b = (float)matrixB[k * size + index.Y];
            sum += a * b;
        }

        result[index.X * size + index.Y] = sum;
    }

    #endregion

    [GlobalCleanup]
    public void Cleanup()
    {
        Dispose();
    }

    public void Dispose()
    {
        accelerator?.Dispose();
        context?.Dispose();
    }
}
