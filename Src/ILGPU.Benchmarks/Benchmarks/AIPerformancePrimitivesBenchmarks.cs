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
using ILGPU.Numerics.AI;
using ILGPU.Runtime;
using ILGPU.Runtime.AI;

namespace ILGPU.Benchmarks.Benchmarks;

/// <summary>
/// Benchmarks for AI performance primitives across different accelerator types.
/// </summary>
[MemoryDiagnoser]
[SimpleJob]
[WarmupCount(5)]
[IterationCount(10)]
public class AIPerformancePrimitivesBenchmarks : IDisposable
{
    private Context? context;
    private Accelerator? accelerator;
    private IPerformancePrimitives? primitives;
    private ITensor<float>? inputTensor;
    private ITensor<float>? weightTensor;
    private ITensor<float>? outputTensor;
    private ITensor<float>? queryTensor;
    private ITensor<float>? keyTensor;
    private ITensor<float>? valueTensor;

    [Params(128, 256, 512, 1024)]
    public int MatrixSize { get; set; }

    [Params(64, 128, 256)]
    public int BatchSize { get; set; }

    [Params(32, 64)]
    public int Channels { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        try
        {
            context = SharedBenchmarkContext.GetOrCreateContext();
            var device = context.GetPreferredDevice(preferCPU: false);
            accelerator = device?.CreateAccelerator(context);
            
            if (accelerator != null)
            {
                primitives = new GenericPerformancePrimitives(accelerator);
                
                // Setup tensors for different benchmarks
                var matrixShape = new TensorShape(MatrixSize, MatrixSize);
                var convInputShape = new TensorShape(BatchSize, Channels, 224, 224);
                var convWeightShape = new TensorShape(Channels * 2, Channels, 3, 3);
                var convOutputShape = new TensorShape(BatchSize, Channels * 2, 222, 222);
                
                inputTensor = TensorFactory.Create<float>(convInputShape, ComputeLocation.Gpu);
                weightTensor = TensorFactory.Create<float>(convWeightShape, ComputeLocation.Gpu);
                outputTensor = TensorFactory.Create<float>(convOutputShape, ComputeLocation.Gpu);
                
                // Attention tensors
                var seqLen = 128;
                var hiddenSize = 512;
                var attentionShape = new TensorShape(BatchSize, seqLen, hiddenSize);
                queryTensor = TensorFactory.Create<float>(attentionShape, ComputeLocation.Gpu);
                keyTensor = TensorFactory.Create<float>(attentionShape, ComputeLocation.Gpu);
                valueTensor = TensorFactory.Create<float>(attentionShape, ComputeLocation.Gpu);
                
                InitializeTensors();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Setup failed: {ex.Message}");
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        Dispose();
    }

    [Benchmark]
    public async Task<double> GEMM_MatrixMultiplication()
    {
        if (primitives == null || inputTensor == null || weightTensor == null || outputTensor == null)
        {
            return 0.0;
        }

        var start = DateTime.UtcNow;
        await primitives.GemmAsync(inputTensor, weightTensor, outputTensor, 1.0f, 0.0f);
        var elapsed = DateTime.UtcNow - start;
        
        // Calculate GFLOPS (2 * M * N * K operations)
        long ops = 2L * MatrixSize * MatrixSize * MatrixSize;
        return ops / elapsed.TotalSeconds / 1e9;
    }

    [Benchmark]
    public async Task<double> BatchedGEMM_Performance()
    {
        if (primitives == null || inputTensor == null || weightTensor == null || outputTensor == null)
        {
            return 0.0;
        }

        var start = DateTime.UtcNow;
        await primitives.BatchedGemmAsync(inputTensor, weightTensor, outputTensor);
        var elapsed = DateTime.UtcNow - start;
        
        // Calculate GFLOPS for batched operations
        long ops = 2L * BatchSize * MatrixSize * MatrixSize * MatrixSize;
        return ops / elapsed.TotalSeconds / 1e9;
    }

    [Benchmark]
    public async Task<double> Conv2D_Performance()
    {
        if (primitives == null || inputTensor == null || weightTensor == null || outputTensor == null)
        {
            return 0.0;
        }

        var parameters = new ConvolutionParameters
        {
            Stride = new Size2D(1, 1),
            Padding = new Size2D(1, 1),
            Dilation = new Size2D(1, 1)
        };

        var start = DateTime.UtcNow;
        await primitives.Conv2DAsync(inputTensor, weightTensor, outputTensor, parameters);
        var elapsed = DateTime.UtcNow - start;
        
        // Calculate GFLOPS for convolution
        long ops = CalculateConvolutionOps(inputTensor.Shape, weightTensor.Shape, parameters);
        return ops / elapsed.TotalSeconds / 1e9;
    }

    [Benchmark]
    public async Task<double> MultiHeadAttention_Performance()
    {
        if (primitives == null || queryTensor == null || keyTensor == null || valueTensor == null)
        {
            return 0.0;
        }

        var parameters = new AttentionParameters
        {
            NumHeads = 8
        };

        var start = DateTime.UtcNow;
        await primitives.MultiHeadAttentionAsync(queryTensor, keyTensor, valueTensor, queryTensor, parameters);
        var elapsed = DateTime.UtcNow - start;
        
        // Calculate GFLOPS for attention (approximate)
        var seqLen = queryTensor.Shape[1];
        var hiddenSize = queryTensor.Shape[2];
        long ops = 4L * BatchSize * parameters.NumHeads * seqLen * seqLen * (hiddenSize / parameters.NumHeads);
        return ops / elapsed.TotalSeconds / 1e9;
    }

    [Benchmark]
    public async Task<double> ScaledDotProductAttention_Performance()
    {
        if (primitives == null || queryTensor == null || keyTensor == null || valueTensor == null)
        {
            return 0.0;
        }

        var start = DateTime.UtcNow;
        await primitives.ScaledDotProductAttentionAsync(queryTensor, keyTensor, valueTensor, queryTensor, 1.0f);
        var elapsed = DateTime.UtcNow - start;
        
        // Calculate GFLOPS for scaled dot-product attention
        var seqLen = queryTensor.Shape[1];
        var hiddenSize = queryTensor.Shape[2];
        long ops = 3L * BatchSize * seqLen * seqLen * hiddenSize;
        return ops / elapsed.TotalSeconds / 1e9;
    }

    [Benchmark]
    public async Task<double> GELU_Activation_Performance()
    {
        if (primitives == null || inputTensor == null || outputTensor == null)
        {
            return 0.0;
        }

        var start = DateTime.UtcNow;
        await primitives.GELUAsync(inputTensor, outputTensor);
        var elapsed = DateTime.UtcNow - start;
        
        // Calculate GFLOPS for GELU (approximately 8 ops per element)
        long ops = 8L * inputTensor.Shape.Length;
        return ops / elapsed.TotalSeconds / 1e9;
    }

    [Benchmark]
    public async Task<double> LayerNorm_Performance()
    {
        if (primitives == null || inputTensor == null || outputTensor == null)
        {
            return 0.0;
        }

        var lastDim = inputTensor.Shape[inputTensor.Shape.Rank - 1];
        var gamma = TensorFactory.Create<float>(new TensorShape(lastDim), ComputeLocation.Gpu);
        var beta = TensorFactory.Create<float>(new TensorShape(lastDim), ComputeLocation.Gpu);

        var start = DateTime.UtcNow;
        await primitives.LayerNormAsync(inputTensor, outputTensor, gamma, beta, 1e-5f);
        var elapsed = DateTime.UtcNow - start;
        
        // Calculate GFLOPS for layer normalization (approximately 5 ops per element)
        long ops = 5L * inputTensor.Shape.Length;
        return ops / elapsed.TotalSeconds / 1e9;
    }

    [Benchmark]
    public async Task<double> MaxPool2D_Performance()
    {
        if (primitives == null || inputTensor == null || outputTensor == null)
        {
            return 0.0;
        }

        var poolSize = new Size2D(2, 2);
        var stride = new Size2D(2, 2);
        var padding = new Size2D(0, 0);

        var start = DateTime.UtcNow;
        await primitives.MaxPool2DAsync(inputTensor, outputTensor, poolSize, stride, padding);
        var elapsed = DateTime.UtcNow - start;
        
        // Calculate throughput (elements processed per second)
        return inputTensor.Shape.Length / elapsed.TotalSeconds / 1e9;
    }

    [Benchmark]
    public async Task<double> AvgPool2D_Performance()
    {
        if (primitives == null || inputTensor == null || outputTensor == null)
        {
            return 0.0;
        }

        var poolSize = new Size2D(2, 2);
        var stride = new Size2D(2, 2);
        var padding = new Size2D(0, 0);

        var start = DateTime.UtcNow;
        await primitives.AvgPool2DAsync(inputTensor, outputTensor, poolSize, stride, padding);
        var elapsed = DateTime.UtcNow - start;
        
        // Calculate throughput (elements processed per second)
        return inputTensor.Shape.Length / elapsed.TotalSeconds / 1e9;
    }

    [Benchmark]
    public Task<double> MemoryBandwidth_CopyAsync()
    {
        if (primitives == null || inputTensor == null || outputTensor == null)
        {
            return Task.FromResult(0.0);
        }

        var start = DateTime.UtcNow;
        inputTensor.CopyTo(outputTensor);
        var elapsed = DateTime.UtcNow - start;
        
        // Calculate memory bandwidth (GB/s)
        long bytes = inputTensor.Shape.Length * sizeof(float) * 2; // Read + Write
        return Task.FromResult(bytes / elapsed.TotalSeconds / 1e9);
    }

    private void InitializeTensors()
    {
        // Initialize tensors with random data for benchmarking
        var random = new Random(42);
        
        if (inputTensor != null)
        {
            for (int i = 0; i < inputTensor.Length; i++)
            {
                var indices = ComputeIndicesFromFlat(inputTensor.Shape, i);
                inputTensor[indices] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
        }
        
        if (weightTensor != null)
        {
            for (int i = 0; i < weightTensor.Length; i++)
            {
                var indices = ComputeIndicesFromFlat(weightTensor.Shape, i);
                weightTensor[indices] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
        }
        
        if (queryTensor != null)
        {
            for (int i = 0; i < queryTensor.Length; i++)
            {
                var indices = ComputeIndicesFromFlat(queryTensor.Shape, i);
                queryTensor[indices] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
        }
        
        if (keyTensor != null)
        {
            for (int i = 0; i < keyTensor.Length; i++)
            {
                var indices = ComputeIndicesFromFlat(keyTensor.Shape, i);
                keyTensor[indices] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
        }
        
        if (valueTensor != null)
        {
            for (int i = 0; i < valueTensor.Length; i++)
            {
                var indices = ComputeIndicesFromFlat(valueTensor.Shape, i);
                valueTensor[indices] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
        }
    }

    private static long CalculateConvolutionOps(TensorShape inputShape, TensorShape weightShape, ConvolutionParameters parameters)
    {
        // Simplified convolution operation count
        var batchSize = inputShape[0];
        var outputChannels = weightShape[0];
        var inputChannels = weightShape[1];
        var kernelHeight = weightShape[2];
        var kernelWidth = weightShape[3];
        var inputHeight = inputShape[2];
        var inputWidth = inputShape[3];
        
        // Calculate output dimensions
        var outputHeight = (inputHeight + 2 * parameters.Padding.Height - kernelHeight) / parameters.Stride.Height + 1;
        var outputWidth = (inputWidth + 2 * parameters.Padding.Width - kernelWidth) / parameters.Stride.Width + 1;
        
        // 2 operations per MAC (multiply-accumulate)
        return 2L * batchSize * outputChannels * outputHeight * outputWidth * inputChannels * kernelHeight * kernelWidth;
    }

    public void Dispose()
    {
        inputTensor?.Dispose();
        weightTensor?.Dispose();
        outputTensor?.Dispose();
        queryTensor?.Dispose();
        keyTensor?.Dispose();
        valueTensor?.Dispose();
        accelerator?.Dispose();
        // Don't dispose context - it's managed by SharedBenchmarkContext
    }

    private static int[] ComputeIndicesFromFlat(ILGPU.Numerics.TensorShape shape, long flatIndex)
    {
        var indices = new int[shape.Rank];
        for (int i = shape.Rank - 1; i >= 0; i--)
        {
            indices[i] = (int)(flatIndex % shape[i]);
            flatIndex /= shape[i];
        }
        return indices;
    }
}
