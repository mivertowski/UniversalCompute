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

using ILGPU.Apple.NeuralEngine;
using ILGPU.Intel.AMX;
using ILGPU.Intel.NPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests.Hardware
{
    /// <summary>
    /// Comprehensive benchmark integration tests for hardware accelerators.
    /// Tests performance, memory bandwidth, and real-world workloads.
    /// </summary>
    public class BenchmarkIntegrationTests : TestBase
    {
        public BenchmarkIntegrationTests(ITestOutputHelper output, TestContext testContext)
            : base(output, testContext)
        {
        }

        [Fact]
        public void CrossAcceleratorPerformanceComparison()
        {
            using var context = Context.Create();
            var accelerators = new List<(string Name, Accelerator Accelerator)>();
            
            // Collect all available hardware accelerators
            try
            {
                if (context.GetAMXDevices().Any())
                    accelerators.Add(("AMX", context.CreateAMXAccelerator(0)));
            }
            catch { }
            
            try
            {
                if (context.GetNPUDevices().Any())
                    accelerators.Add(("NPU", context.CreateNPUAccelerator(0)));
            }
            catch { }
            
            try
            {
                if (context.GetANEDevices().Any())
                    accelerators.Add(("ANE", context.CreateANEAccelerator(0)));
            }
            catch { }
            
            if (!accelerators.Any())
            {
                Output.WriteLine("No hardware accelerators available - skipping benchmark");
                return;
            }
            
            Output.WriteLine($"Benchmarking {accelerators.Count} hardware accelerators:");
            
            const int matrixSize = 512;
            var results = new Dictionary<string, double>();
            
            foreach (var (name, accelerator) in accelerators)
            {
                using (accelerator)
                using (var stream = accelerator.CreateStream())
                {
                    var gflops = BenchmarkMatrixMultiplication(accelerator, stream, matrixSize);
                    results[name] = gflops;
                    
                    Output.WriteLine($"  {name}: {gflops:F2} GFLOPS");
                }
            }
            
            // Find the best performer
            var best = results.OrderByDescending(kvp => kvp.Value).First();
            Output.WriteLine($"\nBest performer: {best.Key} with {best.Value:F2} GFLOPS");
            
            // All accelerators should achieve some reasonable performance
            foreach (var result in results.Values)
            {
                Assert.True(result > 0.1, "Each accelerator should achieve at least 0.1 GFLOPS");
            }
        }

        [Fact]
        [KernelMethod(nameof(MatrixMultiplyBenchmarkKernel))]
        public void MemoryBandwidthBenchmark()
        {
            using var context = Context.Create();
            var accelerators = GetAvailableAccelerators(context);
            
            if (!accelerators.Any())
            {
                Output.WriteLine("No hardware accelerators available - skipping bandwidth test");
                return;
            }
            
            Output.WriteLine("Memory Bandwidth Benchmark:");
            
            var dataSizes = new[] { 1024, 4096, 16384, 65536 }; // Elements
            
            foreach (var (name, accelerator) in accelerators)
            {
                using (accelerator)
                using (var stream = accelerator.CreateStream())
                {
                    Output.WriteLine($"\n{name} Memory Bandwidth:");
                    
                    foreach (var size in dataSizes)
                    {
                        var bandwidth = BenchmarkMemoryBandwidth(accelerator, stream, size);
                        Output.WriteLine($"  {size:N0} elements: {bandwidth:F2} GB/s");
                        
                        Assert.True(bandwidth > 0, "Bandwidth should be positive");
                    }
                }
            }
        }

        [Fact]
        [KernelMethod(nameof(NeuralNetworkBenchmarkKernel))]
        public void NeuralNetworkWorkloadBenchmark()
        {
            using var context = Context.Create();
            var accelerators = GetAvailableAccelerators(context);
            
            if (!accelerators.Any())
            {
                Output.WriteLine("No hardware accelerators available - skipping NN benchmark");
                return;
            }
            
            Output.WriteLine("Neural Network Workload Benchmark:");
            
            // Typical neural network layer sizes
            var layerConfigs = new[]
            {
                (Input: 784, Output: 256, Name: "Dense Layer"),
                (Input: 2048, Output: 512, Name: "Transformer FFN"),
                (Input: 512, Output: 1000, Name: "Classification Head")
            };
            
            foreach (var (name, accelerator) in accelerators)
            {
                using (accelerator)
                using (var stream = accelerator.CreateStream())
                {
                    Output.WriteLine($"\n{name} Neural Network Performance:");
                    
                    foreach (var config in layerConfigs)
                    {
                        var throughput = BenchmarkNeuralNetworkLayer(
                            accelerator, stream, config.Input, config.Output);
                        
                        Output.WriteLine($"  {config.Name} ({config.Input}→{config.Output}): {throughput:F2} samples/sec");
                        Assert.True(throughput > 0, "Throughput should be positive");
                    }
                }
            }
        }

        [Fact]
        public void PowerEfficiencyBenchmark()
        {
            using var context = Context.Create();
            
            Output.WriteLine("Power Efficiency Comparison:");
            
            // Test AMX power efficiency
            if (AMXCapabilities.IsAMXSupported())
            {
                var amxCaps = AMXCapabilities.Query();
                var amxEfficiency = amxCaps.GetPowerEfficiency();
                Output.WriteLine($"  AMX: {amxEfficiency:F1} GOPS/Watt");
                Assert.True(amxEfficiency > 0);
            }
            
            // Test NPU power efficiency
            if (NPUCapabilities.DetectNPU())
            {
                var npuCaps = NPUCapabilities.Query();
                var npuTopsPerWatt = npuCaps.MaxTOPS / Math.Max(npuCaps.GetEstimatedPower(100), 0.1);
                Output.WriteLine($"  NPU: {npuTopsPerWatt:F1} TOPS/Watt");
                Assert.True(npuTopsPerWatt > 0);
            }
            
            // Test ANE power efficiency
            if (ANECapabilities.DetectNeuralEngine())
            {
                var aneCaps = ANECapabilities.Query();
                var aneEfficiency = aneCaps.GetPowerEfficiency();
                Output.WriteLine($"  ANE: {aneEfficiency:F1} TOPS/Watt");
                Assert.True(aneEfficiency > 0);
            }
        }

        [Fact]
        [KernelMethod(nameof(ComputeBoundBenchmarkKernel))]
        public void ComputeBoundVsMemoryBoundAnalysis()
        {
            using var context = Context.Create();
            var accelerators = GetAvailableAccelerators(context);
            
            if (!accelerators.Any())
            {
                Output.WriteLine("No hardware accelerators available - skipping analysis");
                return;
            }
            
            Output.WriteLine("Compute vs Memory Bound Analysis:");
            
            foreach (var (name, accelerator) in accelerators)
            {
                using (accelerator)
                using (var stream = accelerator.CreateStream())
                {
                    // Compute-bound workload (many operations per memory access)
                    var computeBound = BenchmarkComputeBound(accelerator, stream);
                    
                    // Memory-bound workload (few operations per memory access)
                    var memoryBound = BenchmarkMemoryBound(accelerator, stream);
                    
                    var ratio = computeBound / Math.Max(memoryBound, 0.01);
                    
                    Output.WriteLine($"\n{name}:");
                    Output.WriteLine($"  Compute-bound: {computeBound:F2} GFLOPS");
                    Output.WriteLine($"  Memory-bound: {memoryBound:F2} GFLOPS");
                    Output.WriteLine($"  Compute/Memory ratio: {ratio:F2}x");
                    
                    Assert.True(computeBound > 0);
                    Assert.True(memoryBound > 0);
                    Assert.True(ratio >= 1.0, "Compute-bound should be at least as fast as memory-bound");
                }
            }
        }

        [Fact]
        [KernelMethod(nameof(ScalabilityBenchmarkKernel))]
        public void ScalabilityBenchmark()
        {
            using var context = Context.Create();
            var accelerators = GetAvailableAccelerators(context);
            
            if (!accelerators.Any())
            {
                Output.WriteLine("No hardware accelerators available - skipping scalability test");
                return;
            }
            
            Output.WriteLine("Scalability Benchmark:");
            
            var problemSizes = new[] { 64, 128, 256, 512, 1024 };
            
            foreach (var (name, accelerator) in accelerators)
            {
                using (accelerator)
                using (var stream = accelerator.CreateStream())
                {
                    Output.WriteLine($"\n{name} Scalability:");
                    
                    var previousGflops = 0.0;
                    
                    foreach (var size in problemSizes)
                    {
                        var gflops = BenchmarkMatrixMultiplication(accelerator, stream, size);
                        var efficiency = previousGflops > 0 ? gflops / previousGflops : 1.0;
                        
                        Output.WriteLine($"  {size}x{size}: {gflops:F2} GFLOPS (efficiency: {efficiency:F2}x)");
                        
                        Assert.True(gflops > 0);
                        previousGflops = gflops;
                    }
                }
            }
        }

        #region Benchmark Kernels

        static void MatrixMultiplyBenchmarkKernel(
            Index2D index,
            ArrayView<float> a,
            ArrayView<float> b,
            ArrayView<float> c,
            int size)
        {
            var row = index.X;
            var col = index.Y;
            
            if (row >= size || col >= size)
                return;

            float sum = 0;
            for (int k = 0; k < size; k++)
                sum += a[row * size + k] * b[k * size + col];
            
            c[row * size + col] = sum;
        }

        static void NeuralNetworkBenchmarkKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> weights,
            ArrayView<float> bias,
            ArrayView<float> output,
            int inputSize,
            int outputSize)
        {
            var i = index.X;
            if (i >= outputSize)
                return;

            float sum = bias[i];
            for (int j = 0; j < inputSize; j++)
                sum += input[j] * weights[i * inputSize + j];
            
            output[i] = Math.Max(0.0f, sum); // ReLU
        }

        static void ComputeBoundBenchmarkKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int size)
        {
            var i = index.X;
            if (i >= size)
                return;

            var value = input[i];
            
            // Many compute operations per memory access
            for (int iter = 0; iter < 100; iter++)
            {
                value = Math.Sin(value) * Math.Cos(value);
                value = Math.Sqrt(Math.Abs(value));
                value = value * value + 0.001f;
            }
            
            output[i] = value;
        }

        static void MemoryBoundBenchmarkKernel(
            Index1D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int size)
        {
            var i = index.X;
            if (i >= size)
                return;

            // Simple copy operation (memory bound)
            output[i] = input[i] * 1.001f; // Minimal compute
        }

        static void ScalabilityBenchmarkKernel(
            Index2D index,
            ArrayView<float> input,
            ArrayView<float> output,
            int size)
        {
            var row = index.X;
            var col = index.Y;
            
            if (row >= size || col >= size)
                return;

            var idx = row * size + col;
            
            // Standard matrix element computation
            var value = input[idx];
            value = Math.Sin(value * 3.14159f) * Math.Cos(value * 2.71828f);
            output[idx] = value;
        }

        #endregion

        #region Helper Methods

        private List<(string Name, Accelerator Accelerator)> GetAvailableAccelerators(Context context)
        {
            var accelerators = new List<(string, Accelerator)>();
            
            try
            {
                if (context.GetAMXDevices().Any())
                    accelerators.Add(("AMX", context.CreateAMXAccelerator(0)));
            }
            catch { }
            
            try
            {
                if (context.GetNPUDevices().Any())
                    accelerators.Add(("NPU", context.CreateNPUAccelerator(0)));
            }
            catch { }
            
            try
            {
                if (context.GetANEDevices().Any())
                    accelerators.Add(("ANE", context.CreateANEAccelerator(0)));
            }
            catch { }
            
            return accelerators;
        }

        private double BenchmarkMatrixMultiplication(Accelerator accelerator, AcceleratorStream stream, int size)
        {
            var matrixA = new float[size * size];
            var matrixB = new float[size * size];
            
            var random = new Random(42);
            for (int i = 0; i < size * size; i++)
            {
                matrixA[i] = (float)random.NextDouble();
                matrixB[i] = (float)random.NextDouble();
            }

            using var bufferA = accelerator.Allocate1D<float>(matrixA);
            using var bufferB = accelerator.Allocate1D<float>(matrixB);
            using var bufferC = accelerator.Allocate1D<float>(size * size);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
                MatrixMultiplyBenchmarkKernel);

            // Warmup
            kernel(stream, new Index2D(size, size), bufferA.View, bufferB.View, bufferC.View, size);
            stream.Synchronize();

            // Benchmark
            var stopwatch = Stopwatch.StartNew();
            kernel(stream, new Index2D(size, size), bufferA.View, bufferB.View, bufferC.View, size);
            stream.Synchronize();
            stopwatch.Stop();

            var operations = 2.0 * size * size * size; // 2 ops per inner loop iteration
            return operations / (stopwatch.Elapsed.TotalSeconds * 1e9);
        }

        private double BenchmarkMemoryBandwidth(Accelerator accelerator, AcceleratorStream stream, int size)
        {
            var data = new float[size];
            var random = new Random(42);
            for (int i = 0; i < size; i++)
                data[i] = (float)random.NextDouble();

            using var buffer = accelerator.Allocate1D<float>(data);

            var stopwatch = Stopwatch.StartNew();
            var result = buffer.GetAsArray(stream);
            stream.Synchronize();
            stopwatch.Stop();

            var bytes = size * sizeof(float) * 2; // Read + Write
            return bytes / (stopwatch.Elapsed.TotalSeconds * 1e9);
        }

        private double BenchmarkNeuralNetworkLayer(Accelerator accelerator, AcceleratorStream stream, int inputSize, int outputSize)
        {
            var input = new float[inputSize];
            var weights = new float[outputSize * inputSize];
            var bias = new float[outputSize];
            
            var random = new Random(42);
            for (int i = 0; i < inputSize; i++)
                input[i] = (float)random.NextDouble();
            for (int i = 0; i < weights.Length; i++)
                weights[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            for (int i = 0; i < outputSize; i++)
                bias[i] = (float)(random.NextDouble() * 0.1);

            using var inputBuffer = accelerator.Allocate1D<float>(input);
            using var weightsBuffer = accelerator.Allocate1D<float>(weights);
            using var biasBuffer = accelerator.Allocate1D<float>(bias);
            using var outputBuffer = accelerator.Allocate1D<float>(outputSize);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
                NeuralNetworkBenchmarkKernel);

            // Warmup
            kernel(stream, outputSize, inputBuffer.View, weightsBuffer.View, biasBuffer.View, 
                   outputBuffer.View, inputSize, outputSize);
            stream.Synchronize();

            // Benchmark multiple iterations
            const int iterations = 10;
            var stopwatch = Stopwatch.StartNew();
            
            for (int iter = 0; iter < iterations; iter++)
            {
                kernel(stream, outputSize, inputBuffer.View, weightsBuffer.View, biasBuffer.View, 
                       outputBuffer.View, inputSize, outputSize);
            }
            
            stream.Synchronize();
            stopwatch.Stop();

            return iterations / stopwatch.Elapsed.TotalSeconds;
        }

        private double BenchmarkComputeBound(Accelerator accelerator, AcceleratorStream stream)
        {
            const int size = 8192;
            var input = new float[size];
            
            var random = new Random(42);
            for (int i = 0; i < size; i++)
                input[i] = (float)random.NextDouble();

            using var inputBuffer = accelerator.Allocate1D<float>(input);
            using var outputBuffer = accelerator.Allocate1D<float>(size);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int>(
                ComputeBoundBenchmarkKernel);

            var stopwatch = Stopwatch.StartNew();
            kernel(stream, size, inputBuffer.View, outputBuffer.View, size);
            stream.Synchronize();
            stopwatch.Stop();

            var operations = size * 100 * 6; // 100 iterations * ~6 ops per iteration
            return operations / (stopwatch.Elapsed.TotalSeconds * 1e9);
        }

        private double BenchmarkMemoryBound(Accelerator accelerator, AcceleratorStream stream)
        {
            const int size = 1024 * 1024; // Large for memory bandwidth
            var input = new float[size];
            
            var random = new Random(42);
            for (int i = 0; i < size; i++)
                input[i] = (float)random.NextDouble();

            using var inputBuffer = accelerator.Allocate1D<float>(input);
            using var outputBuffer = accelerator.Allocate1D<float>(size);

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int>(
                MemoryBoundBenchmarkKernel);

            var stopwatch = Stopwatch.StartNew();
            kernel(stream, size, inputBuffer.View, outputBuffer.View, size);
            stream.Synchronize();
            stopwatch.Stop();

            var operations = size * 1; // 1 multiply operation per element
            return operations / (stopwatch.Elapsed.TotalSeconds * 1e9);
        }

        #endregion
    }
}